"""§6-§13 corrected critic development. Standard composite critic (scale-correct smoothing) audited with ACTOR-
RELEVANT metrics — Q1/Q2/min-Q separate, CENTERED dQ vs empirical dG, margin-aware ranking, and the load-bearing
empirical +gradQ1 vs -gradQ1 test. If the standard critic fails development, an ADVANTAGE critic that directly
regresses the return marginal A(s,delta)~=G(s,delta)-G(s,0) from paired counterfactuals is trained and audited on
fresh disjoint panels. Ground truth = deterministic counterfactual continuation returns. No actor update.
"""
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_residual_controller import ZeroInitResidualActor  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_critic import CompositeTwinCritic, encode_controller_states, encoder_fingerprint  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_replay import ReplayControllerStateV2, residual_target_action  # noqa: E402
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.coin_stable_engagement import StableEngagementConfig, StableEngagementGate, stable_engagement_signals  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.env.planar_snapshot import restore_planar, snapshot_planar  # noqa: E402

PI0 = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/critic_dev.json"
GAMMA, TAU, BOUND = 0.99, 0.005, 0.25
CENTER_TOL, ENTRY_TOL = 0.02, 0.05
CF_H = 40
# disjoint seed banks
STD_TRAIN = list(range(6000, 6060)); STD_DEV = list(range(7000, 7032)); STD_SEALED = list(range(7060, 7076))
ADV_TRAIN = list(range(6100, 6160)); ADV_DEV = list(range(7100, 7132)); ADV_SEALED = list(range(7140, 7156))


def base_action(pi0, o):
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.tensor(np.asarray(o, np.float32)[None]))[0].numpy(), -4, 4)


def snap_rl(rl):
    return (snapshot_planar(rl.inner), rl._t, rl._strict, rl._touched)


def restore_rl(rl, s):
    restore_planar(rl.inner, s[0]); rl._t, rl._strict, rl._touched = s[1], s[2], s[3]


def family_of(dtz, lc, rc):
    if dtz <= CENTER_TOL:
        return "settling"
    if lc != rc:
        return "contact_retention"
    return "transport" if dtz > ENTRY_TOL else "entry"


def cf_return(rl, pi0, snap, cand, K=CF_H):
    restore_rl(rl, snap); tot = 0.0; disc = 1.0
    o2, r, term, trunc, _ = rl.step(np.asarray(cand, np.float32)); tot += r; o = o2
    for _k in range(K - 1):
        if term or trunc:
            break
        o2, r, term, trunc, _ = rl.step(base_action(pi0, o).astype(np.float32)); disc *= GAMMA; tot += disc * r; o = o2
    return tot


def candidates(base, rng):
    dirs = {"toward": np.array([1, 1, -1, -1.]), "away": np.array([-1, -1, 1, 1.]),
            "orth": np.array([1, -1, 1, -1.]), "rand": rng.standard_normal(4)}
    cs = [("zero", base.copy())]
    for nm, d in dirs.items():
        u = d / (np.linalg.norm(d) + 1e-9)
        for sc in (0.1, 0.25):
            cs.append((f"{nm}{sc}", np.clip(base + BOUND * sc * u, -4, 4)))
    return [(n, a.astype(np.float32)) for n, a in cs]


def build_panel(pi0, seeds, per_family=8):
    """Collect gate-active states by family + snapshot; ground-truth candidate returns by counterfactual rollout."""
    fam = {"transport": [], "entry": [], "settling": [], "contact_retention": []}
    rl = CoinRL4Dof()
    for s in seeds:
        if all(len(v) >= per_family for v in fam.values()):
            break
        o = rl.reset(int(s)); gate = StableEngagementGate(StableEngagementConfig()); seen = {f: 0 for f in fam}
        for _t in range(360):
            b = base_action(pi0, o); m = rl.inner._planar_metrics; dtz = float(m.disk_to_zone)
            lc, rc = bool(m.left_contact), bool(m.right_contact)
            if gate.gate == 1.0:
                f = family_of(dtz, lc, rc); seen[f] += 1
                if len(fam[f]) < per_family and seen[f] in (2, 6):
                    fam[f].append({"seed": int(s), "obs": o.astype(np.float32), "base": b.astype(np.float32),
                                   "snap": snap_rl(rl), "cstate": ReplayControllerStateV2.from_gate(gate).to_dict(),
                                   "family": f})
            o2, r, term, trunc, _ = rl.step(b.astype(np.float32))
            lc2, rc2, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc2, rc2, coin, lt, rtp, terminated=bool(term)); o = o2
            if term or trunc:
                break
    # label candidates by counterfactual return
    for f, states in fam.items():
        for st in states:
            cs = candidates(st["base"], np.random.default_rng(st["seed"]))
            st["cand_names"] = [n for n, _a in cs]
            st["cand_actions"] = [a for _n, a in cs]
            st["cand_delta"] = [(a - st["base"]).astype(np.float32) for _n, a in cs]
            st["G"] = [cf_return(rl, pi0, st["snap"], a) for _n, a in cs]
            st["G0"] = st["G"][0]                                   # zero-residual reference return
    return fam


# ── standard critic (scale-correct smoothing target) ──
def collect_transitions(pi0, seeds, rng, noise=0.20):
    trs = []; rl = CoinRL4Dof()
    for s in seeds:
        o = rl.reset(int(s)); gate = StableEngagementGate(StableEngagementConfig())
        for _t in range(360):
            b = base_action(pi0, o); a = np.clip(b + rng.normal(0, noise, 4), -4, 4).astype(np.float32)
            cs_t = ReplayControllerStateV2.from_gate(gate).to_dict()
            o2, r, term, trunc, _ = rl.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            cs_tp1 = ReplayControllerStateV2.from_gate(gate).to_dict()
            trs.append((o.astype(np.float32), a, float(r), o2.astype(np.float32), float(term), cs_t, cs_tp1)); o = o2
            if term or trunc:
                break
    return trs


def train_std_critic(trs, pi0, residual0, steps, seed=0, lr=1e-4, clip=1.0):
    torch.manual_seed(seed)
    crit, targ = CompositeTwinCritic(), CompositeTwinCritic(); targ.load_state_dict(crit.state_dict())
    opt = torch.optim.Adam(crit.parameters(), lr=lr)
    O = torch.tensor(np.stack([t[0] for t in trs])); A = torch.tensor(np.stack([t[1] for t in trs]))
    R = torch.tensor(np.array([t[2] for t in trs], np.float32)); O2 = torch.tensor(np.stack([t[3] for t in trs]))
    D = torch.tensor(np.array([t[4] for t in trs], np.float32))
    E1 = encode_controller_states([t[5] for t in trs]); E2 = encode_controller_states([t[6] for t in trs])
    G2 = torch.tensor(np.array([float(t[6]["gate"]) for t in trs], np.float32))
    n = len(trs); rng = np.random.default_rng(seed); gen = torch.Generator().manual_seed(seed)
    ckpts = {}
    save_at = {0, 1000, 3000, 6000, 10000, 20000, 40000}
    for i in range(steps + 1):
        if i in save_at:
            c = CompositeTwinCritic(); c.load_state_dict(crit.state_dict()); ckpts[i] = c
        if i == steps:
            break
        idx = rng.integers(0, n, 256)
        with torch.no_grad():
            ta = residual_target_action(pi0, residual0, O2[idx], G2[idx], generator=gen)   # scale-correct smoothing
            tq = torch.min(*targ(O2[idx], ta, E2[idx]))
            y = R[idx] + GAMMA * (1 - D[idx]) * tq
        q1, q2 = crit(O[idx], A[idx], E1[idx])
        loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(crit.parameters(), clip); opt.step()
        with torch.no_grad():
            for p, pt in zip(crit.parameters(), targ.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
    return ckpts


def std_q(crit, obs, action, cstate, head):
    O = torch.tensor(np.asarray(obs, np.float32)[None], requires_grad=False)
    A = torch.tensor(np.asarray(action, np.float32)[None]); E = encode_controller_states([cstate])
    with torch.no_grad():
        q1, q2 = crit(O, A, E)
    return {"Q1": float(q1), "Q2": float(q2), "min": float(min(q1.item(), q2.item()))}[head]


def q1_grad_wrt_action(crit, obs, a0, cstate):
    O = torch.tensor(np.asarray(obs, np.float32)[None]); E = encode_controller_states([cstate])
    A = torch.tensor(np.asarray(a0, np.float32)[None], requires_grad=True)
    q1, _ = crit(O, A, E); q1.backward()
    return A.grad[0].numpy()


def audit_panel(fam, q_of, grad_of, rl, pi0, label):
    """Corrected dev metrics per family: centered dQ vs dG corr, margin-aware ranking, +gradQ1 vs -gradQ1 empirical."""
    res = {}
    for f, states in fam.items():
        if not states:
            res[f] = {"n": 0}; continue
        dq1_all, dq2_all, dg_all = [], [], []
        pair_ok = {1: [0, 0], 5: [0, 0], 10: [0, 0]}; allpair = [0, 0]
        grad_wins = [0, 0]; harmful_rej = []
        for st in states:
            G = np.array(st["G"]); G0 = st["G0"]
            q1 = [q_of(st, a, "Q1") for a in st["cand_actions"]]
            q2 = [q_of(st, a, "Q2") for a in st["cand_actions"]]
            dq1 = np.array(q1) - q1[0]; dq2 = np.array(q2) - q2[0]; dg = G - G0
            dq1_all += dq1.tolist(); dq2_all += dq2.tolist(); dg_all += dg.tolist()
            # margin-aware pairwise (on Q1, the actor-driving critic)
            m = len(G)
            for i in range(m):
                for j in range(i + 1, m):
                    if G[i] == G[j]:
                        continue
                    allpair[1] += 1; allpair[0] += int((q1[i] > q1[j]) == (G[i] > G[j]))
                    for thr in (1, 5, 10):
                        if abs(G[i] - G[j]) >= thr:
                            pair_ok[thr][1] += 1; pair_ok[thr][0] += int((q1[i] > q1[j]) == (G[i] > G[j]))
            harmful_rej.append(int(int(np.argmax(q1)) != int(np.argmin(G))))
            # §10 empirical +gradQ1 vs -gradQ1
            g = grad_of(st)
            if g is not None and np.linalg.norm(g) > 1e-9:
                u = g / np.linalg.norm(g); eps = 0.1 * BOUND
                a_plus = np.clip(st["base"] + eps * u, -4, 4).astype(np.float32)
                a_minus = np.clip(st["base"] - eps * u, -4, 4).astype(np.float32)
                Gp = cf_return(rl, pi0, st["snap"], a_plus); Gm = cf_return(rl, pi0, st["snap"], a_minus)
                if Gp != Gm:
                    grad_wins[1] += 1; grad_wins[0] += int(Gp > Gm)
        corr1 = float(np.corrcoef(dq1_all, dg_all)[0, 1]) if np.std(dq1_all) > 1e-9 and np.std(dg_all) > 1e-9 else 0.0
        corr2 = float(np.corrcoef(dq2_all, dg_all)[0, 1]) if np.std(dq2_all) > 1e-9 and np.std(dg_all) > 1e-9 else 0.0
        res[f] = {"n": len(states), "centered_corr_Q1_vs_dG": round(corr1, 3), "centered_corr_Q2_vs_dG": round(corr2, 3),
                  "allpair_acc": round(allpair[0] / max(allpair[1], 1), 3),
                  "acc_gap1": round(pair_ok[1][0] / max(pair_ok[1][1], 1), 3),
                  "acc_gap5": round(pair_ok[5][0] / max(pair_ok[5][1], 1), 3),
                  "acc_gap10": round(pair_ok[10][0] / max(pair_ok[10][1], 1), 3),
                  "harmful_rej": round(float(np.mean(harmful_rej)), 3),
                  "gradQ1_wins": round(grad_wins[0] / max(grad_wins[1], 1), 3), "grad_n": grad_wins[1]}
        print(f"    [{label}] {f:<18} n={len(states)} corrQ1/dG {res[f]['centered_corr_Q1_vs_dG']:+.2f} "
              f"gap5 {res[f]['acc_gap5']} +gradQ1wins {res[f]['gradQ1_wins']} (n={grad_wins[1]}) "
              f"harmful_rej {res[f]['harmful_rej']}", flush=True)
    return res


# ── advantage critic ──
class ResidualAdvantageCritic(nn.Module):
    def __init__(self, obs_dim=48, enc_dim=11, act_dim=4, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + enc_dim + act_dim, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, obs, enc, delta):
        z = self.net(torch.cat([obs, enc, torch.zeros_like(delta)], -1)).squeeze(-1)
        return self.net(torch.cat([obs, enc, delta], -1)).squeeze(-1) - z    # A(s,0)=0 by construction


def train_adv_critic(fam, steps=4000, seed=0):
    torch.manual_seed(seed)
    O, E, DE, DG = [], [], [], []
    for f, states in fam.items():
        for st in states:
            for a, dg in zip(st["cand_actions"], np.array(st["G"]) - st["G0"]):
                O.append(st["obs"]); E.append(st["cstate"]); DE.append((np.asarray(a) - st["base"]).astype(np.float32)); DG.append(float(dg))
    O = torch.tensor(np.stack(O)); Enc = encode_controller_states(E); DEt = torch.tensor(np.stack(DE)); DGt = torch.tensor(np.array(DG, np.float32))
    adv = ResidualAdvantageCritic(); opt = torch.optim.Adam(adv.parameters(), lr=1e-3)
    n = len(O); rng = np.random.default_rng(seed)
    for _i in range(steps):
        idx = rng.integers(0, n, min(256, n))
        pred = adv(O[idx], Enc[idx], DEt[idx])
        mse = ((pred - DGt[idx]) ** 2).mean()
        i2 = rng.integers(0, n, min(256, n))                       # pairwise ranking loss
        rank = torch.relu(-(torch.sign(DGt[idx] - DGt[i2])) * (pred - adv(O[i2], Enc[i2], DEt[i2])) + 0.0).mean()
        opt.zero_grad(); (mse + 0.5 * rank).backward(); opt.step()
    return adv


def main():
    file_sha = hashlib.sha256(open(PI0, "rb").read()).hexdigest()
    pi0 = load_frozen_clip_actor(PI0, freeze=True); residual0 = ZeroInitResidualActor()
    rng = np.random.default_rng(0); rl = CoinRL4Dof()
    out = {"pi0_sha": file_sha[:8], "encoder_fp": encoder_fingerprint()[:12], "seed_banks": {
        "std_train": [6000, 6059], "std_dev": [7000, 7031], "std_sealed": [7060, 7075],
        "adv_train": [6100, 6159], "adv_dev": [7100, 7131], "adv_sealed": [7140, 7155]}}
    disj = (not set(STD_TRAIN) & set(STD_DEV) and not set(STD_DEV) & set(STD_SEALED)
            and not set(ADV_TRAIN) & set(ADV_DEV) and not set(STD_TRAIN) & set(ADV_TRAIN))
    out["panels_disjoint"] = disj
    print(f"panels disjoint: {disj}", flush=True)

    # ── STANDARD critic dev (§7-§11) ──
    print("collecting std transitions + training standard critic (scale-correct smoothing)...", flush=True)
    trs = collect_transitions(pi0, STD_TRAIN, rng)
    ckpts = train_std_critic(trs, pi0, residual0, 40000)
    print(f"  transitions {len(trs)}; checkpoints {sorted(ckpts)}", flush=True)
    print("building std dev counterfactual panel...", flush=True)
    std_dev = build_panel(pi0, STD_DEV)
    std_ck_results = {}
    for k in sorted(ckpts):
        crit = ckpts[k]
        std_ck_results[k] = audit_panel(std_dev, lambda st, a, h, c=crit: std_q(c, st["obs"], a, st["cstate"], h),
                                        lambda st, c=crit: q1_grad_wrt_action(c, st["obs"], st["base"], st["cstate"]),
                                        rl, pi0, f"std@{k}")
    out["standard_dev"] = std_ck_results
    # §11 development gate: two consecutive ckpts with useful Q1 in transport+contact_retention + gradQ1 wins + harmful rej

    def ck_pass(r):
        t = r.get("transport", {}); cr = r.get("contact_retention", {})
        return (t.get("n", 0) > 0 and cr.get("n", 0) > 0
                and t.get("centered_corr_Q1_vs_dG", -1) > 0.2 and t.get("gradQ1_wins", 0) > 0.55
                and cr.get("gradQ1_wins", 0) > 0.55 and t.get("harmful_rej", 0) > 0.6)
    ks = sorted(ckpts); std_pass = any(ck_pass(std_ck_results[ks[i]]) and ck_pass(std_ck_results[ks[i + 1]])
                                       for i in range(len(ks) - 1))
    out["standard_development_pass"] = std_pass
    print(f"\nSTANDARD critic development pass: {std_pass}", flush=True)

    if std_pass:
        out["critic_route"] = "standard"; out["verdict"] = "PHASE_GATED_RESIDUAL_CRITIC_DEVELOPMENT_PASS"
    else:
        # ── ADVANTAGE critic fallback (§13) ──
        print("STANDARD failed development -> ADVANTAGE critic fallback (fresh disjoint panels)...", flush=True)
        adv_train = build_panel(pi0, ADV_TRAIN)
        adv = train_adv_critic(adv_train)
        print("building adv dev panel...", flush=True)
        adv_dev = build_panel(pi0, ADV_DEV)

        def adv_q(st, a, h, A=adv):
            O = torch.tensor(st["obs"][None]); E = encode_controller_states([st["cstate"]])
            de = torch.tensor((np.asarray(a) - st["base"]).astype(np.float32)[None])
            with torch.no_grad():
                v = float(A(O, E, de))
            return v                                              # A predicts dG directly (all heads identical)

        def adv_grad(st, A=adv):
            O = torch.tensor(st["obs"][None]); E = encode_controller_states([st["cstate"]])
            de = torch.zeros(1, 4, requires_grad=True)
            v = A(O, E, de); v.backward()
            return de.grad[0].numpy()
        adv_res = audit_panel(adv_dev, adv_q, adv_grad, rl, pi0, "adv")
        out["advantage_dev"] = adv_res
        adv_pass = ck_pass(adv_res)
        out["advantage_development_pass"] = adv_pass
        print(f"\nADVANTAGE critic development pass: {adv_pass}", flush=True)
        if adv_pass:
            out["critic_route"] = "advantage"; out["verdict"] = "RESIDUAL_ADVANTAGE_CRITIC_DEVELOPMENT_PASS"
        else:
            out["critic_route"] = "none"; out["verdict"] = "RESIDUAL_CRITIC_ROUTE_BLOCKED"

    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"\nroute={out.get('critic_route')} -> {out['verdict']}", flush=True)
    print("CRITIC_DEV_DONE", flush=True)


if __name__ == "__main__":
    main()
