"""§6 PHASE_GATED_RESIDUAL_CRITIC authorization. Train a composite-action twin critic (learns Q^{pi_0}(s,a) via
stored-gate_tp1 targets), build a residual-specific authorization panel over 4 behaviour families, GROUND-TRUTH the
candidate rankings with counterfactual rollouts (restore state -> apply candidate -> continue frozen pi_0 -> realized
discounted v3 return), and measure whether the critic's Q ordering matches. Includes the residual-magnitude/boundary
sweep. Disjoint train/authorization/final panels by seed. No actor update.
"""
import hashlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_residual_controller import ZeroInitResidualActor  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_critic import CompositeTwinCritic, encode_controller_states, encoder_fingerprint  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_replay import ReplayControllerStateV2, residual_target_action  # noqa: E402
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.coin_stable_engagement import StableEngagementConfig, StableEngagementGate, stable_engagement_signals  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402
from hymeko_rl.env.planar_snapshot import restore_planar, snapshot_planar  # noqa: E402

PI0 = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/critic_auth.json"
GAMMA, TAU, BOUND = 0.99, 0.005, 0.25
TRAIN_SEEDS = list(range(6000, 6060))
AUTH_SEEDS = list(range(7000, 7040))
FINAL_SEEDS = list(range(7060, 7076))
CENTER_TOL, ENTRY_TOL, SETTLE = 0.02, 0.05, 0.06
TRAIN_NOISE = 0.20          # matched near the residual bound (0.25); wide random noise inflated the target variance
CRITIC_STEPS = 30000
CRITIC_BATCH = 256
CRITIC_LR = 1e-4            # lower LR + grad clip to stabilize (twin disagreement was DIVERGING at 3e-4)
CRITIC_GRAD_CLIP = 1.0
CF_HORIZON = 40


def base_action(pi0, obs):
    with torch.no_grad():
        return np.clip(pi0.action_mean(torch.tensor(np.asarray(obs, np.float32)[None]))[0].numpy(), -4, 4)


def snap_rl(rl):
    return (snapshot_planar(rl.inner), rl._t, rl._strict, rl._touched)


def restore_rl(rl, snap):
    restore_planar(rl.inner, snap[0]); rl._t, rl._strict, rl._touched = snap[1], snap[2], snap[3]


# ── training-data collection: composite+gate rollouts with action noise (covers the residual neighborhood) ──
def collect_training(pi0, seeds, rng, noise=TRAIN_NOISE):
    trs = []
    for s in seeds:
        rl = CoinRL4Dof(); o = rl.reset(int(s))
        gate = StableEngagementGate(StableEngagementConfig())
        for _t in range(360):
            b = base_action(pi0, o)
            a = np.clip(b + rng.normal(0, noise, 4), -4, 4).astype(np.float32)   # wide perturbed composite
            cs_t = ReplayControllerStateV2.from_gate(gate).to_dict()
            o2, r, term, trunc, _ = rl.step(a)
            lc, rc, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc, rc, coin, lt, rtp, terminated=bool(term))
            cs_tp1 = ReplayControllerStateV2.from_gate(gate).to_dict()
            trs.append((o.astype(np.float32), a, float(r), o2.astype(np.float32), float(term), cs_t, cs_tp1))
            o = o2
            if term or trunc:
                break
    return trs


def train_critic(trs, pi0, residual0, steps, seed=0):
    torch.manual_seed(seed)
    crit, targ = CompositeTwinCritic(), CompositeTwinCritic(); targ.load_state_dict(crit.state_dict())
    opt = torch.optim.Adam(crit.parameters(), lr=CRITIC_LR)
    O = torch.tensor(np.stack([t[0] for t in trs])); A = torch.tensor(np.stack([t[1] for t in trs]))
    R = torch.tensor(np.array([t[2] for t in trs], np.float32)); O2 = torch.tensor(np.stack([t[3] for t in trs]))
    D = torch.tensor(np.array([t[4] for t in trs], np.float32))
    ES_t = encode_controller_states([t[5] for t in trs]); ES_2 = encode_controller_states([t[6] for t in trs])
    G2 = torch.tensor(np.array([float(t[6]["gate"]) for t in trs], np.float32))
    n = len(trs); rng = np.random.default_rng(seed)
    for _i in range(steps):
        idx = rng.integers(0, n, CRITIC_BATCH)
        with torch.no_grad():
            # DECLARED TD3 target policy smoothing (batch-independent, active stochastic noise) — corrected after the
            # TARGET_SMOOTHING_CONTRACT_MISMATCH audit (was hardcoded zeros = smoothing disabled). residual0==0 so the
            # target action = base + gate*clip(eps, -0.25, 0.25) at gate-active states.
            ta = residual_target_action(pi0, residual0, O2[idx], G2[idx])   # noise=None -> active declared smoothing
            tq = torch.min(*targ(O2[idx], ta, ES_2[idx]))
            y = R[idx] + GAMMA * (1 - D[idx]) * tq
        q1, q2 = crit(O[idx], A[idx], ES_t[idx])
        loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(crit.parameters(), CRITIC_GRAD_CLIP)
        opt.step()
        with torch.no_grad():
            for p, pt in zip(crit.parameters(), targ.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
    return crit


# ── panel: collect gate-active states by family, snapshot each ──
def family_of(dtz, strict, lc, rc):
    if dtz <= CENTER_TOL:             # near the zone — settling/dwell (regardless of contact symmetry)
        return "settling"
    if lc != rc:                      # single-side (fragile) contact away from zone — contact retention
        return "contact_retention"
    if dtz > ENTRY_TOL:
        return "transport"
    return "entry"


def collect_panel(pi0, seeds, rng, per_family=8):
    """Deterministic pi_0 rollouts; capture EVERY gate-active state bucketed by family; keep up to per_family each,
    spreading captures across the episode so entry/settling/contact (later, rarer) are not crowded out by transport."""
    fam = {"transport": [], "entry": [], "settling": [], "contact_retention": []}
    for s in seeds:
        if all(len(v) >= per_family for v in fam.values()):
            break
        rl = CoinRL4Dof(); o = rl.reset(int(s)); gate = StableEngagementGate(StableEngagementConfig())
        seen = {f: 0 for f in fam}
        for _t in range(360):
            b = base_action(pi0, o)
            m = rl.inner._planar_metrics; dtz = float(m.disk_to_zone)
            lc, rc = bool(m.left_contact), bool(m.right_contact)
            if gate.gate == 1.0:
                f = family_of(dtz, rl._strict, lc, rc)
                seen[f] += 1
                take = len(fam[f]) < per_family and seen[f] in (2, 6)   # up to 2 captures/episode, spread in time
                if take:
                    fam[f].append({"seed": int(s), "t": _t, "obs": o.astype(np.float32), "base": b.astype(np.float32),
                                   "snap": snap_rl(rl), "cstate": ReplayControllerStateV2.from_gate(gate).to_dict(),
                                   "dtz": dtz})
            o2, r, term, trunc, _ = rl.step(b.astype(np.float32))          # deterministic pi_0 (on-manifold states)
            lc2, rc2, coin, lt, rtp = stable_engagement_signals(rl.inner)
            gate.update(lc2, rc2, coin, lt, rtp, terminated=bool(term)); o = o2
            if term or trunc:
                break
    return fam


def candidates(base, rng):
    """Diverse composite-action candidates = base + residual perturbation (within/around the residual bound)."""
    dirs = {"toward": np.array([1, 1, -1, -1], float), "away": np.array([-1, -1, 1, 1], float),
            "orth": np.array([1, -1, 1, -1], float), "rand": rng.standard_normal(4)}
    cands = [("ref", base.copy())]
    for name, d in dirs.items():
        u = d / (np.linalg.norm(d) + 1e-9)
        for scale in (0.1, 0.25):
            cands.append((f"{name}+{scale}", np.clip(base + BOUND * scale / 0.25 * u * 0.25, -4, 4)))
        cands.append((f"{name}bound", np.clip(base + BOUND * u, -4, 4)))          # residual-boundary candidate
    return [(nm, a.astype(np.float32)) for nm, a in cands]


def counterfactual_return(rl, pi0, snap, cand, K=40):
    restore_rl(rl, snap)
    tot = 0.0; disc = 1.0
    o2, r, term, trunc, _ = rl.step(np.asarray(cand, np.float32))
    tot += r; o = o2
    for _k in range(K - 1):
        if term or trunc:
            break
        o2, r, term, trunc, _ = rl.step(base_action(pi0, o)); tot += (disc := disc * GAMMA) * r; o = o2
    return tot


def q_of(crit, obs, action, cstate):
    O = torch.tensor(np.asarray(obs, np.float32)[None]); A = torch.tensor(np.asarray(action, np.float32)[None])
    E = encode_controller_states([cstate])
    with torch.no_grad():
        q1, q2 = crit(O, A, E)
    return float(q1), float(q2), float(min(q1.item(), q2.item()))


def rank_metrics(pairs_true, pairs_q):
    """pairwise ranking accuracy of q vs true return over all distinct pairs."""
    n = len(pairs_true); ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pairs_true[i] == pairs_true[j]:
                continue
            tot += 1
            ok += int((pairs_q[i] > pairs_q[j]) == (pairs_true[i] > pairs_true[j]))
    return ok / max(tot, 1), tot


def main():
    file_sha = hashlib.sha256(open(PI0, "rb").read()).hexdigest()
    pi0 = load_frozen_clip_actor(PI0, freeze=True); residual0 = ZeroInitResidualActor()
    rng = np.random.default_rng(0)
    # panel disjointness
    assert not (set(TRAIN_SEEDS) & set(AUTH_SEEDS)) and not (set(AUTH_SEEDS) & set(FINAL_SEEDS)) and not (set(TRAIN_SEEDS) & set(FINAL_SEEDS))
    print("collecting training data...", flush=True)
    trs = collect_training(pi0, TRAIN_SEEDS, rng)
    print(f"train transitions {len(trs)}; training critic...", flush=True)
    crit = train_critic(trs, pi0, residual0, CRITIC_STEPS)
    print("collecting authorization panel + counterfactual labels...", flush=True)
    fam = collect_panel(pi0, AUTH_SEEDS, rng)
    rl = CoinRL4Dof()
    per_family = {}; all_states = []
    for f, states in fam.items():
        accs = []
        for st in states:
            cands = candidates(st["base"], np.random.default_rng(st["seed"]))
            true_r = [counterfactual_return(rl, pi0, st["snap"], a, K=CF_HORIZON) for _n, a in cands]
            qmin = [q_of(crit, st["obs"], a, st["cstate"])[2] for _n, a in cands]
            acc, npairs = rank_metrics(true_r, qmin)
            # top-1: does argmax Q match a top-quartile realized return?
            top_q = int(np.argmax(qmin)); order = np.argsort(true_r)
            top1 = int(top_q in set(order[-max(1, len(order) // 4):].tolist()))
            # harmful rejection: is the worst realized-return candidate NOT the critic's argmax?
            worst = int(np.argmin(true_r)); harmful_rej = int(top_q != worst)
            accs.append({"seed": st["seed"], "rank_acc": acc, "npairs": npairs, "top1": top1,
                         "harmful_rej": harmful_rej, "dtz": round(st["dtz"], 4),
                         "cand_names": [n for n, _a in cands], "true": [round(x, 3) for x in true_r],
                         "qmin": [round(x, 3) for x in qmin]})
            all_states.append((f, st, cands, true_r, qmin))
        macc = float(np.mean([a["rank_acc"] for a in accs])) if accs else float("nan")
        per_family[f] = {"n_states": len(accs), "mean_rank_acc": round(macc, 3),
                         "top1_rate": round(float(np.mean([a["top1"] for a in accs])), 3) if accs else 0.0,
                         "harmful_rej_rate": round(float(np.mean([a["harmful_rej"] for a in accs])), 3) if accs else 0.0,
                         "states": accs}
        print(f"  {f:<18} n={len(accs)} mean_rank_acc={macc:.3f} "
              f"harmful_rej={per_family[f]['harmful_rej_rate']}", flush=True)

    # ── §6.9 residual-boundary sweep: Q vs residual scale along direction/opposite/orthogonal ──
    scales = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    sweep = {"scales": scales, "corr_norm_vs_q": [], "boundary_pref_events": 0, "checked": 0}
    for f, st, cands, true_r, qmin in all_states:
        for dname, d in (("toward", np.array([1, 1, -1, -1.0])), ("orth", np.array([1, -1, 1, -1.0])),
                         ("rand", np.random.default_rng(st["seed"] + 1).standard_normal(4))):
            u = d / (np.linalg.norm(d) + 1e-9)
            qs = [q_of(crit, st["obs"], np.clip(st["base"] + BOUND * sc * u, -4, 4), st["cstate"])[2] for sc in scales]
            norms = [abs(sc) for sc in scales]
            sweep["corr_norm_vs_q"].append(float(np.corrcoef(norms, qs)[0, 1]) if np.std(qs) > 1e-9 else 0.0)
            # boundary preference = Q monotonically increasing toward |scale|=1 without realized support
            sweep["checked"] += 1
            if qs[-1] == max(qs) and qs[0] == sorted(qs)[-2] and qs[4] < qs[-1] and qs[4] < qs[0]:
                sweep["boundary_pref_events"] += 1
    mean_norm_corr = float(np.mean(sweep["corr_norm_vs_q"])) if sweep["corr_norm_vs_q"] else 0.0
    sweep["mean_abs_norm_vs_q_corr"] = round(mean_norm_corr, 3)
    sweep["boundary_pref_rate"] = round(sweep["boundary_pref_events"] / max(sweep["checked"], 1), 3)

    # twin disagreement on gate-active panel states
    tw = []
    for f, st, cands, true_r, qmin in all_states:
        q1, q2, _ = q_of(crit, st["obs"], st["base"], st["cstate"])
        tw.append(abs(q1 - q2))
    twin_disagree = round(float(np.mean(tw)), 4) if tw else float("nan")

    families_pass = {f: per_family[f]["mean_rank_acc"] > 0.55 and per_family[f]["harmful_rej_rate"] >= 0.5
                     for f in per_family}
    no_boundary_pref = sweep["boundary_pref_rate"] < 0.15 and abs(mean_norm_corr) < 0.5
    all_fam = all(families_pass.values())
    out = {"pi0_file_sha": file_sha[:8], "encoder_fingerprint": encoder_fingerprint()[:12],
           "critic_contract_sha": crit.contract_sha256()[:12],
           "panels": {"train": {"n": len(TRAIN_SEEDS), "sha": hashlib.sha256(json.dumps(TRAIN_SEEDS).encode()).hexdigest()[:12]},
                      "auth": {"n": len(AUTH_SEEDS), "sha": hashlib.sha256(json.dumps(AUTH_SEEDS).encode()).hexdigest()[:12]},
                      "final_sealed": {"n": len(FINAL_SEEDS), "sha": hashlib.sha256(json.dumps(FINAL_SEEDS).encode()).hexdigest()[:12]},
                      "disjoint": True},
           "per_family": {f: {k: v for k, v in per_family[f].items() if k != "states"} for f in per_family},
           "families_pass": families_pass, "boundary_sweep": {k: v for k, v in sweep.items() if k != "corr_norm_vs_q"},
           "twin_disagreement": twin_disagree, "no_boundary_preference": no_boundary_pref,
           "detail": {f: per_family[f]["states"] for f in per_family}}
    if all_fam and no_boundary_pref:
        verdict = "PHASE_GATED_RESIDUAL_CRITIC_PASS"
    elif not families_pass.get("contact_retention", True) or not families_pass.get("settling", True):
        verdict = "CRITIC_CONTACT_BLIND" if families_pass.get("transport") else "CRITIC_NO_USEFUL_LOCAL_RANKING"
    elif not no_boundary_pref:
        verdict = "CRITIC_BOUNDARY_PREFERENCE"
    else:
        verdict = "CRITIC_NO_USEFUL_LOCAL_RANKING"
    out["verdict"] = verdict
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nfamilies_pass {families_pass} | boundary_pref_rate {sweep['boundary_pref_rate']} "
          f"norm_corr {mean_norm_corr:.2f} | twin {twin_disagree}", flush=True)
    print(verdict, flush=True); print("CRITIC_AUTH_DONE", flush=True)


if __name__ == "__main__":
    main()
