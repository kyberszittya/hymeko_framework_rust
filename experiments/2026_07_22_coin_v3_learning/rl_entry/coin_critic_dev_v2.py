"""CORRECTED V2 critic-development harness (§6-§13). Replaces the INVALIDATED first-pass ``coin_residual_critic_dev.py``
(kept as INVALIDATED_DIAGNOSTIC). Fixes: gated collection, causal-history critic state, kept ``truncated``, within-group
counterfactuals, full-remaining-horizon labels, frozen dev gate.

Pipeline:
  §6  disjoint seed banks (SHA-manifested) → rich gated transitions (STD_TRAIN)
  §7  grouped counterfactual full-horizon labels on STD_DEV panel (deterministic x2)
  §8  controlled ablation: INSTANTANEOUS (obs48+gate) vs CAUSAL_HISTORY (163) critic on IDENTICAL data
  §9  STANDARD_COMPOSITE_CRITIC_V2 (causal), scale-correct target smoothing, checkpoints {0,1k,3k,6k,10k,20k,40k}
  §10 actor-relevant dev metrics per physical family (centered dQ/dG, within-group ranking, ±gradQ1 empirical)
  §11 FROZEN development gate (two consecutive checkpoints) → PASS / FAILURE
  §12 advantage-critic fallback (within-group ranking) if standard fails
No actor update. No sealed panel opened here (that is a separate authorized step).

Usage: python coin_critic_dev_v2.py <pi0.pt> <out.json> [scale=full|smoke]
"""
import hashlib
import json
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, ".")
from hymeko_rl.coin_delivery.coin_counterfactual_labels import (  # noqa: E402
    GAMMA,
    capture_state_panel,
    collect_critic_transitions,
)
from hymeko_rl.coin_delivery.coin_critic_audit import FAMILIES, audit_family  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_controller import ZeroInitResidualActor  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_critic import CompositeTwinCritic, encode_controller_state  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_critic_causal import CausalCompositeTwinCritic, q1_grad_wrt_action  # noqa: E402
from hymeko_rl.coin_delivery.coin_residual_replay import residual_target_action  # noqa: E402
from hymeko_rl.coin_delivery.coin_rl_env import CoinRL4Dof  # noqa: E402
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor  # noqa: E402

TAU = 0.005
SAVE_AT = (0, 1000, 3000, 6000, 10000, 20000, 40000)
# §6 disjoint banks (by complete seed), disjoint from policy VALIDATION(7000-7029)/FINAL_TEST(8000-8049)/HEADLINE
BANKS = {"std_train": (6000, 6100), "std_dev": (6100, 6200), "std_sealed": (6200, 6250),
         "adv_train": (6300, 6400), "adv_dev": (6400, 6500), "adv_sealed": (6500, 6550)}
# §11 FROZEN development gate thresholds (fixed BEFORE any development result is observed)
GATE = {"transport_corr_Q1": 0.20, "transport_gap5": 0.60, "transport_gradwin": 0.55,
        "transport_harmful_rej": 0.60, "transport_boundary_pref_max": 0.50, "contact_gradwin": 0.55}


def _seeds(name):
    a, b = BANKS[name]; return list(range(a, b))


def _sha_seeds(seeds):
    return hashlib.sha256(json.dumps(list(seeds)).encode()).hexdigest()[:16]


# ── §9 standard twin critic (identical training for both ablation arms) ──
def train_std_critic(trs, pi0, residual0, steps, *, arm, seed=0, lr=1e-4, clip=1.0, log=print):
    torch.manual_seed(seed)
    Crit = CausalCompositeTwinCritic if arm == "causal" else CompositeTwinCritic
    crit, targ = Crit(), Crit(); targ.load_state_dict(crit.state_dict())
    opt = torch.optim.Adam(crit.parameters(), lr=lr)
    O = torch.tensor(np.stack([t["obs_t"] for t in trs])); O2 = torch.tensor(np.stack([t["obs_tp1"] for t in trs]))
    CS = torch.tensor(np.stack([t["cs_t"] for t in trs])); CS2 = torch.tensor(np.stack([t["cs_tp1"] for t in trs]))
    E = torch.tensor(np.stack([t["enc_t"] for t in trs])); E2 = torch.tensor(np.stack([t["enc_tp1"] for t in trs]))
    A = torch.tensor(np.stack([t["act"] for t in trs])); R = torch.tensor(np.array([t["reward"] for t in trs], np.float32))
    TERM = torch.tensor(np.array([float(t["terminated"]) for t in trs], np.float32))
    G2 = torch.tensor(np.array([t["gate_tp1"] for t in trs], np.float32))
    n = len(trs); rng = np.random.default_rng(seed); gen = torch.Generator().manual_seed(seed)
    ckpts, losses = {}, []
    t0 = time.time()
    for i in range(steps + 1):
        if i in SAVE_AT:
            c = Crit(); c.load_state_dict(crit.state_dict()); ckpts[i] = c
        if i == steps:
            break
        idx = rng.integers(0, n, 256)
        with torch.no_grad():
            ta = residual_target_action(pi0, residual0, O2[idx], G2[idx], generator=gen)   # scale-correct smoothing
            qn = targ.min_q(CS2[idx], ta) if arm == "causal" else targ.min_q(O2[idx], ta, E2[idx])
            y = R[idx] + GAMMA * (1.0 - TERM[idx]) * qn                                     # terminated-only mask
        q1, q2 = (crit(CS[idx], A[idx]) if arm == "causal" else crit(O[idx], A[idx], E[idx]))
        loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(crit.parameters(), clip); opt.step()
        with torch.no_grad():
            for p, pt in zip(crit.parameters(), targ.parameters()):
                pt.mul_(1 - TAU).add_(TAU * p)
        if i % 5000 == 0:
            sr = (i + 1) / max(time.time() - t0, 1e-6)
            log(f"    [{arm}] step {i}/{steps} loss {loss.item():.3f} {sr:.0f} it/s "
                f"ETA {(steps - i) / max(sr, 1):.0f}s")
            losses.append((i, float(loss.item())))
    return ckpts, losses


def _q_closures(crit, arm):
    def q_of(g, action, head):
        A = torch.tensor(np.asarray(action, np.float32)[None])
        with torch.no_grad():
            if arm == "causal":
                q1, q2 = crit(torch.tensor(g.causal_state[None]), A)
            else:
                q1, q2 = crit(torch.tensor(g.obs[None]),
                              A, torch.tensor(encode_controller_state(g.cstate)[None]))
        return {"Q1": float(q1), "Q2": float(q2), "min": float(min(q1.item(), q2.item()))}[head]

    def grad_of(g):
        if arm == "causal":
            return q1_grad_wrt_action(crit, g.causal_state, g.base, causal=True)
        return q1_grad_wrt_action(crit, g.obs, g.base, causal=False,
                                  enc_state=torch.tensor(encode_controller_state(g.cstate)[None]))
    return q_of, grad_of


def ck_pass(r):
    t, cr = r.get("transport", {}), r.get("contact_retention", {})
    if t.get("n", 0) == 0 or cr.get("n", 0) == 0 or t.get("gradQ1_wins") is None or cr.get("gradQ1_wins") is None:
        return False
    return (t["centered_corr_Q1_vs_dG"] > GATE["transport_corr_Q1"] and t["acc_gap5"] > GATE["transport_gap5"]
            and t["gradQ1_wins"] > GATE["transport_gradwin"] and cr["gradQ1_wins"] > GATE["contact_gradwin"]
            and t["harmful_rej"] > GATE["transport_harmful_rej"]
            and t["boundary_pref"] < GATE["transport_boundary_pref_max"])


def two_consecutive_pass(ck_results):
    ks = sorted(ck_results)
    return any(ck_pass(ck_results[ks[i]]) and ck_pass(ck_results[ks[i + 1]]) for i in range(len(ks) - 1))


# ── §12 twin advantage critic (within-group ranking; A_i(s,0)=0 by construction) ──
ADV_SAVE_AT = (0, 250, 500, 1000, 2000, 4000, 8000)


def _adv_state(g, arm):
    if arm == "causal":
        return g.causal_state
    return np.concatenate([g.obs, encode_controller_state(g.cstate)]).astype(np.float32)


def _adv_state_dim(arm):
    from hymeko_rl.coin_delivery.coin_residual_critic_state import RESIDUAL_CRITIC_STATE_DIM
    return RESIDUAL_CRITIC_STATE_DIM if arm == "causal" else (48 + 11)


class ResidualAdvantageTwinCritic(nn.Module):
    """Twin residual-advantage critics ``A_i(state, residual)`` with ``A_i(state, 0) = 0`` enforced by construction
    (the value at the zero residual is subtracted). ``A1`` is the actor-driving head; ``A2`` is the conservative twin."""

    def __init__(self, state_dim, act_dim=4, h=256):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(state_dim + act_dim, h), nn.ReLU(),
                                   nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.a1, self.a2 = mk(), mk()

    @staticmethod
    def _adv(net, s, delta):
        z = net(torch.cat([s, torch.zeros_like(delta)], -1)).squeeze(-1)
        return net(torch.cat([s, delta], -1)).squeeze(-1) - z

    def forward(self, s, delta):
        return self._adv(self.a1, s, delta), self._adv(self.a2, s, delta)


def train_adv_twin(groups, arm, *, steps=8000, seed=0, lr=1e-3, log=print):
    torch.manual_seed(seed)
    S, DE, DG, GID = [], [], [], []
    for g in groups:
        for d, gg in zip(g.cand_delta, np.array(g.G) - g.G0):
            S.append(_adv_state(g, arm)); DE.append(np.asarray(d, np.float32)); DG.append(float(gg)); GID.append(g.group_id)
    St = torch.tensor(np.stack(S)); DEt = torch.tensor(np.stack(DE)); DGt = torch.tensor(np.array(DG, np.float32)); GID = np.array(GID)
    adv = ResidualAdvantageTwinCritic(_adv_state_dim(arm)); opt = torch.optim.Adam(adv.parameters(), lr=lr)
    n = len(St); rng = np.random.default_rng(seed)
    by_group = {gid: np.where(GID == gid)[0] for gid in np.unique(GID)}
    ckpts = {}
    for i in range(steps + 1):
        if i in ADV_SAVE_AT:
            c = ResidualAdvantageTwinCritic(_adv_state_dim(arm)); c.load_state_dict(adv.state_dict()); ckpts[i] = c
        if i == steps:
            break
        idx = rng.integers(0, n, min(256, n))
        p1, p2 = adv(St[idx], DEt[idx]); mse = ((p1 - DGt[idx]) ** 2 + (p2 - DGt[idx]) ** 2).mean()
        gid = rng.choice(list(by_group)); mem = by_group[gid]          # WITHIN-GROUP ranking only (§12 hard rule)
        if len(mem) >= 2:
            ii = rng.choice(mem, 128); jj = rng.choice(mem, 128)
            q1i, _ = adv(St[ii], DEt[ii]); q1j, _ = adv(St[jj], DEt[jj])
            rank = torch.relu(-torch.sign(DGt[ii] - DGt[jj]) * (q1i - q1j)).mean()
        else:
            rank = torch.zeros(())
        opt.zero_grad(); (mse + 0.5 * rank).backward(); opt.step()
        if i % 2000 == 0:
            log(f"    [adv:{arm}] step {i}/{steps} mse {mse.item():.3f} rank {rank.item():.3f}")
    return ckpts


def adv_closures(adv, arm):
    def q_of(g, action, head):
        de = torch.tensor((np.asarray(action, np.float32) - g.base)[None])
        with torch.no_grad():
            a1, a2 = adv(torch.tensor(_adv_state(g, arm)[None]), de)
        return {"Q1": float(a1), "Q2": float(a2), "min": float(min(a1.item(), a2.item()))}[head]

    def grad_of(g):
        de = torch.zeros(1, 4, requires_grad=True)
        a1, _ = adv(torch.tensor(_adv_state(g, arm)[None]), de); a1.backward()
        return de.grad[0].numpy()
    return q_of, grad_of


def ablation_matched(inst_results, causal_results):
    """§8 — CONTROLLED comparison at MATCHED checkpoints (not each arm's own best): mean over TRAINED checkpoints
    (excl. @0) of ``causal - instant`` for the key metrics, in transport + contact_retention. Aliasing is confirmed
    only if the causal arm materially AND consistently beats the instantaneous arm — a single @0 blip is not a gain."""
    keys = ["centered_corr_Q1_vs_dG", "acc_gap5", "gradQ1_wins"]
    cks = [k for k in sorted(inst_results) if k != 0]
    delta, consistency = {}, {}
    for fam in ("transport", "contact_retention"):
        for key in keys:
            diffs = []
            for k in cks:
                iv, cv = inst_results[k].get(fam, {}).get(key), causal_results[k].get(fam, {}).get(key)
                if iv is not None and cv is not None:
                    diffs.append(cv - iv)
            if diffs:
                delta[f"{fam}.{key}"] = round(float(np.mean(diffs)), 3)
                consistency[f"{fam}.{key}"] = round(float(np.mean([d > 0 for d in diffs])), 2)
    gains = list(delta.values())
    material = (sum(1 for v in gains if v > 0.10) >= 2 and float(np.mean(gains)) > 0.05
               and float(np.mean([consistency[k] >= 0.6 for k in delta])) >= 0.5)
    return {"causal_minus_instant_mean_over_trained_ckpts": delta, "fraction_ckpts_causal_better": consistency,
            "verdict": "RESIDUAL_CRITIC_STATE_ALIASING_CONFIRMED" if material else "RESIDUAL_CRITIC_CAUSAL_STATE_NO_GAIN"}


def main():
    pi0_path = sys.argv[1]; out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/critic_dev_v2.json"
    scale = sys.argv[3] if len(sys.argv) > 3 else "full"
    per_family = 4 if scale == "smoke" else 10
    train_seeds = _seeds("std_train")[: (12 if scale == "smoke" else 100)]
    steps = 6000 if scale == "smoke" else 40000
    log = lambda *a: print(*a, flush=True)
    file_sha = hashlib.sha256(open(pi0_path, "rb").read()).hexdigest()
    log(f"[{time.strftime('%H:%M:%S')}] pi0 {file_sha[:8]} scale={scale} per_family={per_family} train_seeds={len(train_seeds)} steps={steps}")

    pi0 = load_frozen_clip_actor(pi0_path, freeze=True); residual0 = ZeroInitResidualActor()
    rl = CoinRL4Dof()
    out = {"pi0_sha": file_sha[:8], "scale": scale, "gate_thresholds": GATE,
           "banks": {k: {"range": v, "n": v[1] - v[0], "sha16": _sha_seeds(range(*v))} for k, v in BANKS.items()},
           "banks_disjoint": _banks_disjoint()}

    # §7 dev panel (shared by both ablation arms)
    log(f"[{time.strftime('%H:%M:%S')}] capturing STD_DEV counterfactual panel (per_family={per_family})...")
    t = time.time(); dev = capture_state_panel(pi0, _seeds("std_dev"), per_family=per_family)
    log(f"  dev panel {len(dev)} groups by_family={ {f: sum(g.family==f for g in dev) for f in FAMILIES} } ({time.time()-t:.0f}s)")

    # §6 transitions
    log(f"[{time.strftime('%H:%M:%S')}] collecting gated critic transitions ({len(train_seeds)} seeds)...")
    t = time.time(); trs = collect_critic_transitions(pi0, train_seeds, seed=0)
    log(f"  {len(trs)} transitions, gate-active {sum(x['gate_tp1']>0 for x in trs)}, "
        f"term {sum(x['terminated'] for x in trs)} trunc {sum(x['truncated'] for x in trs)} ({time.time()-t:.0f}s)")

    # §8/§9/§10 — train BOTH arms identically, audit each checkpoint
    arm_ck = {}
    for arm in ("instant", "causal"):
        log(f"[{time.strftime('%H:%M:%S')}] training {arm} standard twin critic...")
        ckpts, losses = train_std_critic(trs, pi0, residual0, steps, arm=arm, log=log)
        results = {}
        for k in sorted(ckpts):
            q_of, grad_of = _q_closures(ckpts[k], arm)
            results[k] = audit_family(dev, q_of, grad_of, rl, pi0)
            tr = results[k].get("transport", {}); cr = results[k].get("contact_retention", {})
            log(f"    [{arm}@{k}] transport corrQ1/dG {tr.get('centered_corr_Q1_vs_dG')} gap5 {tr.get('acc_gap5')} "
                f"+gradQ1 {tr.get('gradQ1_wins')} harmful_rej {tr.get('harmful_rej')} | contact +gradQ1 {cr.get('gradQ1_wins')}")
        arm_ck[arm] = {"results": results, "losses": losses}
    out["ablation_arms"] = {a: arm_ck[a]["results"] for a in arm_ck}
    out["losses"] = {a: arm_ck[a]["losses"] for a in arm_ck}

    # §8 ablation verdict — MATCHED checkpoints (controlled), not each arm's own best
    out["ablation"] = ablation_matched(arm_ck["instant"]["results"], arm_ck["causal"]["results"])
    log(f"\n[§8 ablation] {out['ablation']['verdict']}  "
        f"causal-instant(mean/trained)={out['ablation']['causal_minus_instant_mean_over_trained_ckpts']}")

    # §11 development gate — STANDARD critic is the CAUSAL arm (§9)
    std_pass = two_consecutive_pass(arm_ck["causal"]["results"])
    out["standard_development_pass"] = std_pass
    log(f"[§11] STANDARD (causal) development pass (2 consecutive ckpts): {std_pass}")

    if std_pass:
        out["critic_route"] = "standard"; out["verdict"] = "PHASE_GATED_RESIDUAL_CRITIC_DEVELOPMENT_PASS"
    else:
        log(f"[{time.strftime('%H:%M:%S')}] STANDARD failed dev → TWIN ADVANTAGE fallback (fresh disjoint panels)...")
        adv_steps = 8000 if scale != "smoke" else 2000
        adv_arm = "causal"                                             # richest state; §8 showed no aliasing either way
        adv_train = capture_state_panel(pi0, _seeds("adv_train"), per_family=per_family)
        adv_dev = capture_state_panel(pi0, _seeds("adv_dev"), per_family=per_family)
        adv_ckpts = train_adv_twin(adv_train, adv_arm, steps=adv_steps, log=log)
        adv_results = {}
        for k in sorted(adv_ckpts):
            aq, ag = adv_closures(adv_ckpts[k], adv_arm)
            adv_results[k] = audit_family(adv_dev, aq, ag, rl, pi0)
            tr = adv_results[k].get("transport", {}); cr = adv_results[k].get("contact_retention", {})
            log(f"    [adv@{k}] transport corrA1/dG {tr.get('centered_corr_Q1_vs_dG')} gap5 {tr.get('acc_gap5')} "
                f"+gradA1 {tr.get('gradQ1_wins')} harmful_rej {tr.get('harmful_rej')} | contact +gradA1 {cr.get('gradQ1_wins')}")
        out["advantage_arm"] = adv_arm
        out["advantage_dev"] = adv_results
        adv_pass = two_consecutive_pass(adv_results)
        out["advantage_development_pass"] = adv_pass
        log(f"[§12] TWIN ADVANTAGE development pass (2 consecutive ckpts): {adv_pass}")
        if adv_pass:
            out["critic_route"] = "advantage"; out["verdict"] = "RESIDUAL_ADVANTAGE_CRITIC_DEVELOPMENT_PASS"
        else:
            out["critic_route"] = "none"; out["verdict"] = "RESIDUAL_CRITIC_ROUTE_BLOCKED"

    json.dump(out, open(out_path, "w"), indent=1, default=float)
    log(f"\nroute={out.get('critic_route')} → {out['verdict']}\nwrote {out_path}\nCRITIC_DEV_V2_DONE")


def _banks_disjoint():
    seen = {}
    for k, (a, b) in BANKS.items():
        for s in range(a, b):
            if s in seen:
                return False
            seen[s] = k
    # also disjoint from policy banks
    policy = set(range(7000, 7030)) | set(range(8000, 8050)) | {1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568}
    return not (set(seen) & policy)


if __name__ == "__main__":
    main()
