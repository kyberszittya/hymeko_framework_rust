"""R2 — per-step state-dependent bounded-residual RL over the frozen clone: matched TD3-vs-SAC, gated.

R1 (constant residual) only added coast momentum. R2 learns a PER-STEP residual conditioned on the frozen clone's hidden state
(+ previous residual) to actively follow the coin in light contact toward the release corridor. Matched TD3 (main) vs SAC
(comparator): identical env / reward / snapshot / interaction-budget / net capacity / seed. Warm-started to the clone
(distil residual→0 over the clone trajectory), best checkpoint by the LEXICOGRAPHIC champion (safety ≻ K6 ≻ close+moving release
≻ contact-exit dtz ≻ landing ≻ smoothness), staged eval every 30 options to a 600 hard cap. Gates:

  R2-A `RESIDUAL_STATE_DEPENDENCE_PASS`      — the residual varies over the episode and tracks the contact state (not constant).
  R2-B `CONTACT_RETENTION_FRONTIER_ADVANCES` — contact-exit dtz below 55 mm, no clamp/stall.
  R3   `CLOSE_MOVING_RELEASE_MANIFOLD_REACHED` — min_dtz ≤ 30 mm, +v_par, light force, no stall/heavy clamp.
  R4   `FIRST_LEARNED_S1_K6_DELIVERY`        — strict K6, teacher-free.

Reports the learnability, the contact frontier, an event trace, and a reward decomposition; then stops.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor, KineticClone, KineticCloneController, NormStats
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
    AUG_DIM, KineticTemporalResidualController, TemporalResidualPolicy, deterministic_residual)
from hymeko_rl.coin_delivery.theta_option.residual_option_env import distill_zero_residual
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.option_rl.agents import make_actor

CLONE_CKPT = Path("reports/2026-07-28-coin-r9-k2-clone/clone_seed0.pt")
OUT = Path("reports/2026-07-28-coin-r9-r2-rl")
ALPHA = 0.15
FRONTIER_MM = 55.0                              # R2-B: contact-exit below this advances the frontier


def _load_clone() -> "tuple[KineticClone, NormStats]":
    ckpt = torch.load(CLONE_CKPT, weights_only=False)
    model = KineticClone(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    return model, NormStats(np.array(ckpt["norm"]["mean"]), np.array(ckpt["norm"]["std"]))


def _clone_augs(snap: Any, clone: CloneActor, bounds: ResidualBounds) -> np.ndarray:
    """The augmented states visited under the clone (zero residual) — the warm-start distil set."""
    ctrl = KineticTemporalResidualController(snap, clone, deterministic_residual(TemporalResidualPolicy(zero_init=True)), bounds)
    velocity_rollout(snap, ctrl, kc.DELIVERY_CFG)
    return np.array([s for s, _r in ctrl.aug_trace], np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.corrcoef(a, b)[0, 1])) if len(a) > 2 and a.std() > 1e-9 and b.std() > 1e-9 else 0.0


def _state_dependence(snap: Any, clone: CloneActor, actor: Any, bounds: ResidualBounds) -> dict:
    """R2-A: deploy the actor and measure whether the residual VARIES across the episode and TRACKS the contact / slip /
    geometry (per the R2-A spec — not the force alone). Reports the max correlation of any residual dimension (and its norm)
    with contact force, coin geometry (dtz), and forward velocity."""
    ctrl = KineticTemporalResidualController(snap, clone, deterministic_residual(actor), bounds)
    velocity_rollout(snap, ctrl, kc.DELIVERY_CFG)
    kin = [r for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE"]
    res = np.array([r for _s, r in ctrl.aug_trace], np.float64)
    feats = {"fn": np.array([min(r["fn_l"], r["fn_r"]) for r in kin], np.float64),
             "dtz": np.array([r["dtz_mm"] for r in kin], np.float64),
             "v_par": np.array([r["v_par"] for r in kin], np.float64)}
    std = float(res.std())
    rnorm = np.linalg.norm(res, axis=1) if len(res) else np.zeros(0)
    per_feat = {k: round(max(_corr(res[:, j], v) for j in range(res.shape[1])) if len(res) else 0.0, 3)
                for k, v in feats.items()}
    per_feat = {**per_feat, **{f"norm_{k}": round(_corr(rnorm, v), 3) for k, v in feats.items()}}
    max_corr = max(per_feat.values()) if per_feat else 0.0
    return {"residual_std": round(std, 4), "corr": per_feat, "max_corr": round(max_corr, 3), "n_steps": len(res),
            "varies": bool(std > 0.02), "tracks_state": bool(max_corr > 0.3),
            "residuals": [[round(float(x), 4) for x in r] for r in res]}


def _run_algo(algo: str, snap: Any, model: KineticClone, norm: NormStats, bounds: ResidualBounds,
              cfg: krl2.PerStepConfig, seed: int) -> dict:
    actor = make_actor(algo, AUG_DIM, krl2.ACT_DIM)
    distill_loss = distill_zero_residual(actor, _clone_augs(snap, CloneActor(model, norm), bounds), seed=seed)
    best_actor, history = krl2.train_perstep(algo, snap, CloneActor(model, norm), bounds, krl2.Reward2Weights(), cfg, seed=seed)
    ep = krl2.collect_episode(snap, CloneActor(model, norm), deterministic_residual(best_actor), bounds, krl2.Reward2Weights())
    sd = _state_dependence(snap, CloneActor(model, norm), best_actor, bounds)
    ctrl = KineticTemporalResidualController(snap, CloneActor(model, norm), deterministic_residual(best_actor), bounds)
    velocity_rollout(snap, ctrl, kc.DELIVERY_CFG)
    kin = [{"dtz_mm": r["dtz_mm"], "v_par": r["v_par"], "fn": round(min(r["fn_l"], r["fn_r"]), 3)}
           for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE"]
    milestone = next((h for h in history if h["it"] == 240), history[-1] if history else None)
    return {"algo": algo, "distill_loss": round(distill_loss, 6), "min_dtz_mm": round(ep.min_dtz, 2),
            "exit_dtz_mm": round(ep.exit_dtz, 2), "k6": bool(ep.k6), "safe": bool(ep.safe),
            "released": bool(ep.decomp["released"]), "reward_decomp": ep.decomp, "state_dependence": sd,
            "milestone_240": milestone, "final_champion": history[-1] if history else None, "event_trace": kin}


def run(total_options: int = 600, seed: int = 0) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    model, norm = _load_clone()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    bounds = ResidualBounds(alpha=ALPHA)

    # R2 update-zero identity (temporal wrapper)
    m_clone = velocity_rollout(snap, KineticCloneController(snap, CloneActor(model, norm)), kc.DELIVERY_CFG)
    baseline = _min_dtz_mm(snap, m_clone)
    ctrl0 = KineticTemporalResidualController(snap, CloneActor(model, norm),
                                              deterministic_residual(TemporalResidualPolicy(zero_init=True)), bounds)
    m0 = velocity_rollout(snap, ctrl0, kc.DELIVERY_CFG)
    r2_identity = bool(np.array_equal(np.asarray(m_clone["coin_trace"]), np.asarray(m0["coin_trace"])))

    cfg = krl2.PerStepConfig(total_options=total_options)
    td3 = _run_algo("td3", snap, model, norm, bounds, cfg, seed)
    sac = _run_algo("sac", snap, model, norm, bounds, cfg, seed)

    best = min((td3, sac), key=lambda r: (not r["safe"], not r["k6"], not r["released"], r["exit_dtz_mm"], r["min_dtz_mm"]))
    r2a = bool(best["state_dependence"]["varies"] and best["state_dependence"]["tracks_state"])
    r2b = bool(best["exit_dtz_mm"] < FRONTIER_MM and best["reward_decomp"]["clamp"] == 0.0)
    r3 = bool(best["min_dtz_mm"] <= 30.0 and best["released"] and best["safe"])
    r4 = bool(best["k6"] and best["safe"])
    gates = {"R2A_state_dependence": r2a, "R2B_contact_frontier": r2b, "R3_close_moving_manifold": r3, "R4_k6_delivery": r4}
    verdict = ("FIRST_LEARNED_S1_K6_DELIVERY" if r4 else "CLOSE_MOVING_RELEASE_MANIFOLD_REACHED" if r3
               else "CONTACT_RETENTION_FRONTIER_ADVANCES" if r2b else "RESIDUAL_STATE_DEPENDENCE_PASS" if r2a
               else "R2_NO_STATE_DEPENDENT_ADVANCE")
    out = {"contract": "COIN_KINETIC_R2_RL_V1", "seed": seed, "alpha": ALPHA, "total_options": total_options,
           "r2_update_zero_identity": r2_identity, "clone_baseline_mm": round(baseline, 2),
           "td3": td3, "sac": sac, "best_algo": best["algo"], "gates": gates, "verdict": verdict,
           "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r2_rl.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = run()
    print(f"\nR2 update-zero identity: {r['r2_update_zero_identity']} | clone baseline {r['clone_baseline_mm']}mm")
    print(f"R2 VERDICT: {r['verdict']}  gates {r['gates']}  (best {r['best_algo']}; wall {r['wall_s']}s)\n")
    for algo in ("td3", "sac"):
        a = r[algo]
        sd = a["state_dependence"]
        print(f"  {algo.upper():4s}: min_dtz {a['min_dtz_mm']}mm  exit_dtz {a['exit_dtz_mm']}mm  released={a['released']}  K6={a['k6']}  safe={a['safe']}")
        print(f"        state-dep: std {sd['residual_std']} max-corr {sd['max_corr']} {sd['corr']} varies={sd['varies']} tracks={sd['tracks_state']}")
        print(f"        reward decomp: {a['reward_decomp']}")
        print(f"        milestone@240: {a['milestone_240']['aux'] if a['milestone_240'] else None}")
