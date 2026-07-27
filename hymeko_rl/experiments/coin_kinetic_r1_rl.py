"""R0/R1/R2 — bounded-residual update-zero identity + matched TD3-vs-SAC learnability smoke over the frozen KINETIC clone.

R0: verify the residual wrapper reproduces the K2 clone BIT-FOR-BIT at zero residual (else no training). R1/R2: a small,
pre-registered matched TD3-vs-SAC smoke (identical env / reward / snapshot / interaction-budget / net capacity / seed — only
the algo differs) checking for a reward-driven improvement over the clone's 46.2 mm, teacher-free, with a full-frozen-chain
evaluation and a reward decomposition. TD3 is the main branch (better in the prior bounded-residual coin arc); SAC is the
mandatory matched comparator, re-measured on this new KINETIC sub-task. This does NOT chase K6 — it reports the learnability
signal and stops.

Gates: R0 `RESIDUAL_UPDATE_ZERO_IDENTITY_PASS`; R1 `RESIDUAL_LEARNABILITY_SIGNAL_PASS` (clear improvement over 46.2 mm, no
safety degradation). Run: ``python -m hymeko_rl.experiments.coin_kinetic_r1_rl``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_rl as krl
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor, KineticClone, KineticCloneController, NormStats
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.residual_option_env import distill_zero_residual
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.option_rl.agents import SemiMDPConfig, make_actor, train_semi_mdp

CLONE_CKPT = Path("reports/2026-07-28-coin-r9-k2-clone/clone_seed0.pt")
OUT = Path("reports/2026-07-28-coin-r9-r1-rl")
ALPHA = 0.15                                    # residual bound (small — refines a working clone)
IMPROVE_MM = 1.5                                # "clear improvement" margin over the clone baseline for the R1 gate
ACT_DIM = 4                                     # the residual action basis (per-joint slew-normalised Δτ)


def _smoke_cfg(total_options: int) -> SemiMDPConfig:
    return SemiMDPConfig(gamma=0.99, tau_polyak=0.01, lr=3e-4, batch=64, warmup_options=max(20, total_options // 4),
                         total_options=total_options, updates_per_option=2, eval_every=max(20, total_options // 4),
                         reward_scale=krl.REWARD_SCALE, policy_delay=2, target_noise=0.15, noise_clip=0.3,
                         expl_noise=0.2, alpha=0.1)


def _load_clone() -> "tuple[KineticClone, NormStats]":
    ckpt = torch.load(CLONE_CKPT, weights_only=False)
    model = KineticClone(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    return model, NormStats(np.array(ckpt["norm"]["mean"]), np.array(ckpt["norm"]["std"]))


def _run_algo(algo: str, snap: Any, model: KineticClone, norm: NormStats, init_obs: np.ndarray, baseline: float,
              bounds: ResidualBounds, cfg: SemiMDPConfig, seed: int) -> dict:
    """One matched RL branch: distil residual→0 (update-0 = clone), train, then eval the best_val actor on the full chain."""
    actor = make_actor(algo, krl.OBS_DIM, ACT_DIM)
    distill_loss = distill_zero_residual(actor, init_obs, seed=seed)
    env = krl.KineticResidualOptionEnv(snap, CloneActor(model, norm),
                                       lambda m, d, k: krl.kinetic_reward(m, d, k, baseline), init_obs, bounds)
    dev_eval = krl.make_dev_eval(snap, CloneActor(model, norm), init_obs, bounds, baseline)
    ckpts, history = train_semi_mdp(algo, env, actor, dev_eval, cfg, obs_dim=krl.OBS_DIM, act_dim=ACT_DIM, seed=seed)
    actor.load_state_dict(ckpts["best_val"])
    with torch.no_grad():
        a = actor.mean_action(torch.as_tensor(np.asarray(init_obs, np.float32)[None]))[0].numpy()
    m, min_dtz, kin = krl.deploy_residual(snap, CloneActor(model, norm), a, bounds)
    _reward, decomp = krl.kinetic_reward(m, min_dtz, kin, baseline)
    return {"algo": algo, "distill_loss": round(distill_loss, 6), "best_residual": [round(float(x), 4) for x in a],
            "min_dtz_mm": round(min_dtz, 2), "improved_over_clone": bool(min_dtz < baseline - 1e-6),
            "k6": bool(m["k6_delivered"]), "peak_qdot": round(m["peak_qdot"], 3),
            "peak_coin_speed": round(m["peak_coin_speed"], 3), "reward_decomp": decomp,
            "history_tail": history[-3:]}


def run(total_options: int = 100, seed: int = 0) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    model, norm = _load_clone()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    bounds = ResidualBounds(alpha=ALPHA)
    init_obs = krl.kinetic_init_obs(snap, CloneActor(model, norm))

    # R0 — update-zero identity: zero residual == the K2 clone, bit-for-bit
    m_clone = velocity_rollout(snap, KineticCloneController(snap, CloneActor(model, norm)), kc.DELIVERY_CFG)
    baseline = _min_dtz_mm(snap, m_clone)
    m_zero, dtz_zero, _k = krl.deploy_residual(snap, CloneActor(model, norm), np.zeros(4), bounds)
    r0_identical = bool(np.array_equal(np.asarray(m_clone["coin_trace"]), np.asarray(m_zero["coin_trace"]))
                        and abs(baseline - dtz_zero) < 1e-9)
    r0 = {"gate": "RESIDUAL_UPDATE_ZERO_IDENTITY_PASS" if r0_identical else "RESIDUAL_UPDATE_ZERO_IDENTITY_FAIL",
          "clone_min_dtz_mm": round(baseline, 3), "zero_residual_min_dtz_mm": round(dtz_zero, 3),
          "coin_trace_bit_identical": r0_identical}
    if not r0_identical:
        raise SystemExit(f"R0 FAILED — zero residual is not the clone: {r0}")

    cfg = _smoke_cfg(total_options)
    td3 = _run_algo("td3", snap, model, norm, init_obs, baseline, bounds, cfg, seed)
    sac = _run_algo("sac", snap, model, norm, init_obs, baseline, bounds, cfg, seed)

    best = min((td3, sac), key=lambda r: r["min_dtz_mm"])
    safe = td3["reward_decomp"]["safe"] and sac["reward_decomp"]["safe"]
    signal = bool(best["min_dtz_mm"] < baseline - IMPROVE_MM and safe)
    verdict = "RESIDUAL_LEARNABILITY_SIGNAL_PASS" if signal else "RESIDUAL_LEARNABILITY_SIGNAL_ABSENT"
    out = {"contract": "COIN_KINETIC_R1_RL_V1", "seed": seed, "alpha": ALPHA, "total_options": total_options,
           "r0": r0, "clone_baseline_mm": round(baseline, 2), "td3": td3, "sac": sac,
           "best_algo": best["algo"], "best_min_dtz_mm": best["min_dtz_mm"], "verdict": verdict,
           "matched": {"same_env": True, "same_cfg": True, "same_seed": seed, "same_init_obs": True,
                       "same_bounds_alpha": ALPHA, "obs_dim": krl.OBS_DIM, "act_dim": 4}, "wall_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "r1_rl.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = run()
    print(f"\nR0: {r['r0']['gate']}  (clone {r['r0']['clone_min_dtz_mm']}mm == zero-residual {r['r0']['zero_residual_min_dtz_mm']}mm, "
          f"bit-identical {r['r0']['coin_trace_bit_identical']})")
    print(f"R1 VERDICT: {r['verdict']}  (clone baseline {r['clone_baseline_mm']}mm; best {r['best_algo']} {r['best_min_dtz_mm']}mm; wall {r['wall_s']}s)\n")
    for algo in ("td3", "sac"):
        a = r[algo]
        print(f"  {algo.upper():4s}: min_dtz {a['min_dtz_mm']}mm  improved={a['improved_over_clone']}  K6={a['k6']}  "
              f"residual {a['best_residual']}  peak_qdot {a['peak_qdot']}")
        print(f"        reward decomp: {a['reward_decomp']}")
