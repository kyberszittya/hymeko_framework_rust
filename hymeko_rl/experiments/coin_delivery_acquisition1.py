"""COIN-DELIVERY-OVERNIGHT-2 PART II/III/IV — acquisition oracle campaign, component ablation, chained delivery, and the
PREREGISTERED multi-path gate ladder.

Evaluates the acquisition primitive (train.coin_delivery_acquisition) on the 19 acquisition-wall states and computes
EVERY preregistered gate signal, then applies the gate ladder EXACTLY as specified (no post-hoc weakening):
  STRICT (>=5/19 or >=25%) → Case A (PPO/TD3+BC/SAC 3 seeds)
  Alt1 class (CONTACT_LOSS >=3/4  OR  GEOMETRIC_HARD >=4/15, + correct>scrambled) → Case B
  Alt2 chain (>=3 new acq AND >=2 zone-entry, or >=1 center-reach, no easy loss) → Case B
  Alt3 demo (>=3 unique AND >=100 trajectories AND >=80% reproducibility) → Case B (TD3+BC distill)
  Alt4 budget-scaling (monotonic rise AND full/expanded >=3 unique AND shared structure) → Case B provisional
  Alt5 phase (>=30% abs improvement on a named transition, correct>scrambled, next stage not worse) → Case C
  Alt6 selector (per-state best >=5 while no single primitive >=5) → Case B selector
Structural requirements gate EVERY alt path: correct>scrambled where geometry is claimed; safety; easy-state
preservation; handoff = phase event; chained delivery reported separately; no horizon/count-only artifact.

NO env/reward/dynamics/CORE change. Handoff is a phase event. Reuses train.coin_delivery_acquisition +
coin_delivery1.p_grasp_carry (chained delivery).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from hymeko_rl.experiments.coin_delivery1 import p_grasp_carry
from hymeko_rl.train.coin_delivery_acquisition import (
    AcqParams,
    ApproachMode,
    AcquisitionPrimitive,
    Phase,
    eval_acquisition,
    roll_acquisition,
)

_OUT = Path("experiments/2026_07_20_coin_delivery_acquisition")
_RL2 = Path("experiments/2026_07_20_coin_delivery_rl2/manifests/coin_delivery_rl2.json")
_SCRAMBLE = np.array([-2, 3, -0, 1])                       # fixed geometry-scramble permutation (perm + sign flips)
_CENTER_TOL = 0.02
_ZONE_HALF = 0.04


def _log(m: str) -> None:
    print(m, flush=True)


def _states() -> dict:
    d = json.loads(_RL2.read_text())
    p = d["problems"]
    geo = [int(x["state_id"]) for x in p if x["failure_class"] == "GEOMETRIC_HARD"]
    cl = [int(x["state_id"]) for x in p if x["failure_class"] == "CONTACT_LOSS"]
    easy = [int(x["state_id"]) for x in p if x["failure_class"] == "SCRIPTED_CENTER_SUCCESS"]
    return {"geometric_hard": geo, "contact_loss": cl, "acquisition_wall": geo + cl, "easy": easy}


# ── PART IV — chained delivery (acquisition → grasp_carry → delivery semantics) ──────────────────────────────────────
def chained_delivery_roll(env, seed: int, params: AcqParams, *, h_acq: int = 90, h_deliver: int = 120,
                          scramble_perm=None) -> dict:
    """Acquire with the new primitive, then transport with grasp_carry under corrected delivery semantics (handoff
    non-terminal, terminate on center-reach). Reports the full funnel through to zone-entry / center-reach."""
    from hymeko_rl.train.coin_delivery_acquisition import scramble_geometry
    inner = env._env
    obs, _i = env.reset(seed=int(seed))
    env._horizon = h_acq + h_deliver + 8
    prim = AcquisitionPrimitive(params)
    prim.reset()
    stable = False
    for _ in range(h_acq):                                 # phase 1: acquisition
        o = obs if scramble_perm is None else scramble_geometry(obs, scramble_perm)
        obs, _r, _t, _tr, info = env.step(np.clip(prim.action(o), -1, 1).astype(np.float32))
        if prim.phase == Phase.DONE:
            stable = True
            break
        if info.get("safety_violation"):
            break
    if not stable:
        return {"seed": int(seed), "stable_acquisition": False, "zone_entry": False, "center_reach": False,
                "final_dtz": round(float(inner._planar_metrics.disk_to_zone), 4)}
    zone = center = False
    for _ in range(h_deliver):                             # phase 2: grasp_carry transport under delivery semantics
        obs, _r, _t, _tr, info = env.step(np.clip(p_grasp_carry(inner, 0), -1, 1).astype(np.float32))
        m = inner._planar_metrics
        dtz = float(m.disk_to_zone)
        zone = zone or (dtz <= _ZONE_HALF)
        center = center or (dtz <= _CENTER_TOL)
        if center or info.get("safety_violation"):
            break
    return {"seed": int(seed), "stable_acquisition": True, "zone_entry": zone, "center_reach": center,
            "final_dtz": round(float(inner._planar_metrics.disk_to_zone), 4)}


def chained_eval(seeds, params: AcqParams, *, env, scramble_perm=None) -> dict:
    rows = [chained_delivery_roll(env, s, params, scramble_perm=scramble_perm) for s in seeds]
    return {"n": len(rows), "rows": rows,
            "stable_acq": sum(r["stable_acquisition"] for r in rows),
            "zone_entry": sum(r["zone_entry"] for r in rows),
            "center_reach": sum(r["center_reach"] for r in rows)}


# ── DEMO reproducibility (Alt Gate 3) — perturbed replays of recovered states ────────────────────────────────────────
def demo_reproducibility(recovered_seeds, params: AcqParams, *, env, n_perturb: int = 12) -> dict:
    """For each recovered state, replay with controlled param perturbations (angle/closure/retry/pregrasp) + minor
    seed variation; count successful acquisition trajectories + reproducibility."""
    rng = np.random.default_rng(0)
    trajectories = 0
    successes = 0
    per_state = {}
    for sd in recovered_seeds:
        s_ok = 0
        for _ in range(n_perturb):
            pert = AcqParams(
                pregrasp_radius=float(np.clip(params.pregrasp_radius * (1 + rng.uniform(-0.15, 0.15)), 0.02, 0.1)),
                approach_gain=params.approach_gain, open_amount=params.open_amount, close_amount=params.close_amount,
                approach_angle=params.approach_angle + rng.uniform(-0.2, 0.2),
                asym_offset=params.asym_offset, stagger=params.stagger,
                align_steps=params.align_steps + int(rng.integers(-1, 2)),
                stabilize_dwell=params.stabilize_dwell, retry_count=params.retry_count,
                retry_angle_offset=params.retry_angle_offset + rng.uniform(-0.1, 0.1),
                approach_mode=params.approach_mode, staggered=params.staggered, retry=params.retry,
                regrasp=params.regrasp, geometry_conditioned=params.geometry_conditioned)
            r = roll_acquisition(env, sd, pert, horizon=120)
            trajectories += 1
            if r["stable_acquisition"]:
                successes += 1
                s_ok += 1
        per_state[int(sd)] = s_ok
    repro = round(successes / max(1, trajectories), 4)
    return {"n_unique": len(recovered_seeds), "n_trajectories": trajectories, "n_successful": successes,
            "reproducibility": repro, "per_state_successes": per_state,
            "distinct_success_states": sum(1 for v in per_state.values() if v > 0)}


# ── selector (Alt Gate 6) — per-state best mode ──────────────────────────────────────────────────────────────────────
_MODES = [("symmetric", dict(approach_mode=ApproachMode.SYMMETRIC)),
          ("asym_left", dict(approach_mode=ApproachMode.ASYM_LEFT, asym_offset=0.4)),
          ("asym_right", dict(approach_mode=ApproachMode.ASYM_RIGHT, asym_offset=0.4)),
          ("staggered", dict(staggered=True, stagger=0.4))]


def selector_oracle(seeds, base: AcqParams, *, env) -> dict:
    """Per-state best mode (fixed continuous params from ``base``). SELECTOR gate: per-state best >= 5 while no single
    mode reaches the strict gate."""
    from dataclasses import replace
    per_mode = {}
    for name, kw in _MODES:
        ev = eval_acquisition(replace(base, **kw), seeds, env=env)
        per_mode[name] = set(ev["recovered_seeds"])
    per_state_best = set().union(*per_mode.values()) if per_mode else set()
    return {"per_mode_n": {k: len(v) for k, v in per_mode.items()},
            "per_state_best_n": len(per_state_best), "per_state_best_seeds": sorted(per_state_best),
            "max_single_mode": max((len(v) for v in per_mode.values()), default=0)}


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser(description="COIN-DELIVERY acquisition oracle + preregistered gate ladder")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    from hymeko_rl.experiments.coin_delivery_acquisition1_run import run
    run(fast=a.fast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
