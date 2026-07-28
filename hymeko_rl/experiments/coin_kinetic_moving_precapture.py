"""R10 Stage 1B — phase-shaped moving-precursor reachability gate: HOME_V1 -> ... -> strict K6, >= 3 planner seeds.

Runs the full deterministic chain (frozen analytic transit -> phase-shaped moving precursor -> frozen downstream) for
several independent planner seeds, checks the reachability gate ladder + the READY->CAPTURE boundary, and saves the
winning parameters + causal traces for reproducible replay. Downstream frozen; no RL; no state edit; no tag moved.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_moving_precapture``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.home_states import HOME_STATE_V1_GENERIC, build_home_snapshot
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual
from hymeko_rl.experiments.coin_kinetic_ablation import _rebuild

OUT = Path("reports/2026-07-28-moving-precapture-dynamic-handoff")
CKPT = Path("reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json")
SEEDS = (11, 23, 42)


def _build() -> "tuple[Any, Any, mp.HandoffReference, mp.FrozenDownstream, Any]":
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone
    model, norm = _load_clone()
    cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    coin = _coin_xy(cradle.branch())
    home = build_home_snapshot(cradle, HOME_STATE_V1_GENERIC)
    ready = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, pga.CoinStraddleTargets(coin=coin),
                                pga.TransitConfig()).ready_snapshot
    ref = mp.HandoffReference.from_cradle(cradle)
    r2_fn = deterministic_residual(_rebuild(json.load(open(CKPT))["r2_actor_state"]))
    down = mp.FrozenDownstream(model, norm, r2_fn, cradle.stack)
    return cradle, ready, ref, down, cradle.stack


def _seed_record(res: mp.CaptureResult, kinds: list) -> dict:
    o, p = res.outcome, res.params
    return {"seed": res.seed, "k6": o.k6, "min_dtz_mm": o.min_dtz_mm, "safe": o.safe,
            "cos_dir": o.cos_dir, "vel_scale": o.vel_scale, "dtau": o.dtau, "contacts": o.contacts,
            "handoff_resets": int(kinds.count("HANDOFF_RESET")),
            "params": {"n": p.n, "s": round(p.s, 4), "preload_start": round(p.preload_start, 4),
                       "bmax": round(p.bmax, 4), "residual": np.round(p.residual, 4).tolist()}}


def _boundary(ready: Any, ref: mp.HandoffReference, stack: Any, best: mp.CaptureResult,
              down: mp.FrozenDownstream) -> dict:
    """READY->CAPTURE boundary: the capture continues from READY's exact state (no re-init/edit); a degenerate
    zero-shaping capture does NOT deliver (proves the phase-shaping is load-bearing, not a trivially-met boundary)."""
    cap = mp.PhaseShapeCapture(ready, ref, stack)
    rd = ready.branch().inner.data
    one_step_identity = bool(np.allclose(cap.q0, rd.qpos[:4]) and np.allclose(cap.v0, rd.qvel[:4])
                             and np.allclose(cap.prev0, np.asarray(ready.prev_tau)))
    replay = cap.roll(best.params)
    replay_det = bool(down.deliver(replay.snapshot)[1] == best.outcome.min_dtz_mm)
    degenerate = cap.roll(mp.CaptureParams(n=0.0, s=0.0, preload_start=1.0, bmax=0.0,
                                           residual=np.zeros((2, 4)), steps=best.params.steps))
    off_by_one_rejected = not down.deliver(degenerate.snapshot)[0]
    return {"READY_TO_CAPTURE_ONE_STEP_IDENTITY_PASS": one_step_identity,
            "TRANSIT_TO_CAPTURE_CONTINUATION_PASS": replay_det,
            "BOUNDARY_OFF_BY_ONE_REJECTED_PASS": off_by_one_rejected}


def _solve_seeds(seeds: "tuple[int, ...]", ready: Any, ref: mp.HandoffReference, down: mp.FrozenDownstream,
                 stack: Any) -> "tuple[list, mp.CaptureResult]":
    records, best = [], None
    for seed in seeds:
        res = mp.plan_capture(ready, ref, stack, down, seed=seed)
        _, _, _, kinds = down.deliver_with_trace(res.outcome.snapshot)
        records.append(_seed_record(res, kinds))
        if best is None or (res.outcome.k6 and res.outcome.min_dtz_mm < best.outcome.min_dtz_mm):
            best = res
    return records, best


def _summarize(records: list, boundary: dict, ref: mp.HandoffReference) -> dict:
    k6 = [r for r in records if r["k6"] and r["safe"]]
    reachability = bool(len(k6) >= 3 and all(r["handoff_resets"] == 1 for r in k6) and all(boundary.values()))
    gates = {"MOVING_PRECAPTURE_TO_HANDOFF_PASS": bool(len(k6) >= 1),
             "HOME_V1_TO_DYNAMIC_HANDOFF_REACHABILITY_PASS": reachability}
    verdict = "HOME_V1_TO_DYNAMIC_HANDOFF_REACHABILITY_PASS" if reachability else "MOVING_PRECAPTURE_PARTIAL"
    return {"contract": "MOVING_PRECAPTURE_V1", "immutable_transit": "eb09a16a",
            "handoff_ref": {"q_star": np.round(ref.q_star, 4).tolist(),
                            "qvel_star": np.round(ref.qvel_star, 4).tolist(), "speed": round(ref.speed, 4)},
            "seeds": records, "distinct_params": len({(r["params"]["s"], r["params"]["preload_start"]) for r in k6}),
            "boundary": boundary, "gates": gates, "verdict": verdict}


def run(out: Path = OUT, seeds: "tuple[int, ...]" = SEEDS) -> dict:
    _, ready, ref, down, stack = _build()
    records, best = _solve_seeds(seeds, ready, ref, down, stack)
    summary = _summarize(records, _boundary(ready, ref, stack, best, down), ref)
    out.mkdir(parents=True, exist_ok=True)
    (out / "moving_precapture.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    r = run()
    print(f"\nMOVING PRECAPTURE: {r['verdict']}")
    for rec in r["seeds"]:
        print(f"  seed {rec['seed']}: K6 {rec['k6']} min_dtz {rec['min_dtz_mm']}mm safe {rec['safe']} "
              f"resets {rec['handoff_resets']} | s={rec['params']['s']} ps={rec['params']['preload_start']} "
              f"bmax={rec['params']['bmax']}")
    print(f"  distinct (s,preload_start) among K6 seeds: {r['distinct_params']}")
    for k, v in {**r["boundary"], **r["gates"]}.items():
        print(f"  {'PASS' if v else 'FAIL'} {k}")
