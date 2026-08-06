"""R11.7A U6B — cost probe (§11 wall-time reconciliation before the bounded retrieval pilot).

Measures the real per-(scenario,seed) wall of the teacher-theta bank step on a VARIANT rig: reach → capture →
teacher delivery search (R restarts). One object, one train scenario, a few seeds. The measurement resolves the
"N=20 capture population" cost so the full U6B estimate can be reconciled before launching hours of compute.

Run:  python -m hymeko_rl.experiments.r11_7a_u6b_cost_probe [variant_id] [scenario_id] [n_seeds] [R]
"""
from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import (
    bc_context, best_theta_full, descriptor, full_transport_spec, reconstruct_capture, scenario_by_id)
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig


def _sample_on_rig(rig: dict, cfg: Any, conf: Any, obj: Any, sid: str, seed: int,
                   restarts: int) -> "tuple[str, float, np.ndarray | None]":
    """One bank step on a pre-built (reusable) rig: reach+capture → teacher theta. Returns
    (outcome, dtz_mm, theta) — outcome in {NO_CERTIFIED_CAPTURE, TEACHER_NO_K6, K6}."""
    scen = scenario_by_id(sid)
    rc = reconstruct_capture(rig, cfg, conf, obj, scen, seed)
    if rc is None:
        return "NO_CERTIFIED_CAPTURE", float("nan"), None
    snap = rc.result.outcome.snapshot
    k6, dtz, theta = best_theta_full(snap, full_transport_spec(), restarts)
    if not k6:
        return "TEACHER_NO_K6", float(dtz), None
    _x = descriptor(scen, rc, snap)                          # exercise the descriptor path too
    return "K6", float(dtz), np.asarray(theta, np.float64)


def main() -> int:
    variant_id = sys.argv[1] if len(sys.argv) > 1 else "O1-L"
    sid = sys.argv[2] if len(sys.argv) > 2 else "bank_c1_-0.03_+0.02"
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    print(f"probe: variant={variant_id} scenario={sid} n_seeds={n_seeds} R={restarts}", flush=True)
    cfg, conf, obj = bc_context()
    t_rig = time.perf_counter()
    rig = _rig(object_spec=variant(variant_id).object_spec)   # built ONCE, reused across seeds
    rig_wall = time.perf_counter() - t_rig
    print(f"rig build (acquire cradle): {rig_wall:.1f}s", flush=True)

    per_seed = []
    for seed in range(n_seeds):
        t0 = time.perf_counter()
        outcome, dtz, theta = _sample_on_rig(rig, cfg, conf, obj, sid, seed, restarts)
        dt = time.perf_counter() - t0
        per_seed.append(dt)
        print(f"  seed {seed}: {outcome:22s} dtz={dtz:.2f}mm  ({dt:.1f}s)", flush=True)

    med = float(np.median(per_seed))
    # Full U6B estimate: 3 variants x 8 train scenarios x N=20 seeds x median-per-seed, + rig builds.
    full_h = (3 * 8 * 20 * med + 3 * rig_wall) / 3600.0
    print(f"\nmedian per-seed: {med:.1f}s | rig build: {rig_wall:.1f}s")
    print(f"FULL U6B estimate (3 var x 8 train x N=20 x {restarts}R): ~{full_h:.1f} h "
          f"(bank-gen only; excludes dev/test eval)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
