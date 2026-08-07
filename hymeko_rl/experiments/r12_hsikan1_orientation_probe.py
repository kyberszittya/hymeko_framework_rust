"""R12 / HSiKAN-1 closure-check T2 — PRE-FLIGHT: does object yaw vary across handoffs, and is it aliased?

Before building the orientation×architecture interaction test we must confirm its premise: (a) the coin/box yaw actually
VARIES across the handoff panel (else the interaction is vacuous), and (b) it is not already recoverable from the 30-D
descriptor (else there is nothing to add). Reads the object geom's WORLD rotation matrix (joint-agnostic — avoids the
planar-3DOF vs 6DOF-freejoint ambiguity) from each re-acquired handoff snapshot.

Run:  python -m hymeko_rl.experiments.r12_hsikan1_orientation_probe [n_seeds]
"""
from __future__ import annotations

import math
import sys

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.exact_zero_composition import reach_capture_descriptor
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r12_hsikan1_dataset import _FAMILIES, _SCENARIOS


def _object_yaw(snap) -> float:
    """World-frame yaw (rad) of the object geom, from its rotation matrix — independent of the joint parameterization."""
    inner = snap._rl.inner
    m, d = inner.model, inner.data
    names = [m.geom(g).name or "" for g in range(m.ngeom)]
    gid = next((g for g, nm in enumerate(names) if "disk" in nm or "coin" in nm or "box" in nm), None)
    if gid is None:
        return float("nan")
    r = np.asarray(d.geom_xmat[gid], np.float64).reshape(3, 3)      # world rotation of the object geom
    return math.atan2(r[1, 0], r[0, 0])                            # yaw about world z


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cfg, conf, obj = bc_context()
    print(f"orientation probe: {len(_FAMILIES)} families × {len(_SCENARIOS)} scen × {n_seeds} seeds", flush=True)
    per_family: dict[str, list[float]] = {f: [] for f in _FAMILIES}
    rows: list[tuple] = []
    for fam in _FAMILIES:
        rig = _rig(object_spec=variant(fam).object_spec)
        for sid in _SCENARIOS:
            for seed in range(n_seeds):
                h = reach_capture_descriptor(rig, scenario_by_id(sid), seed, cfg, conf, obj)
                if h.record is not None:
                    continue
                yaw = _object_yaw(h.snap)
                per_family[fam].append(yaw)
                # x[8:10] = coin xy — does the descriptor carry any yaw proxy? (it should not)
                cx, cy = float(h.x[8]), float(h.x[9])
                rows.append((fam, sid, seed, yaw, cx, cy))
    print("\nfam    n   yaw_deg[min,max]   spread_deg   note", flush=True)
    for fam, ys in per_family.items():
        if not ys:
            print(f"  {fam:5s}  0   (no valid handoffs)", flush=True)
            continue
        deg = np.degrees(ys)
        spread = float(deg.max() - deg.min())
        note = "VARIES" if spread > 1.0 else "~constant (yaw irrelevant / symmetric)"
        print(f"  {fam:5s}  {len(ys):2d}  [{deg.min():+7.2f},{deg.max():+7.2f}]  {spread:8.2f}   {note}", flush=True)
    # is yaw linearly recoverable from descriptor coin-xy? (crude aliasing check)
    ys = np.array([r[3] for r in rows])
    cx = np.array([r[4] for r in rows])
    cy = np.array([r[5] for r in rows])
    if len(ys) > 3 and np.std(ys) > 1e-6:
        A = np.c_[cx, cy, np.ones_like(cx)]
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        resid = ys - A @ coef
        r2 = 1.0 - float(np.var(resid) / np.var(ys))
        print(f"\nyaw ~ linear(coin_xy):  R² = {r2:.3f}  "
              f"({'ALIASED (recoverable — no new info)' if r2 > 0.9 else 'NOT recoverable from coin-xy ⇒ genuine new info'})",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
