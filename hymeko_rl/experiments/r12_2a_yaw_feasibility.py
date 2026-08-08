"""R12.2-A feasibility probe (gate G_A) — does a certified straddle grasp PRESERVE a commanded object yaw?

Places the object (box) at a yaw grid via the non-invasive adapter, runs the frozen reach/capture, and measures per
placement-yaw: (i) certification rate, (ii) the post-grasp object yaw. If post-grasp yaw tracks the placement yaw
(slope ≈ 1, spread ≥ 15°), R12.2 is well-posed — orientation-varying handoffs are achievable, and A2/B can proceed.
If the grasp re-aligns (post-grasp yaw ~constant regardless of placement) or fails to certify off-axis, R12.2 needs a
grasp controller that HOLDS a commanded orientation — a separate escalation. This probe is the §11 production-scale
smoke before any multi-seed A2 run.

Run:  python -m hymeko_rl.experiments.r12_2a_yaw_feasibility [family] [n_seeds] [n_scen]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.coin_delivery.r12_orientation import object_yaw, reach_capture_at_yaw
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r12_hsikan1_dataset import _SCENARIOS

_OUT = Path("reports/2026-08-08-r12-2-orientation")
_DEFAULT_YAW_GRID_DEG = (0.0, 15.0, 30.0, 45.0)     # axis-aligned straddle default grid (A, pre-widening)
_GATE_SPREAD_DEG = 15.0                              # G_A: post-grasp yaw must span ≥ this across the grid


def _slope(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or np.std(x) < 1e-9:
        return float("nan")
    return float(np.polyfit(np.asarray(x), np.asarray(y), 1)[0])


def main() -> int:
    fam = sys.argv[1] if len(sys.argv) > 1 else "O4-S"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_scen = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    yaw_grid = tuple(float(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else _DEFAULT_YAW_GRID_DEG
    scenarios = _SCENARIOS[:n_scen]
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(fam).object_spec)
    t0 = time.perf_counter()
    print(f"G_A probe [{fam}]: yaw grid {yaw_grid}° × {len(scenarios)} scen × {n_seeds} seeds", flush=True)

    per_yaw: dict[float, list[float]] = {y: [] for y in yaw_grid}   # placement yaw → post-grasp yaws (deg)
    cert: dict[float, list[int]] = {y: [0, 0] for y in yaw_grid}    # placement yaw → [certified, attempts]
    rows = []
    for yaw_deg in yaw_grid:
        yaw = math.radians(yaw_deg)
        for sid in scenarios:
            scen = scenario_by_id(sid)
            for seed in range(n_seeds):
                h = reach_capture_at_yaw(rig, scen, seed, yaw, cfg, conf, obj)
                cert[yaw_deg][1] += 1
                ok = h.record is None
                cert[yaw_deg][0] += int(ok)
                post = math.degrees(object_yaw(h.snap)) if ok else float("nan")
                if ok:
                    per_yaw[yaw_deg].append(post)
                rows.append({"placement_yaw_deg": yaw_deg, "scenario": sid, "seed": seed,
                             "certified": ok, "post_grasp_yaw_deg": post,
                             "fail": None if ok else h.record.outcome_class})
                print(f"  [{yaw_deg:5.1f}° {sid:22s} s{seed}] cert={ok} post_yaw={post:7.2f}° "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    means = {y: (float(np.mean(v)) if v else float("nan")) for y, v in per_yaw.items()}
    valid = [(y, means[y]) for y in yaw_grid if not math.isnan(means[y])]
    spread = (max(m for _, m in valid) - min(m for _, m in valid)) if len(valid) > 1 else 0.0
    slope = _slope([y for y, _ in valid], [m for _, m in valid])
    gate_pass = spread >= _GATE_SPREAD_DEG and (math.isnan(slope) or slope > 0.5)

    print("\nplace_yaw°  cert   mean_post_yaw°", flush=True)
    for y in yaw_grid:
        c = cert[y]
        print(f"  {y:6.1f}    {c[0]}/{c[1]}   {means[y]:8.2f}", flush=True)
    verdict = "G_A PASS — grasp PRESERVES varied yaw ⇒ R12.2 well-posed, A2/B may proceed" if gate_pass else \
        "G_A FAIL — grasp does NOT preserve yaw (re-aligns / off-axis cert fails) ⇒ HALT: R12.2 needs an " \
        "orientation-holding grasp controller (separate escalation)"
    print(f"\npost-grasp yaw spread {spread:.2f}° (gate ≥{_GATE_SPREAD_DEG}°), slope {slope:.2f} → {verdict}", flush=True)

    _OUT.mkdir(parents=True, exist_ok=True)
    summary = {"family": fam, "yaw_grid_deg": list(yaw_grid), "n_seeds": n_seeds, "scenarios": list(scenarios),
               "per_yaw_mean_post_deg": means, "cert_rate": {y: cert[y][0] / max(1, cert[y][1]) for y in yaw_grid},
               "post_grasp_spread_deg": round(spread, 2), "slope": round(slope, 3) if not math.isnan(slope) else None,
               "gate_spread_deg": _GATE_SPREAD_DEG, "gate_pass": gate_pass, "wall_s": round(time.perf_counter() - t0, 1),
               "rows": rows}
    out_name = "ga_yaw_feasibility.json" if fam == "O4-S" else f"ga_yaw_feasibility_{fam}.json"
    (_OUT / out_name).write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out_name} ({summary['wall_s'] / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
