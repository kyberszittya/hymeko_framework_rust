"""R12.2-A2 — orientation-varying transportability dataset (the supervision R12.2-B learns from).

Like the R12.1 dataset (`r12_hsikan1_dataset.py`) but the handoffs VARY object yaw: for the elongated O5-R rectangle,
re-acquire handoffs on a yaw × scenario × seed grid via `reach_capture_at_yaw` (the co-rotating straddle), record the
post-grasp yaw per handoff, and apply the pooled-θ bank to each → (K6, dtz, safe). Every row carries `yaw_deg`
(commanded) and `post_grasp_yaw_deg` (measured) so R12.2-B can add orientation `(R,ω)` to the descriptor and test the
Δ_HSiKAN − Δ_MLP interaction. Only certified handoffs (a stable straddle grasp) contribute rows; the certified yield
per yaw is logged (no silent caps).

Run:  python -m hymeko_rl.experiments.r12_2a_orientation_dataset [family] [n_seeds] [n_scen] [yaws] [n_theta]

⚠️ FINDING (2026-08-10, first bounded run, O5-R): the POOLED θ bank is axis-aligned-tuned, so it delivers the rotated
rectangle ONLY near yaw≈0 — 17 positives all from yaw 0/early-30, ZERO at yaw 30/60/90 (confirmed 3× at handoffs
5/10/15). With yaw-0 θ, K6 collapses to a trivial "is-yaw-near-0" indicator ⇒ a DEGENERATE, confounded basis for the
R12.2-B interaction test. FIX: this generator needs ORIENTATION-AWARE θ — a per-yaw delivering-θ search (teacher/CEM
at each orientation) so that "which θ delivers" depends on orientation (the orientation×θ structure R12.2-B tests).
Do NOT run this at scale with the pooled bank; wire in a per-yaw θ source first.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import clip_theta
from hymeko_rl.coin_delivery.exact_zero_composition import _delivery_signals
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.coin_delivery.r12_orientation import object_yaw, reach_capture_at_yaw
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r12_hsikan1_dataset import _SCENARIOS, _pooled_thetas

_OUT = Path("reports/2026-08-08-r12-2-orientation")
_DEFAULT_YAWS = (0.0, 30.0, 45.0, 60.0, 90.0)     # commanded placement yaws (cert yield uneven; kept for coverage)


def _gen(fam: str, thetas: np.ndarray, theta_src: list[str], yaws: tuple[float, ...], scenarios: tuple[str, ...],
         n_seeds: int, cfg: Any, conf: Any, obj: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    rig = _rig(object_spec=variant(fam).object_spec)
    rows: list[dict[str, Any]] = []
    n_handoff = n_deliv = n_attempt = 0
    cert_by_yaw: dict[float, list[int]] = {y: [0, 0] for y in yaws}
    for yaw_deg in yaws:
        yaw = math.radians(yaw_deg)
        for sid in scenarios:
            scen = scenario_by_id(sid)
            for seed in range(n_seeds):
                n_attempt += 1
                cert_by_yaw[yaw_deg][1] += 1
                h = reach_capture_at_yaw(rig, scen, seed, yaw, cfg, conf, obj)
                if h.record is not None:
                    continue
                cert_by_yaw[yaw_deg][0] += 1
                n_handoff += 1
                post_yaw = math.degrees(object_yaw(h.snap))
                x = [float(v) for v in np.asarray(h.x, np.float64)]
                for ti, th in enumerate(thetas):
                    s = _delivery_signals(h.snap, clip_theta(th))
                    n_deliv += int(s.k6)
                    rows.append({"handoff_family": fam, "scenario": sid, "seed": seed, "yaw_deg": yaw_deg,
                                 "post_grasp_yaw_deg": round(post_yaw, 4), "theta_idx": ti, "theta_family": theta_src[ti],
                                 "x": x, "theta": [float(t) for t in th], "k6": bool(s.k6), "dtz_mm": s.dtz_mm,
                                 "gap_closed": s.gap_closed, "safe": bool(s.safe)})
                print(f"[{fam} y{yaw_deg:4.0f}° {sid:20s} s{seed}] handoff#{n_handoff}: {len(thetas)}θ pos={n_deliv} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    _OUT.mkdir(parents=True, exist_ok=True)
    with (_OUT / f"orientation_dataset_{fam}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"family": fam, "n_attempts": n_attempt, "n_handoffs": n_handoff, "n_pairs": len(rows),
            "n_positive": n_deliv, "positive_rate": round(n_deliv / max(1, len(rows)), 4),
            "cert_by_yaw": {y: cert_by_yaw[y] for y in yaws}, "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    fam = sys.argv[1] if len(sys.argv) > 1 else "O5-R"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_scen = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    yaws = tuple(float(v) for v in sys.argv[4].split(",")) if len(sys.argv) > 4 else _DEFAULT_YAWS
    n_theta = int(sys.argv[5]) if len(sys.argv) > 5 else 0        # 0 ⇒ full pooled bank
    scenarios = _SCENARIOS[:n_scen]
    thetas, theta_src = _pooled_thetas()
    if n_theta:
        thetas, theta_src = thetas[:n_theta], theta_src[:n_theta]
    cfg, conf, obj = bc_context()
    print(f"R12.2-A2 dataset [{fam}]: yaws {yaws}° × {len(scenarios)} scen × {n_seeds} seeds × {len(thetas)}θ", flush=True)
    meta = _gen(fam, thetas, theta_src, yaws, scenarios, n_seeds, cfg, conf, obj)
    (_OUT / f"orientation_dataset_meta_{fam}.json").write_text(json.dumps(meta, indent=1))
    cert = {y: f"{c[0]}/{c[1]}" for y, c in meta["cert_by_yaw"].items()}
    print(f"\nR12.2-A2 DONE [{fam}]: {meta['n_pairs']} pairs from {meta['n_handoffs']}/{meta['n_attempts']} certified "
          f"handoffs, {round(meta['positive_rate'] * 100)}% positive, cert-by-yaw {cert}, "
          f"{meta['wall_s'] / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
