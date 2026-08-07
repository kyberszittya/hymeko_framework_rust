"""R12 / HSiKAN-1 Phase 0 — transportability dataset generator.

Emits the train-only θ×handoff supervision the structured critic learns from: for one object family, re-acquire
handoff snapshots on a scenario×seed grid and apply the family's θ-bank to EACH — the scaled θ×handoff matrix —
producing (x, θ) → (K6, dtz, safe) rows. One handoff carries many θ, so a modest grid yields hundreds of pairs with
NO new physical scenarios. Rows carry the scenario id so the downstream split is SCENARIO-LEVEL (E1 unseen scenario,
E2 unseen family) — same-handoff θ-pairs never leak train↔test.

Run:  python -m hymeko_rl.experiments.r12_hsikan1_dataset <variant_id> [n_seeds] [n_theta] [scenario_slice]
      (defaults: O4-S, 3 seeds, 30 θ, all scenarios) — a small run is the §11 cost probe.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import clip_theta
from hymeko_rl.coin_delivery.exact_zero_composition import _delivery_signals, reach_capture_descriptor
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

_OUT = Path("reports/2026-08-07-r12-hsikan1")
# Certifying-band scenario grid (fixed across families; center excluded), with (coin,target) diversity.
_SCENARIOS = ("bank_c2_+0.015_+0.000", "bank_c2_+0.025_+0.000", "bank_c3_r6_a+15", "bank_c3_r7_a-15",
              "bank_c2_+0.015_-0.015", "bank_c1_+0.01_+0.00")
# Per-family θ-bank source (delivering teacher θ). Box reuses the dense bank; others regenerate (later phases).
_BANK = {"O4-S": Path("reports/2026-08-06-r11-7a-u6b-box-pilot/bank_dense.json")}


def _load_thetas(variant_id: str, n_theta: int) -> np.ndarray:
    src = _BANK.get(variant_id)
    if src is None or not src.exists():
        raise FileNotFoundError(f"no θ-bank for {variant_id} (expected {src}); regenerate the family bank first")
    thetas = np.asarray([s["theta"] for s in json.loads(src.read_text())["samples"]], np.float64)
    return thetas if n_theta >= len(thetas) else thetas[np.linspace(0, len(thetas) - 1, n_theta).astype(int)]


def main() -> int:
    variant_id = sys.argv[1] if len(sys.argv) > 1 else "O4-S"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_theta = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    scen_slice = int(sys.argv[4]) if len(sys.argv) > 4 else len(_SCENARIOS)
    scenarios = _SCENARIOS[:scen_slice]

    thetas = _load_thetas(variant_id, n_theta)
    cfg, conf, obj = bc_context()
    t0 = time.perf_counter()
    rig = _rig(object_spec=variant(variant_id).object_spec)
    print(f"R12 dataset {variant_id}: {len(scenarios)} scen × {n_seeds} seeds × {len(thetas)} θ "
          f"(rig {time.perf_counter() - t0:.0f}s)", flush=True)

    rows: list[dict[str, Any]] = []
    n_handoff = n_deliv = 0
    for sid in scenarios:
        scen = scenario_by_id(sid)
        for seed in range(n_seeds):
            h = reach_capture_descriptor(rig, scen, seed, cfg, conf, obj)
            if h.record is not None:
                continue                                     # no certified handoff here
            n_handoff += 1
            x = [float(v) for v in np.asarray(h.x, np.float64)]
            th0 = time.perf_counter()
            for ti, th in enumerate(thetas):
                s = _delivery_signals(h.snap, clip_theta(th))
                n_deliv += int(s.k6)
                rows.append({"variant": variant_id, "scenario": sid, "seed": seed, "theta_idx": ti,
                             "x": x, "theta": [float(t) for t in th], "k6": bool(s.k6),
                             "dtz_mm": s.dtz_mm, "gap_closed": s.gap_closed, "safe": bool(s.safe)})
            print(f"[{sid:22s} s{seed}] handoff#{n_handoff}: {len(thetas)} θ rolled, "
                  f"positives so far={n_deliv} ({time.perf_counter() - th0:.0f}s)", flush=True)

    _OUT.mkdir(parents=True, exist_ok=True)
    shard = _OUT / f"dataset_{variant_id}.jsonl"
    with shard.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    wall = time.perf_counter() - t0
    per_pair = wall / max(1, len(rows))
    meta = {"variant": variant_id, "n_scenarios": len(scenarios), "n_seeds": n_seeds, "n_theta": len(thetas),
            "n_handoffs": n_handoff, "n_pairs": len(rows), "n_positive": n_deliv,
            "positive_rate": round(n_deliv / max(1, len(rows)), 4), "wall_s": round(wall, 1),
            "sec_per_pair": round(per_pair, 3), "scenarios": list(scenarios)}
    (_OUT / f"dataset_{variant_id}_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\nDATASET {variant_id}: {len(rows)} pairs from {n_handoff} handoffs, {n_deliv} positive "
          f"({meta['positive_rate']*100:.1f}%), {wall/60:.1f} min ({per_pair:.2f}s/pair). wrote {shard}", flush=True)
    # §11 full-run reconcile (4 families, richer grid)
    full = per_pair * (4 * len(_SCENARIOS) * 5 * 30) / 3600.0
    print(f"§11 full 4-family estimate (all scen × 5 seeds × 30 θ): ~{full:.1f} h at this per-pair cost", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
