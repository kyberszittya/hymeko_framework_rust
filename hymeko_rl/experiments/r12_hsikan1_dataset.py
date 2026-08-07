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
_FAMILIES = ("O0", "O1-L", "O2-M", "O4-S")
_BOX_DENSE = Path("reports/2026-08-06-r11-7a-u6b-box-pilot/bank_dense.json")
_CHARZ = Path("reports/2026-08-07-r11-7b-multiobject")


def _pooled_thetas() -> "tuple[np.ndarray, list[str]]":
    """POOLED candidate-θ bank across families (reuse existing teacher θ — no new bank-gen): the box's dense bank
    (66) plus each family's characterization teacher θ. Applied to EVERY family's handoffs, this yields the
    cross-family transfer data the critic needs (a θ from one family tested on another's handoff)."""
    thetas: list[list[float]] = []
    src: list[str] = []
    for s in json.loads(_BOX_DENSE.read_text())["samples"]:
        thetas.append(s["theta"])
        src.append("O4-S")
    for fam in _FAMILIES:
        p = _CHARZ / f"characterization_{fam}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text())["records"]:
            if r.get("captured") and r.get("teacher_k6"):
                thetas.append(r["theta"])
                src.append(fam)
    arr = np.asarray(thetas, np.float64)
    return arr, src


def _gen_family(fam: str, thetas: np.ndarray, theta_src: list[str], scenarios: tuple, n_seeds: int,
                cfg: Any, conf: Any, obj: Any) -> dict[str, Any]:
    """Acquire this family's handoffs and apply the POOLED θ-bank to each → rows. Handoffs are this family's;
    each θ carries its source family so cross-family transfer is labelled."""
    t0 = time.perf_counter()
    rig = _rig(object_spec=variant(fam).object_spec)
    rows: list[dict[str, Any]] = []
    n_handoff = n_deliv = 0
    for sid in scenarios:
        scen = scenario_by_id(sid)
        for seed in range(n_seeds):
            h = reach_capture_descriptor(rig, scen, seed, cfg, conf, obj)
            if h.record is not None:
                continue
            n_handoff += 1
            x = [float(v) for v in np.asarray(h.x, np.float64)]
            for ti, th in enumerate(thetas):
                s = _delivery_signals(h.snap, clip_theta(th))
                n_deliv += int(s.k6)
                rows.append({"handoff_family": fam, "scenario": sid, "seed": seed, "theta_idx": ti,
                             "theta_family": theta_src[ti], "x": x, "theta": [float(t) for t in th],
                             "k6": bool(s.k6), "dtz_mm": s.dtz_mm, "gap_closed": s.gap_closed, "safe": bool(s.safe)})
            print(f"[{fam:5s} {sid:22s} s{seed}] handoff#{n_handoff}: {len(thetas)} θ, pos={n_deliv} "
                  f"({time.perf_counter() - t0:.0f}s cum)", flush=True)
    _OUT.mkdir(parents=True, exist_ok=True)
    with (_OUT / f"dataset_{fam}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return {"family": fam, "n_handoffs": n_handoff, "n_pairs": len(rows), "n_positive": n_deliv,
            "positive_rate": round(n_deliv / max(1, len(rows)), 4), "wall_s": round(time.perf_counter() - t0, 1)}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "O4-S"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    scenarios = _SCENARIOS
    thetas, theta_src = _pooled_thetas()
    families = list(_FAMILIES) if which == "ALL" else [which]
    src_counts = {f: theta_src.count(f) for f in sorted(set(theta_src))}
    print(f"R12 dataset [{','.join(families)}]: pooled bank {len(thetas)} θ "
          f"({src_counts}) × {len(scenarios)} scen × {n_seeds} seeds", flush=True)
    cfg, conf, obj = bc_context()
    t0 = time.perf_counter()
    metas = [_gen_family(fam, thetas, theta_src, scenarios, n_seeds, cfg, conf, obj) for fam in families]
    total_pairs = sum(m["n_pairs"] for m in metas)
    meta = {"families": families, "n_theta_pooled": len(thetas), "theta_src_counts": {f: theta_src.count(f) for f in set(theta_src)},
            "n_seeds": n_seeds, "scenarios": list(scenarios), "per_family": metas,
            "total_pairs": total_pairs, "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / f"dataset_meta_{which}.json").write_text(json.dumps(meta, indent=1))
    per_fam = [(m["family"], m["n_pairs"], round(m["positive_rate"] * 100)) for m in metas]
    print(f"\nR12 DATASET DONE [{which}]: {total_pairs} pairs, per-family(fam,pairs,pos%) {per_fam}, "
          f"{meta['wall_s'] / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
