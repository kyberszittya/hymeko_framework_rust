"""Exhaustive cortical sweep: 5 seeds × 12 architectures × 2 backbones
(2026-05-20 overnight queue item from Phase 12.5).

Produces a JSONL row per (backbone, architecture, seed) carrying:

  * the P-graph unit names (the "ABB-would-pick-this-if-weighted-X" tag)
  * the mapped CorticalBenchmarkExperiment kwargs
  * the per-ROI r² and noise-corrected r²
  * wall-clock for the run

Then computes a Pareto-frontier summary (architectural cost vs mean
r²) per backbone.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _all_architectures(backbone: str) -> list[list[str]]:
    """Enumerate the full 3×2×2 = 12 cortical-search-space architectures
    for the given backbone (`resnet` or `gomb`)."""
    widths = (
        ["d_hidden_4", "d_hidden_8", "d_hidden_16"]
        if backbone == "resnet"
        else ["gomb_d4", "gomb_d8", "gomb_d16"]
    )
    binnings = ["binning_shallow", "binning_deep"]
    plss = ["pls_25", "pls_50"]
    return [list(c) for c in itertools.product(widths, binnings, plss)]


def _arch_cost(units: list[str]) -> int:
    """Scalar cost mirror of the .hymeko fixtures."""
    cost_map = {
        "d_hidden_4": 4, "d_hidden_8": 8, "d_hidden_16": 16,
        "gomb_d4": 4, "gomb_d8": 8, "gomb_d16": 16,
        "binning_shallow": 3, "binning_deep": 9,
        "pls_25": 5, "pls_50": 10,
    }
    return sum(cost_map[u] for u in units)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--backbones", type=str, nargs="+",
                    default=["resnet", "gomb"])
    ap.add_argument("--n-images", type=int, default=30)
    ap.add_argument("--n-subjects", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--n-cv-folds", type=int, default=4)
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "reports" /
                            "cortical_exhaustive_2026_05_20.jsonl")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore")

    from signedkan_wip.src.cortical_pgraph_mapping import (
        merge_structure_knobs, benchmark_kwargs,
    )
    from signedkan_wip.experiments.runs.run_cortical_msg_sweep import (
        _run_one_benchmark_seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    total = sum(len(_all_architectures(b)) * len(args.seeds)
                for b in args.backbones)
    print(f"Running {total} cortical-benchmark training cells "
          f"({len(args.backbones)} backbones × "
          f"{len(_all_architectures(args.backbones[0]))} archs × "
          f"{len(args.seeds)} seeds).", file=sys.stderr)

    t0 = time.time()
    cell = 0
    with args.output.open("w") as fh:
        for backbone in args.backbones:
            for units in _all_architectures(backbone):
                merged = merge_structure_knobs(units)
                base = {
                    "n_images": args.n_images,
                    "n_subjects": args.n_subjects,
                    "image_size": args.image_size,
                    "in_channels": 1,
                    "snr": 0.3,
                    "n_cv_folds": args.n_cv_folds,
                }
                for seed in args.seeds:
                    cell += 1
                    kw = benchmark_kwargs(seed=seed, structure=merged, base=base)
                    t_cell = time.time()
                    try:
                        r = _run_one_benchmark_seed(seed=seed, kw=kw)
                    except Exception as exc:  # noqa: BLE001
                        r = {"error": repr(exc)}
                    wall = time.time() - t_cell
                    row = {
                        "backbone": backbone,
                        "pgraph_units": units,
                        "scalar_cost": _arch_cost(units),
                        "d_hidden": kw["d_hidden"],
                        "binning": kw["binning"],
                        "n_pls": kw["n_pls_components"],
                        "seed": seed,
                        "wall_s": round(wall, 3),
                        **{k: v for k, v in r.items()
                           if k not in ("backbone", "d_hidden", "binning",
                                         "n_pls_components", "seed")},
                    }
                    fh.write(json.dumps(row, default=str) + "\n")
                    fh.flush()
                    rows.append(row)
                    if cell % 10 == 0:
                        print(f"  cell {cell}/{total} "
                              f"({backbone} {units} seed={seed}) "
                              f"wall={wall:.2f}s "
                              f"V1_r2={r.get('V1_r2', 'err'):.3f}"
                              if isinstance(r.get("V1_r2"), float) else
                              f"  cell {cell}/{total} ({backbone} ERROR)",
                              file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f} s "
          f"({elapsed/total:.2f} s/cell average)\n", file=sys.stderr)

    # ─── Pareto-frontier summary per backbone ─────────────────────
    def _aggregate(rs):
        v1 = [r["V1_r2"] for r in rs if "V1_r2" in r]
        v2 = [r["V2_r2"] for r in rs if "V2_r2" in r]
        v4 = [r["V4_r2"] for r in rs if "V4_r2" in r]
        if not v1:
            return None
        means = (statistics.fmean(v1), statistics.fmean(v2), statistics.fmean(v4))
        stds = (statistics.pstdev(v1), statistics.pstdev(v2), statistics.pstdev(v4))
        return means, stds

    print(json.dumps({
        "n_cells": len(rows),
        "wall_total_s": round(elapsed, 1),
        "output": str(args.output),
    }, indent=2))

    print("\n=== Per-architecture aggregates (3 ROIs averaged) ===",
          file=sys.stderr)
    summary: dict[str, list[dict[str, Any]]] = {}
    for backbone in args.backbones:
        print(f"\n[{backbone}]", file=sys.stderr)
        summary[backbone] = []
        for units in _all_architectures(backbone):
            cell_rows = [r for r in rows
                         if r["backbone"] == backbone and r["pgraph_units"] == units]
            agg = _aggregate(cell_rows)
            if agg is None:
                print(f"  {units}: errored", file=sys.stderr)
                continue
            (m1, m2, m4), (s1, s2, s4) = agg
            mean_all = (m1 + m2 + m4) / 3
            sigma_all = (s1 + s2 + s4) / 3
            cost = cell_rows[0]["scalar_cost"]
            entry = {
                "units": units, "scalar_cost": cost,
                "V1_r2": round(m1, 4), "V2_r2": round(m2, 4), "V4_r2": round(m4, 4),
                "V1_std": round(s1, 4), "V2_std": round(s2, 4), "V4_std": round(s4, 4),
                "mean_r2": round(mean_all, 4),
                "mean_sigma": round(sigma_all, 4),
            }
            summary[backbone].append(entry)
            print(f"  cost={cost:>2} units={units}\n"
                  f"      mean r² = {mean_all:.4f}, mean σ = {sigma_all:.4f}",
                  file=sys.stderr)

    # Save summary alongside JSONL.
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written → {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
