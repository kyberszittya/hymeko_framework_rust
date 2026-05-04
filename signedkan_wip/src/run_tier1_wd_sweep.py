"""Tier 1 (B) — focused weight_decay sweep at the HSiKAN canonical recipe.

Sweep weight_decay ∈ {1e-5 (base), 1e-4, 1e-3, 5e-3} on bitcoin_alpha
and bitcoin_otc, 3 seeds per cell. All other knobs locked at
`HighwaySignedKAN.recommended_training_recipe()` + `catmull_rom`.

n_epochs=120 (matches the GA `--n_epochs` default in
`run_hsikan_genetic.py`). Optimizer = Adam (the canonical baseline);
the AdamW question is Tier 2 (G).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .highway_signedkan import HighwaySignedKAN
from .run_compare import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/tier1_wd_sweep.json")
    args = ap.parse_args()

    base = HighwaySignedKAN.recommended_training_recipe()
    base = {**base, "spline_kind": "catmull_rom"}

    wd_grid = [1e-5, 1e-4, 1e-3, 5e-3]
    runs = []
    t_total = time.time()
    for wd in wd_grid:
        for dataset in args.datasets:
            for seed in args.seeds:
                kwargs = {**base, "weight_decay": wd}
                r = run_one(
                    "signedkan", dataset, hidden=32, seed=seed,
                    n_epochs=args.n_epochs, lr=5e-2, **kwargs,
                )
                r["wd"] = wd
                print(f"  wd={wd:>7.0e}  {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    # Per-cell median.
    summary = {}
    for wd in wd_grid:
        for dataset in args.datasets:
            cell = [r for r in runs if r["wd"] == wd and r["dataset"] == dataset]
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            summary[f"wd={wd:.0e}|{dataset}"] = {
                "auc_med":   round(statistics.median(aucs), 4),
                "f1m_med":   round(statistics.median(f1ms), 4),
                "auc_seeds": [round(a, 4) for a in aucs],
                "f1m_seeds": [round(f, 4) for f in f1ms],
                "n_seeds":   len(cell),
            }

    out = {
        "runs": runs,
        "summary": summary,
        "wall_clock_s": round(time.time() - t_total, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(runs)} runs in "
          f"{out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
