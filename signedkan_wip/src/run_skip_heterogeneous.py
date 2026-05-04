"""Heterogeneous skip-connection ablation, EC recipe.

Tests CV-style "head / neck / spine" composition: skip connections
on some spline positions but not others, rather than uniform-everywhere.

The SignedKAN layer has two spline positions:
  - inner: per-vertex, pre-aggregation transform (the "spine")
  - outer: per-sign-aggregate transform (the "neck" between spine and
    classifier head)

We test the 3x3 = 9 combinations of {none, residual, highway} on each
position. The diagonal (uniform) and the (none, none) cell were
established earlier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_compare import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/skip_heterogeneous.json")
    args = ap.parse_args()

    # (inner_skip, outer_skip)
    configs = [
        # New heterogeneous combinations.
        ("none",     "highway"),     # head-clean / neck-gated  ← CV-style
        ("none",     "residual"),    # head-clean / neck-residual
        ("residual", "highway"),     # spine-residual / neck-gated
        ("highway",  "none"),        # spine-gated / neck-clean
        ("residual", "none"),        # spine-residual / neck-clean
        ("highway",  "residual"),    # spine-gated / neck-residual
    ]

    results = []
    for inner, outer in configs:
        tag = f"EC+skip(in={inner},out={outer})"
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=5,
                             inner_skip=inner,
                             outer_skip=outer)
                r["cfg"] = tag
                print(f"  {tag:42s} {dataset:14s} "
                      f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"params={r['n_params']:,}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
