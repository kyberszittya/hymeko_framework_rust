"""Multi-layer SignedKAN, take 2: Jumping Knowledge + sum aggregation.

Addresses the two likely failure modes of the first multi-layer
ablation (negative result in `multilayer_ec.json`):

  1. Mean-pool inter-layer aggregation (canonical oversmoothing
     operator) → try sum-pool.
  2. Classifier sees only the last layer's triad embeddings → try
     Jumping Knowledge (Xu et al., 2018): aggregate across layers
     before the classifier.

All on EC recipe (early stopping + class-weighted BCE, $G=5$,
$200$ epochs, $h=32$, three seeds), Bitcoin Alpha and OTC.
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
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/multilayer_v2.json")
    args = ap.parse_args()

    # (n_layers, jk_mode, pool_mode, tag)
    configs = [
        (2, "last",   "mean", "L2-jkLAST-mean"),  # control = v1 result
        (2, "sum",    "mean", "L2-jkSUM-mean"),
        (2, "last",   "sum",  "L2-jkLAST-sum"),
        (2, "sum",    "sum",  "L2-jkSUM-sum"),
        (2, "concat", "sum",  "L2-jkCAT-sum"),
        (3, "sum",    "sum",  "L3-jkSUM-sum"),
    ]

    results = []
    for n_layers, jk, pool, tag in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, args.hidden, seed,
                             args.n_epochs, lr=args.lr,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=args.val_every,
                             n_layers=n_layers,
                             spline_kinds=["bspline"] * n_layers,
                             pool_mode=pool,
                             jk_mode=jk)
                r["cfg"] = tag
                print(f"  {tag:18s} {dataset:14s} "
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
