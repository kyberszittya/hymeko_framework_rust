"""Catmull-Rom ablation on the EC recipe.

Tests whether replacing the cubic B-spline activation with a uniform
Catmull-Rom interpolating spline at the same grid $G=5$ closes or
widens the SGCN gap at lower parameter cost.

Catmull-Rom:
  - $G$ free parameters per channel (vs $G+k-1$ for B-spline cubic).
  - $C^1$ continuous (vs $C^2$ for B-spline cubic).
  - Interpolatory: curve passes through the control points.

Run on the EC recipe (early stopping + class-weighted BCE, $G=5$,
$200$ epochs) for direct comparison with the gap-sweep table.

Run:
  python -m signedkan_wip.src.run_catmull_ec
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
                    "signedkan_wip/experiments/results/catmull_ec.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one("signedkan", dataset, args.hidden, seed,
                         args.n_epochs, lr=args.lr,
                         early_stopping=True,
                         class_weighted=True,
                         grid=5,
                         val_every=args.val_every,
                         spline_kind="catmull_rom")
            r["cfg"] = "EC-CR"
            print(f"  EC-CR  signedkan   {dataset:14s} "
                  f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                  f"AUC={r['test_auc']:.4f}  "
                  f"F1_mac={r['test_f1_macro']:.4f}  "
                  f"params={r['n_params']:,}  "
                  f"{r['elapsed_s']:.1f}s")
            results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
