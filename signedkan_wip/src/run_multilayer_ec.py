"""Multi-layer SignedKAN ablation on the EC recipe.

L=2 sweep (whether stacking helps at all):
  - (bspline, bspline)
  - (bspline, catmull_rom)
  - (catmull_rom, bspline)
  - (bspline, kochanek_bartels)

L=3 sweep (boundary-vs-hidden spline allocation):
  - (bspline, bspline, bspline)
  - (bspline, catmull_rom, bspline)    -- user hypothesis
  - (bspline, kochanek_bartels, bspline)

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
                    "signedkan_wip/experiments/results/multilayer_ec.json")
    args = ap.parse_args()

    configs = [
        # L=2
        (2, ["bspline", "bspline"]),
        (2, ["bspline", "catmull_rom"]),
        (2, ["catmull_rom", "bspline"]),
        (2, ["bspline", "kochanek_bartels"]),
        # L=3 — user hypothesis
        (3, ["bspline", "bspline", "bspline"]),
        (3, ["bspline", "catmull_rom", "bspline"]),
        (3, ["bspline", "kochanek_bartels", "bspline"]),
    ]

    results = []
    for n_layers, kinds in configs:
        tag = f"L{n_layers}-" + "-".join(
            {"bspline": "BS", "catmull_rom": "CR",
             "kochanek_bartels": "KB"}[k] for k in kinds
        )
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, args.hidden, seed,
                             args.n_epochs, lr=args.lr,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=args.val_every,
                             n_layers=n_layers,
                             spline_kinds=kinds)
                r["cfg"] = tag
                print(f"  {tag:20s} {dataset:14s} "
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
