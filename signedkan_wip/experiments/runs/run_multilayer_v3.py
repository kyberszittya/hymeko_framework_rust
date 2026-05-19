"""Multi-layer v3: combine the two wins.

L2-jkCAT-sum was the macro-F1 winner of the multi-layer v2 sweep
(+0.009 Alpha, +0.012 OTC). The single-layer mixed-spline sweep
showed bspline_cr / cr_bspline beat pure BS or CR on macro-F1 by a
similar margin. Test whether the two effects compose: L=2 JK-concat
+ sum-pool with B-spline first layer and Catmull-Rom second layer
(or vice versa).
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
                    "signedkan_wip/experiments/results/multilayer_v3.json")
    args = ap.parse_args()

    configs = [
        # (n_layers, spline_kinds, jk, pool, tag)
        (2, ["bspline",     "catmull_rom"], "concat", "sum", "L2-CAT-sum-BS-CR"),
        (2, ["catmull_rom", "bspline"],     "concat", "sum", "L2-CAT-sum-CR-BS"),
        (2, ["catmull_rom", "catmull_rom"], "concat", "sum", "L2-CAT-sum-CR-CR"),
    ]

    results = []
    for n_layers, kinds, jk, pool, tag in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, args.hidden, seed,
                             args.n_epochs, lr=args.lr,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=args.val_every,
                             n_layers=n_layers,
                             spline_kinds=kinds,
                             pool_mode=pool,
                             jk_mode=jk)
                r["cfg"] = tag
                print(f"  {tag:22s} {dataset:14s} "
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
