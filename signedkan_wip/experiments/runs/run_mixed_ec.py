"""Mixed-spline ablation on the EC recipe: B-spline / Catmull-Rom
mixed across inner and outer activations within the SignedKAN
layer.

Tests two intermediate points between full B-spline and full
Catmull-Rom:
  - "bspline_cr" — B-spline inner, Catmull-Rom outer
  - "cr_bspline" — Catmull-Rom inner, B-spline outer

Conjecture: keeping B-spline on the inner (where smoothness affects
node-embedding gradients via the dominant parameter pool) and using
Catmull-Rom on the outer should preserve most of the accuracy at
roughly half the speedup of full CR.
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
                    "signedkan_wip/experiments/results/mixed_ec.json")
    args = ap.parse_args()

    results = []
    for spline_kind in ("bspline_cr", "cr_bspline"):
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, args.hidden, seed,
                             args.n_epochs, lr=args.lr,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=args.val_every,
                             spline_kind=spline_kind)
                r["cfg"] = f"EC-{spline_kind}"
                print(f"  {spline_kind:12s} {dataset:14s} "
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
