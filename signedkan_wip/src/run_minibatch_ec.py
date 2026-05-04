"""Step 2b: balanced minibatch sampler.

Replaces class-weighted BCE (Step 2a, EC recipe) with SGCN-style
balanced minibatching: each gradient step samples batch_size/2
positive and batch_size/2 negative training edges with replacement.
The forward pass over triads stays full; only the per-edge loss is
subsampled.

Tests the residual ~0.075 AUC gap to SGCN, which we attributed to
the optimisation recipe rather than architecture.

Configurations:
  EM     : minibatch + early stop, single layer, B-spline
  EM-jk  : minibatch + early stop, L=2 jkCAT-sum-BS-BS
  EM-cw  : minibatch + early stop + class-weighted (test if both stack)

All on h=32, lr=5e-2, 200 epochs × 20 steps, batch_size=256, three seeds.
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
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps-per-epoch", type=int, default=20)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/minibatch_ec.json")
    args = ap.parse_args()

    configs = [
        # (tag, n_layers, spline_kinds, jk, pool, class_weighted)
        ("EM",       1, None,                    "last",   "mean", False),
        ("EM-jk",    2, ["bspline", "bspline"],  "concat", "sum",  False),
        ("EM-cw",    1, None,                    "last",   "mean", True),
    ]

    results = []
    for tag, n_layers, kinds, jk, pool, cw in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                kwargs = dict(
                    early_stopping=True,
                    class_weighted=cw,
                    grid=5,
                    val_every=args.val_every,
                    minibatch=True,
                    batch_size=args.batch_size,
                    steps_per_epoch=args.steps_per_epoch,
                    n_layers=n_layers,
                    pool_mode=pool,
                    jk_mode=jk,
                )
                if kinds is not None:
                    kwargs["spline_kinds"] = kinds
                r = run_one("signedkan", dataset, args.hidden, seed,
                             args.n_epochs, lr=args.lr, **kwargs)
                r["cfg"] = tag
                print(f"  {tag:7s} {dataset:14s} "
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
