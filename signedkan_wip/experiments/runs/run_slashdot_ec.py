"""Step 3: extend the EC recipe to Slashdot.

Slashdot is much larger than Bitcoin (82k nodes, 549k edges, 580k
triads, vs Bitcoin Alpha's 3.7k nodes / 24k edges / 22k triads).
Tests whether the gap-closing recipes (L=1 EC, L=2 jkCAT-sum BS-BS)
generalise to a benchmark where (i) depth-friendly architectures
historically shine and (ii) parameter budget shifts toward node
embeddings rather than spline coefficients.

Skipping minibatch on Slashdot — at 580k triads × 20 steps/epoch ×
200 epochs = 4M forward passes per run, that's prohibitive.
Full-batch only here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_compare import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["slashdot"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/slashdot_ec.json")
    args = ap.parse_args()

    # Slashdot is too big for B-spline on a 7.6GB GPU even at h=16
    # (Cox-de Boor recursion intermediates overflow). Catmull-Rom works
    # at h=8 for L=1; L=2 needs h=4 to fit. Per-config hidden override.
    configs = [
        # (tag, n_layers, kinds, jk, pool, hidden)
        ("EC-CR",           1, None,                            "last",   "mean", 8),
        ("L2-jkCAT-sum-CR", 2, ["catmull_rom", "catmull_rom"],  "concat", "sum",  4),
    ]

    results = []
    for tag, n_layers, kinds, jk, pool, hidden_override in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                kwargs = dict(
                    early_stopping=True,
                    class_weighted=True,
                    grid=5,
                    val_every=args.val_every,
                    n_layers=n_layers,
                    pool_mode=pool,
                    jk_mode=jk,
                    spline_kind="catmull_rom",
                )
                if kinds is not None:
                    kwargs["spline_kinds"] = kinds
                r = run_one("signedkan", dataset, hidden_override, seed,
                             args.n_epochs, lr=args.lr, **kwargs)
                r["cfg"] = tag
                print(f"  {tag:14s} {dataset:10s} "
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
