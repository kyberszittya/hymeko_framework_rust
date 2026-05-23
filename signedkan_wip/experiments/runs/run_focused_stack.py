"""Focused stacking sweep: multi-layer + EC + R2 + JK-concat
+ heterogeneous skip placement. Drops the entropy regulariser and
optionally drops LayerNorm + weight-sharing to isolate which piece
of the kitchen sink fought R2.

Configurations tested:
  EC                                      : single-layer baseline
  EC+R2                                   : single-layer + R2 (the morning recipe)
  L=2 + JK-concat + R2 + skip(R,N)        : multi-layer, no LN, no share
  L=3 + JK-concat + R2 + skip(R,N)
  L=2 + JK + LN + share + R2 + skip(R,N)  : multi-layer with stabilisation
  L=3 + JK + LN + share + R2 + skip(R,N)
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
                    "signedkan_wip/experiments/results/focused_stack.json")
    args = ap.parse_args()

    base = dict(
        early_stopping=True,
        class_weighted=True,
        grid=5,
        val_every=5,
        participation_lam=0.05,
        inner_skip="residual",
        outer_skip="none",
        pool_mode="sum",
        jk_mode="concat",
    )

    configs = [
        ("L2-multi+R2+skip",       {**base, "n_layers": 2,
                                    "spline_kinds": ["bspline"]*2}),
        ("L3-multi+R2+skip",       {**base, "n_layers": 3,
                                    "spline_kinds": ["bspline"]*3}),
        ("L2-multi+LN+share+R2+skip",  {**base, "n_layers": 2,
                                        "spline_kinds": ["bspline"]*2,
                                        "layer_norm_between": True,
                                        "share_weights": True}),
        ("L3-multi+LN+share+R2+skip",  {**base, "n_layers": 3,
                                        "spline_kinds": ["bspline"]*3,
                                        "layer_norm_between": True,
                                        "share_weights": True}),
    ]

    results = []
    for tag, kwargs in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2, **kwargs)
                r["cfg"] = tag
                print(f"  {tag:30s} {dataset:14s} "
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
