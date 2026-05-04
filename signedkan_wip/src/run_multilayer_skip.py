"""Multi-layer SignedKAN combined with heterogeneous skip-connection
placement.

The L=2 jkCAT-sum BS-BS recipe was the multi-layer win (+0.012
macro-F1 on OTC, AUC roughly tied) and (inner=residual, outer=none)
was the L=1 win (+0.0033 AUC OTC, +0.017 F1m, zero added params).
Test whether they compose.
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
                    "signedkan_wip/experiments/results/multilayer_skip.json")
    args = ap.parse_args()

    configs = [
        # (n_layers, inner_skip, outer_skip, jk_mode, pool_mode, tag)
        (2, "residual", "none",     "concat", "sum", "L2-jkCAT-sum+skip(R,N)"),
        (2, "residual", "highway",  "concat", "sum", "L2-jkCAT-sum+skip(R,H)"),
        (2, "none",     "residual", "concat", "sum", "L2-jkCAT-sum+skip(N,R)"),
        (3, "residual", "none",     "concat", "sum", "L3-jkCAT-sum+skip(R,N)"),
    ]

    results = []
    for n_layers, inner, outer, jk, pool, tag in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=5,
                             n_layers=n_layers,
                             spline_kinds=["bspline"] * n_layers,
                             pool_mode=pool,
                             jk_mode=jk,
                             inner_skip=inner,
                             outer_skip=outer)
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
