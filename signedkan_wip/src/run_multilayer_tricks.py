"""Multi-layer SignedKAN tricks: LayerNorm between layers + weight
sharing across layers.

Tries to make multi-layer pay on the link-sign benchmark by adding
two standard deep-net stabilisations missing from the v1/v2/v3
sweeps:

  - LayerNorm on the per-vertex embedding between layers
  - Weight sharing — one shared SignedKANLayer applied $L$ times
    (recurrent multi-layer; no extra parameters per layer)
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
                    "signedkan_wip/experiments/results/multilayer_tricks.json")
    args = ap.parse_args()

    # (n_layers, jk, pool, layer_norm, share_weights, tag)
    configs = [
        # LayerNorm only
        (2, "concat", "sum",  True,  False, "L2-jkCAT-sum+LN"),
        (3, "concat", "sum",  True,  False, "L3-jkCAT-sum+LN"),
        # Weight sharing only
        (2, "concat", "sum",  False, True,  "L2-jkCAT-sum+share"),
        (3, "concat", "sum",  False, True,  "L3-jkCAT-sum+share"),
        # Combined
        (2, "concat", "sum",  True,  True,  "L2-jkCAT-sum+LN+share"),
        (3, "concat", "sum",  True,  True,  "L3-jkCAT-sum+LN+share"),
        # Recurrent-style L=4 with shared weights (cheap to try since
        # parameter count is equivalent to L=1).
        (4, "concat", "sum",  True,  True,  "L4-jkCAT-sum+LN+share"),
    ]

    results = []
    for n_layers, jk, pool, ln, share, tag in configs:
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
                             layer_norm_between=ln,
                             share_weights=share)
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
