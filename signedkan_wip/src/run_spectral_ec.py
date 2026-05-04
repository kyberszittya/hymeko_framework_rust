"""Spectral-init ablation: top-k signed-Laplacian eigenvectors as
node-embedding initialisation prior, EC recipe.

Tests whether seeding ``node_embed.weight`` with the top-$k$
smallest eigenvectors of the symmetric normalised signed Laplacian
$L_s = I - D_s^{-1/2} A_s D_s^{-1/2}$ accelerates training and/or
moves AUC.

The structural prior is interpretable and free at inference: it
only changes initialisation, not the architecture or the loss.
HyMeKo can in principle generate this initialisation tensor as a
front-matter description; the loop here just builds it directly.
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
                    "signedkan_wip/experiments/results/spectral_ec.json")
    args = ap.parse_args()

    configs = [
        # (tag, n_layers, kinds, jk, pool, spectral_k)
        ("EC+spec(k=8)",       1, None, "last",   "mean", 8),
        ("EC+spec(k=16)",      1, None, "last",   "mean", 16),
        ("EC+spec(k=32)",      1, None, "last",   "mean", 32),  # all dims spectral
        ("L2-jkCAT-sum+spec(k=16)", 2, ["bspline", "bspline"],
                                              "concat", "sum",  16),
    ]

    results = []
    for tag, n_layers, kinds, jk, pool, k in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                kwargs = dict(
                    early_stopping=True,
                    class_weighted=True,
                    grid=5,
                    val_every=5,
                    n_layers=n_layers,
                    pool_mode=pool,
                    jk_mode=jk,
                    spectral_init=True,
                    spectral_k=k,
                )
                if kinds is not None:
                    kwargs["spline_kinds"] = kinds
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
