"""Refined entropy regulariser composed with the L=3 + LN + share +
R2 + skip(R,N) recipe.

Tests whether the entropy term — which fought R2 in the original
kitchen sink — composes when given the two refinements:
  1. KL update normalised by log2(rank(A)) so eta has scale-invariant
     meaning across architectures.
  2. EMA momentum on lam_eff to smooth schedule across steps and
     prevent thrashing on transient spectral spikes.
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
                    "signedkan_wip/experiments/results/refined_entropy.json")
    args = ap.parse_args()

    base = dict(
        early_stopping=True,
        class_weighted=True,
        grid=5,
        val_every=5,
        participation_lam=0.05,
        n_layers=3,
        spline_kinds=["bspline"] * 3,
        pool_mode="sum",
        jk_mode="concat",
        layer_norm_between=True,
        share_weights=True,
        inner_skip="residual",
        outer_skip="none",
        entropy_lam0=0.01,
        entropy_target=0.5,
        entropy_eta=5.0,
    )

    configs = [
        ("recipe+oldH",         {**base, "entropy_kl_normalized": False,
                                  "entropy_momentum": 0.0}),
        ("recipe+H(klnorm)",    {**base, "entropy_kl_normalized": True,
                                  "entropy_momentum": 0.0}),
        ("recipe+H(mom)",       {**base, "entropy_kl_normalized": False,
                                  "entropy_momentum": 0.9}),
        ("recipe+H(klnorm+mom)", {**base, "entropy_kl_normalized": True,
                                  "entropy_momentum": 0.9}),
    ]

    results = []
    for tag, kwargs in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2, **kwargs)
                r["cfg"] = tag
                print(f"  {tag:24s} {dataset:14s} "
                      f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"H={r['last_h_norm']:.3f}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
