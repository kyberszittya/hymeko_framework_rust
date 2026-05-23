"""Bilinear endpoint classifier ablation, EC recipe.

Adds a LayerScale-gated bilinear head $\\gamma h_u^\\top W h_v$ on
edge endpoints, alongside the existing linear-on-pooled-triads
classifier. $\\gamma$ initialised at $10^{-4}$ so the bilinear path
is dead at training start; optimiser scales it up only if helpful.
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
                    "signedkan_wip/experiments/results/bilinear_ec.json")
    args = ap.parse_args()

    # Patch gamma_init to be smaller than the default 1e-3.
    import signedkan_wip.src.core.bilinear_head as bh
    orig_b = bh.BilinearHead.__init__
    orig_l = bh.LowRankBilinearHead.__init__
    def patched_b(self, d, gamma_init=1e-4):
        orig_b(self, d, gamma_init=gamma_init)
    def patched_l(self, d, rank=4, init_scale=None, gamma_init=1e-4):
        orig_l(self, d, rank=rank, init_scale=init_scale,
               gamma_init=gamma_init)
    bh.BilinearHead.__init__ = patched_b
    bh.LowRankBilinearHead.__init__ = patched_l

    results = []
    for rank, tag in [(0, "EC+bilinear-full"),
                       (4, "EC+bilinear-r4"),
                       (2, "EC+bilinear-r2")]:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=5,
                             use_bilinear=True,
                             bilinear_rank=rank)
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
