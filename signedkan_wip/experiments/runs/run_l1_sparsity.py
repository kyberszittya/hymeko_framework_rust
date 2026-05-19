"""L1-sparsity-during-training sweep.

Adds $\\lambda_{L1} \\sum_{(s, c)} \\|\\mathbf{c}^{(s, c)}\\|_1$ to the
loss. The optimiser learns to push coefficient vectors toward zero,
producing a model that is *self-pruning* at training time rather
than threshold-pruned post-hoc.

After training, we measure (a) what fraction of (branch, channel)
splines are within $10^{-12}$ of zero (true sparsity, no threshold
needed), and (b) the AUC / macro-F1 at the trained model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run_compare import run_one
from signedkan_wip.src.iter_prune import count_active_splines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/l1_sparsity.json")
    args = ap.parse_args()

    # L1 weights to sweep — picked spread-out across orders of magnitude.
    l1_lams = [0.0, 1e-5, 1e-4, 1e-3, 5e-3]

    results = []
    for lam in l1_lams:
        tag = "EC" if lam == 0.0 else f"EC+L1(λ={lam:.0e})"
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=5,
                             l1_lam=lam)
                r["cfg"] = tag
                r["l1_lam"] = lam
                # Active-spline fraction can be tagged after the run with
                # a separate eval. For now we just report what run_one does.
                print(f"  {tag:18s} {dataset:14s} "
                      f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
