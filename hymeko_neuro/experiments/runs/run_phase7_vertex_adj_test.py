"""Quick honest no-leak test with vertex_adjacency M_e on multiple datasets.

Each cell: vertex_adjacency M_e + feature_edges=train_val (full leak removal).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

from .run_phase2_mixed_arity import run_one_mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="karate")
    ap.add_argument("--arities", nargs="+", type=int, default=[3])
    ap.add_argument("--max_per_arity", type=int, default=5000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--grid", type=int, default=5)
    args = ap.parse_args()

    arities = tuple(args.arities)
    max_per = {k: args.max_per_arity for k in arities}
    print(f"Dataset: {args.dataset}  arities={arities}  "
          f"max_per_arity={args.max_per_arity}", flush=True)
    aucs = []
    for seed in args.seeds:
        t0 = time.time()
        r = run_one_mixed(args.dataset, seed=seed,
                           hidden=args.hidden, n_layers=2, grid=args.grid,
                           n_epochs=args.n_epochs,
                           arities=arities,
                           max_per_arity=max_per,
                           coef_smooth_lam=0.0, participation_lam=0.0,
                           grad_clip=0.0, weight_decay=0.0,
                           early_stopping=False, class_weighted=False,
                           lr_schedule="cosine",
                           feature_edges="train_val",
                           m_e_mode="vertex_adjacency")
        aucs.append(r["test_auc"])
        print(f"  seed={seed}  AUC={r['test_auc']:.4f}  "
              f"alpha={['%.2f' % x for x in r['alpha']]}  "
              f"sizes={r['n_tuples_per_arity']}  "
              f"{time.time()-t0:.1f}s",
              flush=True)
    if len(aucs) >= 2:
        print(f"  ===> median={statistics.median(aucs):.4f}  "
              f"std={statistics.stdev(aucs):.4f}", flush=True)


if __name__ == "__main__":
    main()
