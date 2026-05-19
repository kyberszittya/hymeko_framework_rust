"""Arity sweep k=2..10 on a small signed graph (karate by default).

Single-arity HSiKAN at each k, 3 seeds. Plots AUC as a function of
hyperedge arity. The expectation from prior work: k=2 (raw edges)
should give a baseline; k=3 (Heider triads) should jump; higher k
extends the Davis-balance criterion. The αₖ ablation has already
shown that *mixing* arities concentrates weight on k=4/5 — but here
we test single arities in isolation.

k=1 is degenerate (a single vertex has no edges to define a balance
criterion from, and `node_embed` already provides per-vertex bias) —
the sweep starts at k=2 and goes up to k=10. On karate the
enumeration cost is trivial even at k=10.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .run_phase2_mixed_arity import run_one_mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="karate")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase7_arity_sweep.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--ks", nargs="+", type=int,
                    default=[2, 3, 4, 5, 6, 7, 8, 9, 10])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--max_per_arity", type=int, default=10_000)
    ap.add_argument("--lr_schedule", default="cosine")
    ap.add_argument("--feature_edges", default="all")
    ap.add_argument("--cell_timeout_s", type=int, default=300)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells = [(k, seed) for k in args.ks for seed in args.seeds]

    rows = []
    t_start = time.time()
    for i, (k, seed) in enumerate(cells):
        tag = f"k={k:>2d}  seed={seed}"
        print(f"\n[{i+1:>2d}/{len(cells)}] {tag} ...", flush=True)
        t0 = time.time()
        try:
            kwargs = dict(
                dataset=args.dataset, seed=seed,
                hidden=args.hidden, n_layers=2, grid=args.grid,
                n_epochs=args.n_epochs,
                arities=(k,),
                coef_smooth_lam=0.0, participation_lam=0.0,
                grad_clip=0.0, weight_decay=0.0,
                early_stopping=False, class_weighted=False,
                lr_schedule=args.lr_schedule,
                feature_edges=args.feature_edges,
            )
            # Per-arity cap: route through max_per_arity dict.
            if k >= 3:
                kwargs["max_per_arity"] = {k: args.max_per_arity}
            r = run_one_mixed(**kwargs)
            r["k"] = k
            rows.append(r)
            with out_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  AUC={r['test_auc']:.4f}  "
                  f"F1m={r['test_f1_macro']:.4f}  "
                  f"alpha={['%.2f' % x for x in r['alpha']]}  "
                  f"sizes={r['n_tuples_per_arity']}  "
                  f"{time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e!r}")
            with out_path.open("a") as f:
                f.write(json.dumps(dict(
                    k=k, seed=seed, status="error", error=repr(e),
                )) + "\n")

    print(f"\nTotal: {time.time()-t_start:.1f}s   results → {out_path}")
    print("\n=== Median across seeds (sorted by k) ===")
    print(f"{'k':>3s}  {'AUC_med':>8s}  {'F1m_med':>8s}  {'std':>6s}  {'n_tuples':>9s}")
    import statistics
    for k in sorted(set(args.ks)):
        cell = [r for r in rows if r.get("k") == k]
        if not cell:
            continue
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        sizes = [list(r.get("n_tuples_per_arity", {}).values())[0]
                  if r.get("n_tuples_per_arity") else 0
                  for r in cell]
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        med_size = int(statistics.median(sizes))
        print(f"{k:>3d}  {statistics.median(aucs):>8.4f}  "
              f"{statistics.median(f1ms):>8.4f}  {std:>6.4f}  "
              f"{med_size:>9,d}")


if __name__ == "__main__":
    main()
