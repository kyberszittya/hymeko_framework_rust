"""Accelerated Slashdot arity-gap test.

Three accelerations vs the original CPU-only run_phase7_slashdot_arity:
  1. GPU (training ~10× faster than CPU)
  2. max_k=150k (was 500k; AUC plateaus past ~100k)
  3. early_stop DFS (was reservoir; ~10²-10³× faster on k=5)

Per-cell wall-time: ~30-90s instead of 5-6 hours.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from .run_phase2_mixed_arity import run_one_mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase7_slashdot_fast.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=80)
    ap.add_argument("--max_k4", type=int, default=150_000)
    ap.add_argument("--max_k5", type=int, default=150_000)
    ap.add_argument("--max_k3", type=int, default=30_000)
    ap.add_argument("--cells", nargs="+", default=[
        "k34", "k4only", "k345", "k34_300k",
    ])
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cell_specs = {
        "k34":      dict(arities=(3, 4),
                          max_per_arity={3: args.max_k3, 4: args.max_k4}),
        "k4only":   dict(arities=(4,),
                          max_per_arity={4: args.max_k4}),
        "k345":     dict(arities=(3, 4, 5),
                          max_per_arity={3: args.max_k3,
                                          4: args.max_k4,
                                          5: args.max_k5}),
        "k34_300k": dict(arities=(3, 4),
                          max_per_arity={3: args.max_k3, 4: 300_000}),
    }

    common = dict(
        dataset="slashdot",
        hidden=16, n_layers=2, grid=3,
        n_epochs=args.n_epochs,
        coef_smooth_lam=0.0, participation_lam=0.0,
        grad_clip=0.0, weight_decay=0.0,
        early_stopping=False, class_weighted=False,
        cycle_batch_size=10_000,
        cycle_early_stop=True,                   # the speedup
        feature_edges="all",                     # original protocol (leaky)
    )

    rows = []
    t_start = time.time()
    for cell in args.cells:
        if cell not in cell_specs:
            print(f"unknown cell: {cell}; skipping", flush=True); continue
        spec = cell_specs[cell]
        for seed in args.seeds:
            tag = f"{cell:<10s} seed={seed}"
            print(f"[{tag}] ...", flush=True, end="")
            t0 = time.time()
            try:
                r = run_one_mixed(seed=seed, **common, **spec)
                r["cell"] = cell
                rows.append(r)
                with out_path.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"  AUC={r['test_auc']:.4f}  "
                      f"alpha={['%.2f' % x for x in r['alpha']]}  "
                      f"sizes={r['n_tuples_per_arity']}  "
                      f"{time.time()-t0:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"  FAILED: {e!r}", flush=True)

    print(f"\nTotal: {time.time()-t_start:.1f}s   results → {out_path}",
          flush=True)
    print("\n=== Median across seeds ===", flush=True)
    for cell in args.cells:
        cell_rows = [r for r in rows if r.get("cell") == cell]
        if not cell_rows:
            continue
        aucs = [r["test_auc"] for r in cell_rows]
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        n_arities = len(cell_rows[0]["alpha"])
        amed = [round(statistics.median(
            [r["alpha"][i] for r in cell_rows]), 2) for i in range(n_arities)]
        print(f"{cell:<10s}  AUC_med={statistics.median(aucs):.4f}  "
              f"std={std:.4f}  alpha_med={amed}",
              flush=True)


if __name__ == "__main__":
    main()
