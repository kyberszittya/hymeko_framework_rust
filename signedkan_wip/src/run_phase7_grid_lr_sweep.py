"""Grid × lr-schedule sweep on Bitcoin Alpha.

Tests whether HSiKAN is under-using KAR expressivity (small grid) and/or
undertraining (fixed lr without anneal). If best-cell AUC moves
substantially above the grid=3/fixed-lr baseline, KAR is delivering
and the previous numbers were a hyperparameter limitation.

Grid: grid ∈ {3, 5, 7, 11} × lr_schedule ∈ {fixed, cosine} × 3 seeds.
Recipe: h=16, n_layers=2, max_k4=30k, n_epochs=200 (vs sweep's 80).
Bitcoin Alpha leaky baseline (features="all") for fast feedback;
no-leak rerun queued separately if best cell is interesting.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .run_phase2_mixed_arity import run_one_mixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase7_grid_lr_sweep.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--grids", nargs="+", type=int, default=[3, 5, 7, 11])
    ap.add_argument("--schedules", nargs="+", default=["fixed", "cosine"])
    ap.add_argument("--feature_edges", default="all")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(
        dataset="bitcoin_alpha",
        hidden=16, n_layers=2,
        n_epochs=args.n_epochs,
        arities=(3, 4),
        max_k3=0,
        max_k4=30_000,
        coef_smooth_lam=0.0, participation_lam=0.0,
        grad_clip=0.0, weight_decay=0.0,
        early_stopping=False, class_weighted=False,
        feature_edges=args.feature_edges,
    )

    cells = []
    for grid in args.grids:
        for sched in args.schedules:
            for seed in args.seeds:
                cells.append((grid, sched, seed))

    rows = []
    t_start = time.time()
    for i, (grid, sched, seed) in enumerate(cells):
        tag = f"grid={grid:>2d}  sched={sched:<6s}  seed={seed}"
        print(f"\n[{i+1:>2d}/{len(cells)}] {tag} ...", flush=True)
        t0 = time.time()
        try:
            r = run_one_mixed(seed=seed, grid=grid, lr_schedule=sched,
                                **common)
            r["grid"] = grid
            r["lr_schedule"] = sched
            rows.append(r)
            with out_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"  AUC={r['test_auc']:.4f}  "
                  f"F1m={r['test_f1_macro']:.4f}  "
                  f"alpha={['%.2f' % x for x in r['alpha']]}  "
                  f"{time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e!r}")

    print(f"\nTotal: {time.time()-t_start:.1f}s   results → {out_path}")

    print("\n=== Median across seeds ===")
    print(f"{'grid':>4s}  {'sched':<7s}  {'AUC_med':>8s}  {'F1m_med':>8s}  "
          f"{'std':>6s}")
    import statistics
    best = (-1.0, None)
    for grid in args.grids:
        for sched in args.schedules:
            cell = [r for r in rows if r["grid"] == grid
                     and r["lr_schedule"] == sched]
            if not cell:
                continue
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
            med = statistics.median(aucs)
            if med > best[0]:
                best = (med, (grid, sched))
            print(f"{grid:>4d}  {sched:<7s}  {med:>8.4f}  "
                  f"{statistics.median(f1ms):>8.4f}  {std:>6.4f}")
    print(f"\nBest cell: grid={best[1][0]}, sched={best[1][1]} → AUC={best[0]:.4f}")


if __name__ == "__main__":
    main()
