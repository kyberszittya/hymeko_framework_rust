"""Slashdot best-config no-leak rerun.

Recipe matches the sweep ceiling (h=16, max_k4=500k, L=2). Compares
``feature_edges=all`` (leaky transductive baseline — sweep median 0.7686)
to ``feature_edges=train_val`` (test edges held out of cycle structure).

The Bitcoin directed × leak experiment showed the undirected leak was
~0.09 AUC. Slashdot is denser (avg degree 13.4 vs Bitcoin's ~6.4), so
the leak might be smaller or larger; either way the no-leak number is
the publishable one.
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
                    "signedkan_wip/experiments/results/phase7_slashdot_noleak.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=80)
    ap.add_argument("--max_k4", type=int, default=500_000)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(
        dataset="slashdot",
        hidden=16, n_layers=2, grid=3,
        n_epochs=args.n_epochs,
        arities=(3, 4),
        max_k3=30_000,
        max_k4=args.max_k4,
        coef_smooth_lam=0.0, participation_lam=0.0,
        grad_clip=0.0, weight_decay=0.0,
        early_stopping=False, class_weighted=False,
        cycle_batch_size=10_000,
    )

    cells = []
    for feature_edges in ("all", "train_val"):
        for seed in args.seeds:
            cells.append((feature_edges, seed))

    rows = []
    t_start = time.time()
    for i, (feature_edges, seed) in enumerate(cells):
        tag = f"features={feature_edges}  seed={seed}"
        print(f"\n[{i+1:>2d}/{len(cells)}] {tag} ...", flush=True)
        t0 = time.time()
        try:
            r = run_one_mixed(seed=seed, feature_edges=feature_edges, **common)
            r["feature_edges"] = feature_edges
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

    print(f"\nTotal: {time.time()-t_start:.1f}s   results → {out_path}")

    print("\n=== Median across seeds ===")
    print(f"{'features':<12s}  {'AUC_med':>8s}  {'F1m_med':>8s}  {'std':>6s}  {'Δ_AUC':>8s}")
    import statistics
    aucs_all = []
    for feature_edges in ("all", "train_val"):
        cell = [r for r in rows if r["feature_edges"] == feature_edges]
        if not cell:
            continue
        aucs = [r["test_auc"] for r in cell]
        aucs_all.append((feature_edges, aucs))
        f1ms = [r["test_f1_macro"] for r in cell]
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        delta = ""
        if feature_edges == "train_val" and aucs_all:
            base = statistics.median(aucs_all[0][1])
            delta = f"{statistics.median(aucs)-base:+.4f}"
        print(f"{feature_edges:<12s}  {statistics.median(aucs):>8.4f}  "
              f"{statistics.median(f1ms):>8.4f}  {std:>6.4f}  {delta:>8s}")


if __name__ == "__main__":
    main()
