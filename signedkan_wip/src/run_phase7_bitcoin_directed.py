"""Bitcoin Alpha — directed × leak comparison.

Tests two questions in one run:
  (a) Does the directed-vs-undirected gap survive at full training?
  (b) How much of the gap is leak amplification (cycles built from the
      full graph encode test-edge signs in their σ patterns)?

Grid: {undirected, directed} × {features=all, features=train_val} × 3 seeds
     = 12 cells. h=16 / max_k4=30k / L=2 / 60 epochs. Bitcoin Alpha is
     small enough that GPU runs each cell in ~30-60s.

Reading
-------
- features="all": cycles from g.edges (transductive, leaky baseline)
- features="train_val": cycles from train+val edges only — test edges
  are held out of the cycle structure used to predict them.

If directed wins under "all" but ties under "train_val", the gap was
leak-amplified. If directed wins under both, the gap is real signal.
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
                    "signedkan_wip/experiments/results/phase7_bitcoin_directed.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=60)
    ap.add_argument("--max_k4", type=int, default=30_000)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common = dict(
        dataset="bitcoin_alpha",
        hidden=16, n_layers=2, grid=3,
        n_epochs=args.n_epochs,
        arities=(3, 4),
        max_k3=0,
        max_k4=args.max_k4,
        coef_smooth_lam=0.0, participation_lam=0.0,
        grad_clip=0.0, weight_decay=0.0,
        early_stopping=False, class_weighted=False,
    )

    cells = []
    for directed in (False, True):
        for feature_edges in ("all", "train_val"):
            for seed in args.seeds:
                cells.append((directed, feature_edges, seed))

    rows = []
    t_start = time.time()
    for i, (directed, feature_edges, seed) in enumerate(cells):
        tag = (f"directed={directed}  features={feature_edges}  "
               f"seed={seed}")
        print(f"\n[{i+1:>2d}/{len(cells)}] {tag} ...", flush=True)
        t0 = time.time()
        try:
            r = run_one_mixed(seed=seed, directed=directed,
                                feature_edges=feature_edges, **common)
            r["directed"] = directed
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
    print(f"{'directed':<9s}  {'features':<12s}  {'AUC_med':>8s}  "
          f"{'F1m_med':>8s}  {'std':>6s}")
    import statistics
    for directed in (False, True):
        for feature_edges in ("all", "train_val"):
            cell = [r for r in rows if r["directed"] == directed
                     and r["feature_edges"] == feature_edges]
            if not cell:
                continue
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
            print(f"{str(directed):<9s}  {feature_edges:<12s}  "
                  f"{statistics.median(aucs):>8.4f}  "
                  f"{statistics.median(f1ms):>8.4f}  {std:>6.4f}")


if __name__ == "__main__":
    main()
