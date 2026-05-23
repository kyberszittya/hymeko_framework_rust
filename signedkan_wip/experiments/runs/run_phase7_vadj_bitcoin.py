"""Bitcoin Alpha — honest no-leak vertex-adjacency arity sweep.

Combines: m_e_mode='vertex_adjacency' (no σ-as-label leak, generalises
the k=2 line-graph adjacency to k≥3) + feature_edges='train_val' (test
edges removed from cycle structure). This is the structurally clean
evaluation protocol.

Configs: single-arity k=3, k=4, k=5; mixed (3,4), (3,4,5), (4,5).
3 seeds, h=16, grid=5, n_epochs=120, max_per_arity=30k.

Target to beat: SGCN ~0.91 on Bitcoin Alpha.
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
                    "signedkan_wip/experiments/results/phase7_vadj_bitcoin.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--max_per", type=int, default=30000)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    configs = [
        ("k3",      (3,)),
        ("k4",      (4,)),
        ("k5",      (5,)),
        ("k34",     (3, 4)),
        ("k45",     (4, 5)),
        ("k345",    (3, 4, 5)),
    ]

    rows = []
    t_start = time.time()
    for cell_name, arities in configs:
        for seed in args.seeds:
            tag = f"{cell_name:<6s} seed={seed}"
            t0 = time.time()
            print(f"[{tag}] ...", flush=True, end="")
            try:
                r = run_one_mixed(
                    "bitcoin_alpha", seed=seed,
                    hidden=args.hidden, n_layers=2, grid=args.grid,
                    n_epochs=args.n_epochs,
                    arities=arities,
                    max_per_arity={k: args.max_per for k in arities},
                    coef_smooth_lam=0.0, participation_lam=0.0,
                    grad_clip=0.0, weight_decay=0.0,
                    early_stopping=False, class_weighted=False,
                    lr_schedule="cosine",
                    feature_edges="train_val",
                    m_e_mode="vertex_adjacency",
                )
                r["cell"] = cell_name
                rows.append(r)
                with out_path.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"  AUC={r['test_auc']:.4f}  "
                      f"alpha={['%.2f' % x for x in r['alpha']]}  "
                      f"{time.time()-t0:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"  FAILED: {e!r}", flush=True)

    print(f"\nTotal: {time.time()-t_start:.1f}s   results → {out_path}")
    print("\n=== Median across seeds ===")
    print(f"{'cell':<8s}  {'AUC_med':>8s}  {'std':>6s}  {'alpha':>20s}",
          flush=True)
    for cell_name, _ in configs:
        cell_rows = [r for r in rows if r.get("cell") == cell_name]
        if not cell_rows:
            continue
        aucs = [r["test_auc"] for r in cell_rows]
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        n_arities = len(cell_rows[0]["alpha"])
        amed = [round(statistics.median([r["alpha"][i] for r in cell_rows]), 2)
                for i in range(n_arities)]
        print(f"{cell_name:<8s}  {statistics.median(aucs):>8.4f}  "
              f"{std:>6.4f}  {str(amed):>20s}",
              flush=True)


if __name__ == "__main__":
    main()
