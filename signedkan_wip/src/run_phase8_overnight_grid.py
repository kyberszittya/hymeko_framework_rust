"""Overnight multi-dataset HSiKAN benchmark.

Goals
-----
1. Find balance_lambda sweet spot per dataset (Bitcoin pattern λ=1.0 wins
   doesn't transfer to Slashdot — need per-dataset tuning).
2. Test arity mixing (k=3+k=4 vs k=3+k=4+k=5) per dataset.
3. Compare across real (Bitcoin Alpha/OTC, Slashdot) and synthetic
   (SBM with controllable balance, hierarchical, karate) datasets to
   understand what dataset properties favor HSiKAN.

All cells use:
  - feature_edges="all"      (leaky transductive — matches published)
  - m_e_mode="edge_in_cycle" (the original M_e construction)
  - lr_schedule="cosine"
  - h=16, grid=5, L=2

Resumable: results stream to JSONL; rerun skips completed (dataset, λ,
arities, seed) cells.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from itertools import product
from pathlib import Path

from .run_phase2_mixed_arity import run_one_mixed


# Per-dataset config: (max_per_arity_dict, allowed_arities, cycle_early_stop)
DATASET_CONFIGS = {
    # Real signed networks.
    "bitcoin_alpha":   ({3: 30000, 4: 30000, 5: 30000},
                          [(3, 4), (3, 4, 5)], False),
    "bitcoin_otc":     ({3: 30000, 4: 30000, 5: 30000},
                          [(3, 4), (3, 4, 5)], False),
    # Slashdot: skip k345 because k=5 reservoir is hours per cell.
    "slashdot":        ({3: 30000, 4: 30000},
                          [(3, 4)], False),
    # Synthetic SBM panel (default balance).
    "sbm_n200_k4_s0":  ({3: 10000, 4: 10000, 5: 10000},
                          [(3, 4), (3, 4, 5)], False),
    "sbm_n400_k5_s0":  ({3: 10000, 4: 10000, 5: 10000},
                          [(3, 4), (3, 4, 5)], False),
    # Hierarchical SBM (different balance pattern).
    "hier_n240_s0":    ({3: 10000, 4: 10000, 5: 10000},
                          [(3, 4), (3, 4, 5)], False),
    # SBM positivity sweep (50% pos = symmetric balance).
    "sbmsweep_pos50_s0": ({3: 10000, 4: 10000, 5: 10000},
                            [(3, 4), (3, 4, 5)], False),
    "sbmsweep_pos85_s0": ({3: 10000, 4: 10000, 5: 10000},
                            [(3, 4), (3, 4, 5)], False),
    # Karate (tiny canonical).
    "karate":          ({3: 5000, 4: 5000, 5: 5000},
                          [(3, 4), (3, 4, 5)], False),
}

LAMBDAS = [0.0, 0.1, 1.0]


def _existing_keys(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    keys = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((
            r.get("dataset"),
            r.get("balance_lambda"),
            tuple(r.get("arities", [])),
            r.get("grid"),
            r.get("lr_schedule"),
            r.get("seed"),
        ))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase8_overnight_grid.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--datasets", nargs="+",
                    default=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--lambdas", nargs="+", type=float, default=LAMBDAS)
    ap.add_argument("--grids", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--lr_schedules", nargs="+", default=["cosine", "fixed"])
    ap.add_argument("--cell_timeout_s", type=int, default=2400)  # 40 min
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_keys(out_path)
    print(f"Resuming from {len(done)} completed cells in {out_path}",
          flush=True)

    # Build the full cell list.
    cells = []
    for dataset in args.datasets:
        if dataset not in DATASET_CONFIGS:
            print(f"  WARN unknown dataset {dataset!r}; skipping", flush=True)
            continue
        max_per, arities_list, early_stop = DATASET_CONFIGS[dataset]
        for lam in args.lambdas:
            for arities in arities_list:
                for grid in args.grids:
                    for lrs in args.lr_schedules:
                        for seed in args.seeds:
                            cells.append((
                                dataset, lam, arities, grid, lrs, seed,
                                max_per, early_stop,
                            ))
    print(f"Total cells planned: {len(cells)}", flush=True)

    t_start = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for i, (dataset, lam, arities, grid, lrs, seed,
             max_per, early_stop) in enumerate(cells):
        key = (dataset, lam, tuple(arities), grid, lrs, seed)
        if key in done:
            n_skipped += 1
            continue
        per_k = {k: max_per[k] for k in arities if k in max_per}
        tag = (f"[{i+1:>4d}/{len(cells)}] {dataset:<20s}  "
               f"λ={lam:.2f}  ar={arities!s:<11s}  "
               f"g={grid}  lr={lrs:<6s}  s={seed}")
        print(f"{tag} ...", flush=True, end="")
        t0 = time.time()
        try:
            r = run_one_mixed(
                dataset, seed=seed,
                hidden=16, n_layers=2, grid=grid,
                n_epochs=args.n_epochs,
                arities=arities,
                max_per_arity=per_k,
                coef_smooth_lam=0.0, participation_lam=0.0,
                grad_clip=0.0, weight_decay=0.0,
                early_stopping=False, class_weighted=False,
                lr_schedule=lrs,
                feature_edges="all",
                m_e_mode="edge_in_cycle",
                balance_lambda=lam,
                cycle_early_stop=early_stop,
            )
            r["dataset"] = dataset
            r["balance_lambda"] = lam
            r["arities"] = list(arities)
            r["grid"] = grid
            r["lr_schedule"] = lrs
            with out_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
            n_done += 1
            print(f"  AUC={r['test_auc']:.4f}  "
                  f"alpha={['%.2f' % x for x in r['alpha']]}  "
                  f"{time.time()-t0:.0f}s",
                  flush=True)
        except Exception as e:
            n_failed += 1
            with out_path.open("a") as f:
                f.write(json.dumps(dict(
                    dataset=dataset, balance_lambda=lam,
                    arities=list(arities), grid=grid, lr_schedule=lrs,
                    seed=seed, status="error", error=repr(e),
                )) + "\n")
            print(f"  FAILED: {e!r}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal: {elapsed:.0f}s   "
          f"new={n_done}  resumed_skip={n_skipped}  failed={n_failed}",
          flush=True)
    print(f"Results: {out_path}", flush=True)

    # Per-dataset summary.
    rows = [json.loads(line) for line in out_path.read_text().splitlines()
            if line.strip()]
    rows = [r for r in rows if r.get("test_auc") is not None]
    print("\n=== Best (dataset, λ, arities, grid, lr) by median AUC ===",
          flush=True)
    print(f"{'dataset':<20s}  {'λ':>5s}  {'arities':<11s}  "
          f"{'g':>2s}  {'lr':<7s}  "
          f"{'AUC_med':>8s}  {'F1m_med':>8s}  {'std':>6s}",
          flush=True)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r.get("dataset"), r.get("balance_lambda"),
                tuple(r.get("arities", [])),
                r.get("grid"), r.get("lr_schedule"))
        groups[key].append(r)
    summaries = []
    for key, grp in groups.items():
        aucs = [r["test_auc"] for r in grp]
        f1ms = [r["test_f1_macro"] for r in grp]
        std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        summaries.append((statistics.median(aucs),
                            statistics.median(f1ms), std, key, len(grp)))
    summaries.sort(key=lambda x: (-x[0], x[3]))
    for auc, f1m, std, (ds, lam, ar, gr, lr), n in summaries[:50]:
        print(f"{ds:<20s}  {lam:>5.2f}  {str(ar):<11s}  "
              f"{gr:>2d}  {lr:<7s}  "
              f"{auc:>8.4f}  {f1m:>8.4f}  {std:>6.4f}",
              flush=True)
    # Per-dataset top.
    print(f"\n=== Top per dataset ===", flush=True)
    by_ds = defaultdict(list)
    for s in summaries:
        by_ds[s[3][0]].append(s)
    for ds, lst in by_ds.items():
        best = lst[0]
        print(f"{ds:<20s}  best AUC={best[0]:.4f}  "
              f"(λ={best[3][1]}, ar={best[3][2]}, "
              f"g={best[3][3]}, lr={best[3][4]})", flush=True)


if __name__ == "__main__":
    main()
