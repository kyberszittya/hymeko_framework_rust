"""Phase 9 — k=3+4+5 mixed-arity sweep on all fixtures.

Closes the n-tuples paper story: mixed k=3+k=4+k=5 with learned αₖ.

The architectural claim is that higher-arity cycles carry
*complementary* signed-balance information; the αₖ-mixture should be
able to selectively emphasise whichever arity carries the strongest
signal on each fixture. Phase 6 established that k=3+k=4 wins by
+0.15-0.30 AUC on SBM/hier-SBM. This phase asks: does adding k=5
help, hurt, or stay neutral?

Datasets:
  - karate, sbm_n200_k4_s0, sbm_n400_k5_s0, hier_n240_s0
    (the phase-6 panel — small + balanced regime)
  - bitcoin_alpha, bitcoin_otc
    (the imbalanced trust regime — for completeness)

Recipes:
  - hsikan_k34   — current best mixed (3, 4)        — phase 2/6 baseline
  - hsikan_k345  — extended mixed (3, 4, 5)         — phase 9 contribution
  - hsikan_k45   — drop k=3                         — ablation
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .run_phase2_mixed_arity import run_one_mixed


DATASETS = [
    "karate",
    "sbm_n200_k4_s0",
    "sbm_n400_k5_s0",
    "hier_n240_s0",
    "bitcoin_alpha",
    "bitcoin_otc",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--max_per_arity", type=int, default=30000,
                    help="Per-arity subsample cap for k=3, k=4, k=5.")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase9_k345_mixed.json")
    args = ap.parse_args()

    cap = {3: args.max_per_arity, 4: args.max_per_arity, 5: args.max_per_arity}
    runs = []
    t_total = time.time()

    for dataset in args.datasets:
        for seed in args.seeds:
            for arities, label in [
                ((3, 4),    "hsikan_k34"),
                ((3, 4, 5), "hsikan_k345"),
                ((4, 5),    "hsikan_k45"),
            ]:
                try:
                    r = run_one_mixed(
                        dataset, seed, hidden=16, n_layers=2, grid=3,
                        n_epochs=args.n_epochs,
                        arities=arities,
                        max_per_arity=cap,
                        only_k3=False,
                        coef_smooth_lam=0.0, participation_lam=0.0,
                        grad_clip=0.0, weight_decay=0.0,
                        early_stopping=False, class_weighted=False,
                    )
                    r["arch"] = label
                    r["arities"] = list(arities)
                    print(f"  {label:14s} {dataset:18s} seed={seed}  "
                          f"AUC={r['test_auc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
                          f"alpha={[round(a,3) for a in r['alpha']]}  "
                          f"{r['elapsed_s']:.1f}s")
                    runs.append(r)
                except Exception as e:
                    print(f"  {label:14s} FAILED on {dataset} seed={seed}: {e!r}")

    summary = {}
    keys = sorted({(r["arch"], r["dataset"]) for r in runs})
    for arch, dataset in keys:
        cell = [r for r in runs if r["arch"] == arch and r["dataset"] == dataset]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        elap = [r["elapsed_s"] for r in cell]
        alphas_per_seed = [r.get("alpha", []) for r in cell]
        summary[f"{arch}|{dataset}"] = {
            "auc_med":   round(statistics.median(aucs), 4),
            "f1m_med":   round(statistics.median(f1ms), 4),
            "auc_mean":  round(float(np.mean(aucs)), 4),
            "auc_std":   round(float(np.std(aucs)), 4),
            "f1m_mean":  round(float(np.mean(f1ms)), 4),
            "elapsed_med_s": round(statistics.median(elap), 2),
            "n_seeds":   len(cell),
            "auc_seeds": [round(a, 4) for a in aucs],
            "f1m_seeds": [round(f, 4) for f in f1ms],
            "alpha_seeds": [[round(a, 3) for a in alpha] for alpha in alphas_per_seed],
        }

    out = {
        "runs": runs,
        "summary": summary,
        "wall_clock_s": round(time.time() - t_total, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(runs)} runs in {out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
