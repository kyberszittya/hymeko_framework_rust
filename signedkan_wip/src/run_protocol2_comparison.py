"""Protocol 2 comparison — each architecture with its native auxiliary
balance loss.

Pairings:
  - SGCN + Derr 2018 §3.3 extended structural balance loss.
  - HSiKAN T6 (deployment recipe) + arity-agnostic NTupleBalanceLoss
    (Davis weak balance over signed n-tuple cycle edges; reduces to
    TriadLoss-equivalent at k=3).

Both at the same train/val/test splits, EC backbone (early stopping
+ class-weighted BCE + weight decay), 3 seeds × 2 datasets.

Auxiliary alpha = 0.5 for both (single point; sweep can come later).

Why this is fairer than `run_sgcn_baseline` alone: the bare-BCE
SGCN underperforms its published 0.93 by ~0.03 because Derr's full
loss includes the structural balance term. Adding L_balance to SGCN
without an architecture-aware analog on HSiKAN biases the
comparison; with NTupleBalanceLoss on HSiKAN, both architectures
encode Cartwright-Harary balance theory in their loss as their
authors intended.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .highway_signedkan import HighwaySignedKAN
from .run_compare import run_one
from .run_sgcn_baseline import run_one_sgcn


PER_FIXTURE_HSIKAN = {
    "bitcoin_alpha": dict(coef_entropy_lam=0.005,
                            coef_smooth_lam=0.010),
    "bitcoin_otc":   dict(coef_entropy_lam=0.010,
                            coef_smooth_lam=0.010),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Auxiliary loss weight (both architectures)")
    ap.add_argument("--balance_margin", type=float, default=1.0,
                    help="SGCN's L_balance margin")
    ap.add_argument("--ntuple_margin", type=float, default=0.5,
                    help="HSiKAN's NTupleBalanceLoss margin")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/protocol2_comparison.json")
    args = ap.parse_args()

    base_hsikan = HighwaySignedKAN.recommended_training_recipe()
    base_hsikan = {**base_hsikan,
                    "spline_kind": "catmull_rom",
                    "weight_decay": 1e-4,
                    "grad_clip": 1.0}

    runs = []
    t_total = time.time()

    # ── HSiKAN T6 + NTupleBalanceLoss ────────────────────────────────
    for dataset in args.datasets:
        per_fixture = PER_FIXTURE_HSIKAN[dataset]
        for seed in args.seeds:
            kwargs = {**base_hsikan,
                       **per_fixture,
                       "ntuple_balance_alpha": args.alpha,
                       "ntuple_balance_margin": args.ntuple_margin}
            r = run_one(
                "signedkan", dataset, hidden=32, seed=seed,
                n_epochs=args.n_epochs, lr=5e-2, **kwargs,
            )
            r["arch"] = "hsikan_t6_balance"
            r["aux_alpha"] = args.alpha
            print(f"  hsikan_t6_balance  {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  "
                  f"F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

    # ── SGCN + extended balance loss ────────────────────────────────
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one_sgcn(
                dataset, seed, hidden=32, n_layers=2,
                n_epochs=args.n_epochs, lr=5e-3, weight_decay=1e-4,
                balance_alpha=args.alpha,
                balance_margin=args.balance_margin,
            )
            r["arch"] = "sgcn_balance"
            r["aux_alpha"] = args.alpha
            print(f"  sgcn_balance       {dataset:14s} seed={seed}  "
                  f"AUC={r['test_auc']:.4f}  "
                  f"F1m={r['test_f1_macro']:.4f}  "
                  f"{r['elapsed_s']:.1f}s")
            runs.append(r)

    # Summary
    summary = {}
    for arch in ("hsikan_t6_balance", "sgcn_balance"):
        for dataset in args.datasets:
            cell = [r for r in runs if r["arch"] == arch
                     and r["dataset"] == dataset]
            if not cell:
                continue
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            elap = [r["elapsed_s"] for r in cell]
            summary[f"{arch}|{dataset}"] = {
                "auc_med":   round(statistics.median(aucs), 4),
                "f1m_med":   round(statistics.median(f1ms), 4),
                "elapsed_med_s": round(statistics.median(elap), 2),
                "auc_seeds": [round(a, 4) for a in aucs],
                "f1m_seeds": [round(f, 4) for f in f1ms],
            }

    out = {
        "runs": runs,
        "summary": summary,
        "wall_clock_s": round(time.time() - t_total, 1),
        "alpha": args.alpha,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(runs)} runs in "
          f"{out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
