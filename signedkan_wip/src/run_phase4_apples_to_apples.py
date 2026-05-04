"""Phase-4 — final apples-to-apples comparison.

Variants exercised (each x 2 datasets x 3 seeds = 6 cells):

  HSiKAN-mixed @ h=16, G=3, L=2, k=(3,4) lean recipe, varied along
  two axes:
    A. EC-on vs EC-off (ec backbone = early stopping + class-weighted
       BCE + weight decay).
    B. lean vs leanest (leanest = drop participation_lam, grad_clip,
       and coef_smooth_lam — all flagged redundant/harmful in Phase 3).

  HSiKAN variants in this phase:
    1. mixed_lean_ec        : Phase-2 baseline (smooth+R2+wd+clip+EC)
    2. mixed_leanest_ec     : drop R2/clip/smooth, keep EC
    3. mixed_lean_strict    : Phase-2 lean, but strict-Derr (no EC)
    4. mixed_leanest_strict : leanest + strict-Derr

  SGCN+balance variants (already in Phase 3, restated for the table):
    5. sgcn_full_ec         : with EC backbone
    6. sgcn_strict_derr     : noEC (ie Derr-faithful)

  We do NOT permanently delete any HSiKAN code paths — every "drop"
  is via kwargs only.

The strict-Derr-vs-strict-Derr cell (4 vs 6) is the headline
apples-to-apples claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .run_phase2_mixed_arity import run_one_mixed
from .run_sgcn_baseline import run_one_sgcn


HSIKAN_VARIANTS = {
    "mixed_lean_ec":        dict(coef_smooth_lam=0.010,
                                  participation_lam=0.05,
                                  grad_clip=1.0,
                                  weight_decay=1e-4,
                                  early_stopping=True,
                                  class_weighted=True),
    "mixed_leanest_ec":     dict(coef_smooth_lam=0.0,
                                  participation_lam=0.0,
                                  grad_clip=0.0,
                                  weight_decay=1e-4,
                                  early_stopping=True,
                                  class_weighted=True),
    "mixed_lean_strict":    dict(coef_smooth_lam=0.010,
                                  participation_lam=0.05,
                                  grad_clip=1.0,
                                  weight_decay=0.0,
                                  early_stopping=False,
                                  class_weighted=False),
    "mixed_leanest_strict": dict(coef_smooth_lam=0.0,
                                  participation_lam=0.0,
                                  grad_clip=0.0,
                                  weight_decay=0.0,
                                  early_stopping=False,
                                  class_weighted=False),
}

SGCN_VARIANTS = {
    "sgcn_full_ec": dict(early_stopping=True,
                          class_weighted=True,
                          weight_decay=1e-4),
    "sgcn_strict_derr": dict(early_stopping=False,
                              class_weighted=False,
                              weight_decay=0.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--max_k4", type=int, default=30000)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase4_apples_to_apples.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()

    # ── HSiKAN-mixed variants ────────────────────────────────────────
    for variant, ovr in HSIKAN_VARIANTS.items():
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one_mixed(
                    dataset, seed,
                    hidden=16, n_layers=2, grid=3,
                    n_epochs=args.n_epochs,
                    max_k4=args.max_k4,
                    only_k3=False,
                    arities=(3, 4),
                    **ovr,
                )
                r["arch"] = "hsikan_mixed_k34"
                r["variant"] = variant
                print(f"  {variant:24s} {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"alpha={[round(a,3) for a in r['alpha']]}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    # ── SGCN+balance variants ────────────────────────────────────────
    for variant, ovr in SGCN_VARIANTS.items():
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one_sgcn(
                    dataset, seed, hidden=32, n_layers=2,
                    n_epochs=args.n_epochs, lr=5e-3,
                    balance_alpha=0.5,
                    adj_protocol="full_graph",
                    **ovr,
                )
                r["arch"] = "sgcn_balance"
                r["variant"] = variant
                print(f"  {variant:24s} {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    # Summary table.
    summary = {}
    keys = sorted({(r["arch"], r["variant"], r["dataset"]) for r in runs})
    for arch, variant, dataset in keys:
        cell = [r for r in runs
                 if r["arch"] == arch and r["variant"] == variant
                 and r["dataset"] == dataset]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        elap = [r["elapsed_s"] for r in cell]
        summary[f"{arch}|{variant}|{dataset}"] = {
            "auc_med":   round(statistics.median(aucs), 4),
            "f1m_med":   round(statistics.median(f1ms), 4),
            "elapsed_med_s": round(statistics.median(elap), 2),
            "auc_seeds": [round(a, 4) for a in aucs],
            "f1m_seeds": [round(f, 4) for f in f1ms],
            "n_seeds":   len(cell),
        }

    out = {
        "runs": runs,
        "summary": summary,
        "wall_clock_s": round(time.time() - t_total, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(runs)} runs in "
          f"{out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
