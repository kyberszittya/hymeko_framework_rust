"""Phase-3 — single-piece-removal ablation on the lean recipe.

Lean baseline (Phase 1 winner): h=16, G=3, L=2, smooth=0.010,
participation_lam=0.05, wd=1e-4, clip=1.0, adam, EC backbone,
highway gates, layer-norm between, JK-concat, share_weights,
signed branches (S=2).

This phase flips ONE piece off at a time vs the lean baseline.
If the metric stays within seed-noise of baseline, that piece is
redundant and can be dropped — yielding a smaller, faster, less-
mysterious deployment recipe.

Also runs SGCN at "no-EC" (no early-stop, no class-weighted BCE,
no weight decay) on the same splits to see whether our SGCN
over-reproduction (0.96 vs published 0.93) is the EC backbone.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .run_compare import run_one
from .run_sgcn_baseline import run_one_sgcn


def _hsikan_lean_kwargs(**overrides):
    base = dict(
        spline_kind="catmull_rom",
        spline_kinds=["catmull_rom"] * 2,
        n_layers=2,
        grid=3,
        pool_mode="sum",
        jk_mode="concat",
        layer_norm_between=True,
        share_weights=True,
        inner_skip="highway",
        outer_skip="none",
        init_scale=0.05,
        early_stopping=True,
        class_weighted=True,
        val_every=5,
        weight_decay=1e-4,
        grad_clip=1.0,
        participation_lam=0.05,
        coef_smooth_lam=0.010,
    )
    base.update(overrides)
    return base


HSIKAN_VARIANTS = [
    # (tag, overrides — flips one piece off the lean baseline)
    ("baseline",     {}),
    ("-R2",          dict(participation_lam=0.0)),
    ("-wd",          dict(weight_decay=0.0)),
    ("-smooth",      dict(coef_smooth_lam=0.0)),
    ("-clip",        dict(grad_clip=0.0)),
    ("-highway",     dict(inner_skip="none")),
    ("-LN",          dict(layer_norm_between=False)),
    ("-share",       dict(share_weights=False)),
    ("-signed",      dict(use_minus_branch=False)),
    ("-jk",          dict(jk_mode="last")),
    ("-cw_BCE",      dict(class_weighted=False)),
    ("-early_stop",  dict(early_stopping=False)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase3_redundancy.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()

    # ── HSiKAN ablations ───────────────────────────────────────────
    for tag, ovr in HSIKAN_VARIANTS:
        for dataset in args.datasets:
            for seed in args.seeds:
                kw = _hsikan_lean_kwargs(**ovr)
                r = run_one("signedkan", dataset, hidden=16, seed=seed,
                             n_epochs=args.n_epochs, lr=5e-2, **kw)
                r["arch"] = "hsikan_lean"
                r["variant"] = tag
                print(f"  {tag:14s} {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"params={r['n_params']:>6,}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    # ── SGCN protocol-difference probe ─────────────────────────────
    # full_graph adjacency throughout (matches HSiKAN's protocol).
    # Vary the EC backbone to measure how much it contributes to
    # our 0.96 vs published 0.93.
    sgcn_variants = [
        ("ec_full_graph",       dict(early_stopping=True,  class_weighted=True,
                                      weight_decay=1e-4)),
        ("noEC_full_graph",     dict(early_stopping=False, class_weighted=False,
                                      weight_decay=0.0)),
        ("noCW_full_graph",     dict(early_stopping=True,  class_weighted=False,
                                      weight_decay=1e-4)),
        ("noWD_full_graph",     dict(early_stopping=True,  class_weighted=True,
                                      weight_decay=0.0)),
    ]
    for tag, ovr in sgcn_variants:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one_sgcn(dataset, seed, hidden=32, n_layers=2,
                                  n_epochs=args.n_epochs, lr=5e-3,
                                  balance_alpha=0.5,
                                  adj_protocol="full_graph", **ovr)
                r["arch"] = "sgcn_balance"
                r["variant"] = tag
                print(f"  sgcn {tag:18s} {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    # Summary
    summary = {}
    keys_seen = set()
    for r in runs:
        keys_seen.add((r["arch"], r["variant"], r["dataset"]))
    for arch, variant, dataset in sorted(keys_seen):
        cell = [r for r in runs
                 if r["arch"] == arch and r["variant"] == variant
                 and r["dataset"] == dataset]
        aucs = [r["test_auc"] for r in cell]
        f1ms = [r["test_f1_macro"] for r in cell]
        summary[f"{arch}|{variant}|{dataset}"] = {
            "auc_med":   round(statistics.median(aucs), 4),
            "f1m_med":   round(statistics.median(f1ms), 4),
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
