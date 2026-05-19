"""Phase-1 lean-HSiKAN ablation.

Two questions:
  Q1 — "Are the entropy regs (embedding + coef) redundant when
       coef_smooth + weight_decay are present?"
       Compare T6-full (smooth+coef-entropy+embedding-entropy) vs
       T6-lean (smooth-only) at canonical (h=32, G=5, L=3).
  Q2 — "Can HSiKAN be a lot faster at smaller (h, G, L)?"
       Vary (h, G, L) at the lean recipe and measure both metrics
       and wall-clock.

All variants share: wd=1e-4, adam, clip=1.0, R2 sq_max(lam=0.05),
class-weighted BCE, early stopping on val AUC, 120 epochs.

3 seeds × 2 datasets × N configs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .run_compare import run_one


# Per-fixture coef_lam (only used in T6-full control).
COEF_LAM = {"bitcoin_alpha": 0.005, "bitcoin_otc": 0.010}


CONFIGS = [
    # (tag, hidden, grid, n_layers, recipe_kwargs)
    ("T6-full",    32, 5, 3, dict(use_embed_entropy=True,  use_coef_entropy=True)),
    ("T6-lean",    32, 5, 3, dict(use_embed_entropy=False, use_coef_entropy=False)),
    ("h32-G3-L2",  32, 3, 2, dict(use_embed_entropy=False, use_coef_entropy=False)),
    ("h16-G5-L3",  16, 5, 3, dict(use_embed_entropy=False, use_coef_entropy=False)),
    ("h16-G3-L3",  16, 3, 3, dict(use_embed_entropy=False, use_coef_entropy=False)),
    ("h16-G3-L2",  16, 3, 2, dict(use_embed_entropy=False, use_coef_entropy=False)),
]


def _kwargs_for(tag, hidden, grid, n_layers, recipe, dataset):
    base = dict(
        # Architectural pieces (HSiKAN canonical):
        spline_kind="catmull_rom",
        spline_kinds=["catmull_rom"] * n_layers,
        n_layers=n_layers,
        grid=grid,
        pool_mode="sum",
        jk_mode="concat",
        layer_norm_between=True,
        share_weights=True,
        inner_skip="highway",
        outer_skip="none",
        init_scale=0.05,
        # EC backbone:
        early_stopping=True,
        class_weighted=True,
        val_every=5,
        # Tier 1+2 pieces:
        weight_decay=1e-4,
        grad_clip=1.0,
        # R2 (kept; Tier 4 was inconclusive; sq_max default):
        participation_lam=0.05,
        # Tier 6 smoothness — common to all variants:
        coef_smooth_lam=0.010,
    )
    # Embedding-side spectral entropy (T6-full only).
    if recipe["use_embed_entropy"]:
        base.update(dict(
            entropy_lam0=0.01,
            entropy_target=0.5,
            entropy_eta=5.0,
            entropy_kl_normalized=True,
            entropy_momentum=0.9,
        ))
    # Coef-side spectral entropy (T6-full only).
    if recipe["use_coef_entropy"]:
        base["coef_entropy_lam"] = COEF_LAM[dataset]
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase1_lean.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()
    for tag, hidden, grid, n_layers, recipe in CONFIGS:
        for dataset in args.datasets:
            for seed in args.seeds:
                kw = _kwargs_for(tag, hidden, grid, n_layers, recipe, dataset)
                r = run_one(
                    "signedkan", dataset, hidden=hidden, seed=seed,
                    n_epochs=args.n_epochs, lr=5e-2, **kw,
                )
                r["cfg_tag"] = tag
                r["cfg_h"] = hidden
                r["cfg_G"] = grid
                r["cfg_L"] = n_layers
                print(f"  {tag:12s} h={hidden} G={grid} L={n_layers}  "
                      f"{dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"params={r['n_params']:>7,}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    summary = {}
    for tag, hidden, grid, n_layers, _ in CONFIGS:
        for dataset in args.datasets:
            cell = [r for r in runs
                     if r["cfg_tag"] == tag and r["dataset"] == dataset]
            if not cell:
                continue
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            elap = [r["elapsed_s"] for r in cell]
            params = cell[0]["n_params"]
            summary[f"{tag}|{dataset}"] = {
                "auc_med":   round(statistics.median(aucs), 4),
                "f1m_med":   round(statistics.median(f1ms), 4),
                "elapsed_med_s": round(statistics.median(elap), 2),
                "n_params":  params,
                "auc_seeds": [round(a, 4) for a in aucs],
                "f1m_seeds": [round(f, 4) for f in f1ms],
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
