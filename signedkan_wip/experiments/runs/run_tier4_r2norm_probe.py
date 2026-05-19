"""Tier 4 (F) — R2 (participation) degree-weighting normalisation A/B.

Compares `deg_mode = "sq_max"` (original, `deg² / max(deg)²`,
hub-amplifying on power-law graphs) vs `deg_mode = "log"`
(`log(1+deg) / log(1+max(deg))`, heavy-tail-compressed).

Per-fixture-best Tier 3 recipe is used:
  - Alpha: `coef_entropy_lam=0.005`
  - OTC  : `coef_entropy_lam=0.010`
plus `wd=1e-4`, `grad_clip=1.0`, adam.

2 modes × 2 datasets × 3 seeds = 12 runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from signedkan_wip.src.highway_signedkan import HighwaySignedKAN
from .run_compare import run_one


PER_FIXTURE_COEF = {
    "bitcoin_alpha": 0.005,
    "bitcoin_otc":   0.010,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--modes", nargs="+",
                    default=["sq_max", "log"])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/tier4_r2norm.json")
    args = ap.parse_args()

    base = HighwaySignedKAN.recommended_training_recipe()
    base = {**base,
             "spline_kind": "catmull_rom",
             "weight_decay": 1e-4,
             "grad_clip": 1.0}

    runs = []
    t_total = time.time()
    for mode in args.modes:
        for dataset in args.datasets:
            coef_lam = PER_FIXTURE_COEF[dataset]
            for seed in args.seeds:
                kwargs = {**base,
                           "coef_entropy_lam": coef_lam,
                           "participation_deg_mode": mode}
                r = run_one(
                    "signedkan", dataset, hidden=32, seed=seed,
                    n_epochs=args.n_epochs, lr=5e-2, **kwargs,
                )
                r["deg_mode"] = mode
                r["coef_lam"] = coef_lam
                print(f"  deg_mode={mode:6s}  {dataset:14s} "
                      f"coef_lam={coef_lam:.3f}  seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    summary = {}
    for mode in args.modes:
        for dataset in args.datasets:
            cell = [r for r in runs
                     if r["deg_mode"] == mode and r["dataset"] == dataset]
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            summary[f"{mode}|{dataset}"] = {
                "auc_med":   round(statistics.median(aucs), 4),
                "f1m_med":   round(statistics.median(f1ms), 4),
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
