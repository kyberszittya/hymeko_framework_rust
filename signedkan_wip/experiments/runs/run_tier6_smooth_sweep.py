"""Tier 6 (E) — second-difference smoothness on spline coef tensors.

Sweep `coef_smooth_lam ∈ {0.0, 0.001, 0.01, 0.1, 1.0}` at the
per-fixture Tier 3 best (Alpha coef_lam=0.005, OTC coef_lam=0.010);
R2 sq_max, no lam_KL (Tier 5 was null). 5 lams × 2 datasets × 3
seeds = 30 runs.

Hypothesis: penalising ‖Δ²coef‖² along the grid axis discourages
oscillatory splines without zeroing them (orthogonal to L1, which
collapses, and to coef-spectral-entropy, which targets the basis
distribution rather than local control-polygon shape).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from signedkan_wip.src.core.highway_signedkan import HighwaySignedKAN
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
    ap.add_argument("--lams", nargs="+", type=float,
                    default=[0.0, 0.001, 0.01, 0.1, 1.0])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/tier6_smooth_sweep.json")
    args = ap.parse_args()

    base = HighwaySignedKAN.recommended_training_recipe()
    base = {**base,
             "spline_kind": "catmull_rom",
             "weight_decay": 1e-4,
             "grad_clip": 1.0}

    runs = []
    t_total = time.time()
    for lam in args.lams:
        for dataset in args.datasets:
            coef_lam = PER_FIXTURE_COEF[dataset]
            for seed in args.seeds:
                kwargs = {**base,
                           "coef_entropy_lam": coef_lam,
                           "coef_smooth_lam": lam}
                r = run_one(
                    "signedkan", dataset, hidden=32, seed=seed,
                    n_epochs=args.n_epochs, lr=5e-2, **kwargs,
                )
                r["smooth_lam"] = lam
                r["coef_lam"] = coef_lam
                print(f"  smooth_lam={lam:>6.3f}  {dataset:14s} "
                      f"coef_lam={coef_lam:.3f}  seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    summary = {}
    for lam in args.lams:
        for dataset in args.datasets:
            cell = [r for r in runs
                     if abs(r["smooth_lam"] - lam) < 1e-9
                     and r["dataset"] == dataset]
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            summary[f"smooth_lam={lam:.3f}|{dataset}"] = {
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
