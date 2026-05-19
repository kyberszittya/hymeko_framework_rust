"""Tier 5 (D) — KL-to-uniform target term on the embedding entropy reg.

Adds `lam_eff · lam_KL · KL(p ‖ uniform) / log₂(rank) =
lam_eff · lam_KL · (1 − H_norm)` to the existing reg. Pushes H_norm
UP (spread-spectrum prior) — opposite direction to `lam_b · H_norm`.

Sweep `entropy_lam_kl ∈ {0.0, 0.05, 0.1, 0.3}` at the per-fixture
Tier 3 best (Alpha coef_lam=0.005, OTC coef_lam=0.010); R2 sq_max
(Tier 4 inconclusive). 2 datasets × 4 lams × 3 seeds = 24 runs.
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
    ap.add_argument("--lams", nargs="+", type=float,
                    default=[0.0, 0.05, 0.1, 0.3])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/tier5_kl_sweep.json")
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
                           "entropy_lam_kl": lam}
                r = run_one(
                    "signedkan", dataset, hidden=32, seed=seed,
                    n_epochs=args.n_epochs, lr=5e-2, **kwargs,
                )
                r["lam_kl"] = lam
                r["coef_lam"] = coef_lam
                print(f"  lam_KL={lam:>5.2f}  {dataset:14s} "
                      f"coef_lam={coef_lam:.3f}  seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"H_norm={r.get('last_h_norm', float('nan')):.3f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    summary = {}
    for lam in args.lams:
        for dataset in args.datasets:
            cell = [r for r in runs
                     if abs(r["lam_kl"] - lam) < 1e-9
                     and r["dataset"] == dataset]
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            hns  = [r.get("last_h_norm", float("nan")) for r in cell]
            summary[f"lam_KL={lam:.2f}|{dataset}"] = {
                "auc_med":   round(statistics.median(aucs), 4),
                "f1m_med":   round(statistics.median(f1ms), 4),
                "h_norm_med": round(statistics.median(hns), 3),
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
