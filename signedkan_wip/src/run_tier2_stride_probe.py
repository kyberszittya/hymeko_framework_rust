"""Tier 2 (H) — entropy-schedule stride probe.

stride ∈ {1, 5} at canonical recipe + wd=1e-4 (Tier 1 winner).
2 strides × 2 datasets × 3 seeds = 12 runs.

Hypothesis: stride=5 is essentially behaviour-equivalent to stride=1
because momentum=0.9 already smooths lam_eff across steps; the
schedule update at every step is wasted compute. Reports
metrics + wall-clock.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .highway_signedkan import HighwaySignedKAN
from .run_compare import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--strides", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/tier2_stride_probe.json")
    args = ap.parse_args()

    base = HighwaySignedKAN.recommended_training_recipe()
    base = {**base, "spline_kind": "catmull_rom", "weight_decay": 1e-4}

    runs = []
    t_total = time.time()
    for stride in args.strides:
        for dataset in args.datasets:
            for seed in args.seeds:
                kwargs = {**base, "entropy_stride": stride}
                r = run_one(
                    "signedkan", dataset, hidden=32, seed=seed,
                    n_epochs=args.n_epochs, lr=5e-2, **kwargs,
                )
                r["stride"] = stride
                print(f"  stride={stride}  {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    summary = {}
    for stride in args.strides:
        for dataset in args.datasets:
            cell = [r for r in runs
                     if r["stride"] == stride and r["dataset"] == dataset]
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            elap = [r["elapsed_s"] for r in cell]
            summary[f"stride={stride}|{dataset}"] = {
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
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(runs)} runs in "
          f"{out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
