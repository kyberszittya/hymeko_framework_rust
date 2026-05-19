"""R2 (participation) and HD (hyperedge density) regulariser sweep.

R2 penalises vertex-degree-weighted node-embedding magnitude.
HD penalises triad-density-weighted triad-embedding magnitude.
Both stay in the entropy-feedback / structural-prior family.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_compare import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/density_sweep.json")
    args = ap.parse_args()

    configs = [
        # (tag, participation_lam, density_lam)
        ("EC+R2(0.05)",        0.05,  0.0),
        ("EC+R2(0.1)",         0.1,   0.0),
        ("EC+R2(0.5)",         0.5,   0.0),
        ("EC+HD(0.001)",       0.0,   0.001),
        ("EC+HD(0.01)",        0.0,   0.01),
        ("EC+HD(0.05)",        0.0,   0.05),
        ("EC+R2+HD(0.1,0.01)", 0.1,   0.01),
    ]

    results = []
    for tag, p_lam, d_lam in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=5,
                             participation_lam=p_lam,
                             density_lam=d_lam)
                r["cfg"] = tag
                print(f"  {tag:24s} {dataset:14s} "
                      f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
