"""Map the SignedKAN training-saturation curve: how does test AUC /
macro-F1 evolve with the epoch budget? Tests whether the gap to
SGCN's published 0.93 AUC is closeable by simply training longer.

Sweep: epochs ∈ {50, 100, 200, 300} × {SignedKAN, Vanilla KAN}
       × seeds {0, 1, 2} × datasets {bitcoin_alpha, bitcoin_otc}.
Hidden 32, lr 5e-2, no entropy reg.

Result: a JSON catalogue + a saturation-curve plot.
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
    ap.add_argument("--epoch-grid", nargs="+", type=int,
                    default=[50, 100, 200, 300])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--out",
                    default="signedkan_wip/experiments/results/saturation.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for n_epochs in args.epoch_grid:
            for model_name in ("signedkan", "vanillakan"):
                for seed in args.seeds:
                    r = run_one(model_name, dataset, args.hidden,
                                 seed, n_epochs, lr=args.lr)
                    print(f"  {model_name:11s} {dataset:14s} "
                          f"epochs={n_epochs:3d}  seed={seed}  "
                          f"AUC={r['test_auc']:.4f}  "
                          f"F1_mac={r['test_f1_macro']:.4f}  "
                          f"{r['elapsed_s']:.1f}s")
                    results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
