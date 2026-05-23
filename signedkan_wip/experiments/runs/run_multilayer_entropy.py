"""Compose multi-layer (L2-jkCAT-sum-BS-BS) with the spectral-entropy
regulariser on the EC recipe.

Tests whether the three macro-F1 levers (class-weighted BCE,
multi-layer JK-concat, spectral-entropy schedule) compose. Runs the
best multi-layer recipe with the EC+H regulariser settings
($\\lambda_0\\!=\\!10^{-2}$, $H^*\\!=\\!0.5$, $\\eta\\!=\\!5$).
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
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--entropy-lam0", type=float, default=1e-2)
    ap.add_argument("--target-entropy", type=float, default=0.5)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/multilayer_entropy.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_one("signedkan", dataset, args.hidden, seed,
                         args.n_epochs, lr=args.lr,
                         early_stopping=True,
                         class_weighted=True,
                         grid=5,
                         val_every=args.val_every,
                         n_layers=2,
                         spline_kinds=["bspline", "bspline"],
                         pool_mode="sum",
                         jk_mode="concat",
                         entropy_lam0=args.entropy_lam0,
                         entropy_target=args.target_entropy)
            r["cfg"] = "L2-jkCAT-sum+H"
            print(f"  L2-jkCAT-sum+H {dataset:14s} "
                  f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                  f"AUC={r['test_auc']:.4f}  "
                  f"F1m={r['test_f1_macro']:.4f}  "
                  f"H={r['last_h_norm']:.3f}  "
                  f"params={r['n_params']:,}  "
                  f"{r['elapsed_s']:.1f}s")
            results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
