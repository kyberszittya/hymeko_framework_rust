"""Run the entropy-on-EC sweep: layer the Lyapunov-safe spectral
entropy regulariser on top of the EC recipe (early stopping +
class-weighted BCE, G=5).

Tests whether the entropy regulariser composes additively with the
class-balanced loss. Two mechanisms target orthogonal failure modes:
class-weighted BCE addresses imbalance under-training; the entropy
schedule addresses post-saturation overfitting (cf. Fig. saturation).

Run:
  python -m signedkan_wip.src.run_entropy_on_ec
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
                    "signedkan_wip/experiments/results/entropy_on_ec.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for model in ("signedkan", "vanillakan"):
            for seed in args.seeds:
                r = run_one(model, dataset, args.hidden, seed,
                             args.n_epochs, lr=args.lr,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=args.val_every,
                             entropy_lam0=args.entropy_lam0,
                             entropy_target=args.target_entropy)
                r["cfg"] = "EC+entropy"
                print(f"  EC+H  {model:11s} {dataset:14s} "
                      f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1_mac={r['test_f1_macro']:.4f}  "
                      f"H={r['last_h_norm']:.3f}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
