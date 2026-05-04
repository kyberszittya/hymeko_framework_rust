"""Step-1 of the SGCN-gap-closing plan: validation-based early
stopping. Run the same configurations as the comparison sweep but
with val-AUC checkpointing; report the test AUC of the best-val
checkpoint.

Free-lunch hypothesis: stopping at the validation-AUC peak should
recover ~0.02-0.03 test AUC over the fixed-100-epoch protocol,
because the saturation curve we measured in run_saturation showed
SignedKAN AUC peaks near epoch 50 and decays thereafter.

Run:
  python -m signedkan_wip.src.run_early_stop
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
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/early_stop.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for model in ("signedkan", "vanillakan"):
            for seed in args.seeds:
                r = run_one(model, dataset, args.hidden, seed,
                             args.n_epochs, lr=args.lr,
                             early_stopping=True,
                             val_every=args.val_every)
                print(f"  {model:11s} {dataset:14s} seed={seed}  "
                      f"best_ep={r['best_epoch']:3d}  "
                      f"val_auc={r['best_val_auc']:.4f}  "
                      f"test_auc={r['test_auc']:.4f}  "
                      f"F1_mac={r['test_f1_macro']:.4f}  "
                      f"params={r['n_params']:,}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
