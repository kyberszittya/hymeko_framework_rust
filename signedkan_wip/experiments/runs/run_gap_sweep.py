"""Run the gap-closing sweep: each step in isolation and combined.
Builds the §IV.8 table for the SignedKAN paper.

Configurations:
  W (WiP baseline):  early_stopping=False, class_weighted=False, grid=5
  E (Step 1 only):   early_stopping=True,  class_weighted=False, grid=5
  EC (Steps 1+2):    early_stopping=True,  class_weighted=True,  grid=5
  ECG (Steps 1+2+3): early_stopping=True,  class_weighted=True,  grid=3

W and E are already in entropy_sweep.json / early_stop.json. This
runner produces EC and ECG. Joint table is built by reading all
three result files.

Run:
  python -m signedkan_wip.experiments.runs.run_gap_sweep
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
                    "signedkan_wip/experiments/results/gap_sweep.json")
    args = ap.parse_args()

    results = []
    cfgs = [
        ("EC",  dict(early_stopping=True,  class_weighted=True,  grid=5)),
        ("ECG", dict(early_stopping=True,  class_weighted=True,  grid=3)),
    ]
    for cfg_label, cfg_kwargs in cfgs:
        for dataset in args.datasets:
            for model in ("signedkan", "vanillakan"):
                for seed in args.seeds:
                    r = run_one(model, dataset, args.hidden, seed,
                                 args.n_epochs, lr=args.lr,
                                 val_every=args.val_every, **cfg_kwargs)
                    r["cfg"] = cfg_label
                    print(f"  {cfg_label:3s}  {model:11s} {dataset:14s} "
                          f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                          f"AUC={r['test_auc']:.4f}  "
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
