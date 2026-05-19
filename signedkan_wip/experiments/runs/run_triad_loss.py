"""Hypergraph tuple loss sweep.

Layers the Cartwright-Harary balance prior into the loss as a
triad-level margin term:

    L = bce_weight * L_BCE  +  alpha * L_triad

with $L_{triad}(t) = \\mathrm{relu}(margin - \\beta_t \\cdot s(t))$
and $s(t) = \\sum_{(i,j) \\in t} \\sigma_{ij} \\hat h_i \\cdot \\hat h_j$.

Sweep over alpha at the EC recipe (early stop + class-weighted BCE,
$h\\!=\\!32$, $G\\!=\\!5$, $200$ epochs, three seeds). Tests whether
the loss-side structural prior closes part of the residual ~$0.06$
AUC gap to SGCN that survived all earlier (architecture-side and
training-recipe-side) interventions.

Configurations:
  EC                     :  baseline, alpha=0
  EC+T(alpha=0.1)        :  light triad regulariser
  EC+T(alpha=0.5)        :  medium
  EC+T(alpha=1.0)        :  strong
  EC+T(alpha=2.0)        :  very strong (triad dominates)
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
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/triad_loss.json")
    args = ap.parse_args()

    alphas = [0.0, 0.1, 0.5, 1.0, 2.0]

    results = []
    for alpha in alphas:
        tag = "EC" if alpha == 0.0 else f"EC+T(a={alpha})"
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, args.hidden, seed,
                             args.n_epochs, lr=args.lr,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=args.val_every,
                             triad_loss_alpha=alpha,
                             triad_loss_margin=args.margin)
                r["cfg"] = tag
                print(f"  {tag:14s} {dataset:14s} "
                      f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"params={r['n_params']:,}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
