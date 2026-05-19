"""Signed-triad attention sweep, EC recipe.

Replaces the mean-pool $\\mathbf{h}_e = \\frac{1}{|T_e|}\\sum_t \\mathbf{h}_t$
with $\\mathbf{h}_e = \\sum_t \\tanh(\\mathbf{a}^\\top [h_u; h_v; h_t])\\,\\mathbf{h}_t$.
Adds the per-edge entropy regulariser
$L_{ae} = -\\lambda_{ae}\\mathrm{mean}_e H(|\\boldsymbol{\\alpha}_e|/\\|\\boldsymbol{\\alpha}_e\\|_1)$
to prevent attention collapse onto a single triad.

Configurations:
  EC+attn(lam_ae=0)     : signed-triad attention, no entropy term
  EC+attn(lam_ae=0.01)  : with mild attention-entropy term
  EC+attn(lam_ae=0.05)  : with stronger attention-entropy term
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
                    "signedkan_wip/experiments/results/attention_ec.json")
    args = ap.parse_args()

    configs = [
        ("EC+attn",            0.0),
        ("EC+attn+ae(0.01)",   0.01),
        ("EC+attn+ae(0.05)",   0.05),
    ]

    results = []
    for tag, lam_ae in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2,
                             early_stopping=True,
                             class_weighted=True,
                             grid=5,
                             val_every=5,
                             use_attention=True,
                             attention_entropy_lam=lam_ae)
                r["cfg"] = tag
                print(f"  {tag:22s} {dataset:14s} "
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
