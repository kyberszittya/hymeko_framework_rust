"""R2 composition sweep — does R2 (participation regulariser at λ=0.05)
stack with the other established interventions?

R2 alone gave the cleanest simultaneous AUC+F1m win on Alpha
(+0.005 AUC, +0.008 F1m). Test five natural compositions:

  1. R2 + entropy reg
  2. R2 + triad loss
  3. R2 + heterogeneous skip (residual on inner, none on outer)
  4. R2 + L=3 LayerNorm + weight-sharing (multi-layer trick)
  5. R2 + bilinear endpoint head
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
                    "signedkan_wip/experiments/results/r2_compositions.json")
    args = ap.parse_args()

    base = dict(
        early_stopping=True,
        class_weighted=True,
        grid=5,
        val_every=5,
        participation_lam=0.05,
    )

    configs = [
        # (tag, kwargs override)
        ("EC+R2",                base),
        ("EC+R2+H",              {**base, "entropy_lam0": 0.01, "entropy_target": 0.5}),
        ("EC+R2+T(0.5)",         {**base, "triad_loss_alpha": 0.5}),
        ("EC+R2+skip(R,N)",      {**base, "inner_skip": "residual",
                                  "outer_skip": "none"}),
        ("EC+R2+bilinear-full",  {**base, "use_bilinear": True,
                                  "bilinear_rank": 0}),
        ("L=3+LN+share+R2",      {**base, "n_layers": 3,
                                  "spline_kinds": ["bspline"]*3,
                                  "pool_mode": "sum",
                                  "jk_mode": "concat",
                                  "layer_norm_between": True,
                                  "share_weights": True}),
    ]

    results = []
    for tag, kwargs in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2, **kwargs)
                r["cfg"] = tag
                print(f"  {tag:24s} {dataset:14s} "
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
