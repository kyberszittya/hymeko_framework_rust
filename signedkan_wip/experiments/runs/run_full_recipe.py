"""Full deployment-recipe sweep.

Headline: grid pruning (G=3) + refined entropy (kl_normalized + momentum)
+ EC + R2 + multi-layer (L=3, LN, share, JK-concat, sum-pool) +
heterogeneous skip placement (inner=highway, outer=none).

Highway on the inner spline gives an interpretable parameter-usage
signal: when the gate $T(x) = \\sigma(W_T x + b_T)$ averaged over
training inputs is small, the layer's spline is mostly bypassed and
the parameters are effectively pruneable post-training.

Configurations tested:
  full-recipe-G3-Hwy : the headline; grid=3 + highway-skip
  full-recipe-G3-Res : same but residual skip (control: highway vs residual)
  full-recipe-G5-Hwy : same as headline but grid=5 (control: grid pruning)
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
                    "signedkan_wip/experiments/results/full_recipe.json")
    args = ap.parse_args()

    base = dict(
        early_stopping=True,
        class_weighted=True,
        val_every=5,
        # R2.
        participation_lam=0.05,
        # Multi-layer with the focused-stack stabilisation.
        n_layers=3,
        spline_kinds=["bspline"] * 3,
        pool_mode="sum",
        jk_mode="concat",
        layer_norm_between=True,
        share_weights=True,
        # Outer is unskipped per the heterogeneous-skip finding.
        outer_skip="none",
        # Refined entropy regulariser (kl_normalized + momentum).
        entropy_lam0=0.01,
        entropy_target=0.5,
        entropy_eta=5.0,
        entropy_kl_normalized=True,
        entropy_momentum=0.9,
    )

    configs = [
        ("full-recipe-G3-Hwy", {**base, "grid": 3, "inner_skip": "highway"}),
        ("full-recipe-G3-Res", {**base, "grid": 3, "inner_skip": "residual"}),
        ("full-recipe-G5-Hwy", {**base, "grid": 5, "inner_skip": "highway"}),
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
