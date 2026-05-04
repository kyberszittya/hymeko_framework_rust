"""Cross-branch information regulariser sweep, EC + entropy reg recipe.

Adds the cross-branch coefficient cosine penalty
$L_\\mathrm{cross} = \\lambda_\\mathrm{mi} \\cdot \\overline{|\\cos|}$
on top of the existing entropy regulariser (which already carries
KL feedback). Tests whether forcing the per-sign-branch splines to
encode distinct functions buys additional accuracy.

Configurations:
  EC                            : baseline (no entropy, no cross-branch)
  EC+H                          : entropy reg only (lam_0=0.01, H*=0.5)
  EC+H+cross(lam=0.01)          : both
  EC+H+cross(lam=0.05)          : both, stronger cross-branch
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
                    "signedkan_wip/experiments/results/cross_branch_ec.json")
    args = ap.parse_args()

    configs = [
        ("EC+H+cross(0.005)", 1e-2, 0.005),
        ("EC+H+cross(0.01)",  1e-2, 0.01),
        ("EC+H+cross(0.05)",  1e-2, 0.05),
        ("EC+cross(0.01)",    0.0,  0.01),  # cross alone (no entropy)
    ]

    results = []
    for tag, ent_lam, cross_lam in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                kwargs = dict(
                    early_stopping=True,
                    class_weighted=True,
                    grid=5,
                    val_every=5,
                    cross_branch_lam=cross_lam,
                )
                if ent_lam > 0:
                    kwargs["entropy_lam0"] = ent_lam
                    kwargs["entropy_target"] = 0.5
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2, **kwargs)
                r["cfg"] = tag
                print(f"  {tag:24s} {dataset:14s} "
                      f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"H={r['last_h_norm']:.3f}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
