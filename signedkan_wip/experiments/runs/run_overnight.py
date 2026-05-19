"""Overnight cross-dataset kitchen-sink run.

Runs the recommended deployment configuration of SignedKAN — every
positive-effect lever stacked — across multiple signed-graph
benchmarks for an apples-to-apples cross-dataset evaluation.

Configurations:
  EC                    : the L=1 baseline (recipe of §IV.8)
  EC+R2                 : R2 alone (the cleanest single-regulariser win)
  KITCHEN-SINK          : every load-bearing positive lever stacked

Datasets:
  bitcoin_alpha (~3.7k nodes)
  bitcoin_otc   (~5.9k nodes)
  slashdot      (~82k nodes; reduced hidden_dim due to GPU memory)

The "kitchen sink" stacks:
  - Class-weighted BCE + early stopping (EC recipe base)
  - Spectral entropy regulariser (KL-feedback Lyapunov-safe)
  - R2 vertex-degree participation regulariser
  - Multi-layer L=3 + LayerNorm + weight-sharing + JK-concat + sum-pool
  - Heterogeneous skip placement (inner: residual, outer: none)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .run_compare import run_one


def _cfg_for(dataset: str) -> dict:
    """Per-dataset hidden_dim — scale down on Slashdot due to GPU
    memory; B-spline on Slashdot OOMs at any h>=16, so we use
    Catmull-Rom on that dataset for memory headroom."""
    if dataset == "slashdot":
        return {"hidden": 8, "spline_kind": "catmull_rom"}
    return {"hidden": 32, "spline_kind": "bspline"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--include-slashdot", action="store_true")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/overnight.json")
    args = ap.parse_args()

    datasets = list(args.datasets)
    if args.include_slashdot and "slashdot" not in datasets:
        datasets.append("slashdot")

    # (tag, kwargs override; common base added below)
    base_ec = dict(
        early_stopping=True,
        class_weighted=True,
        grid=5,
        val_every=5,
    )
    base_r2 = {**base_ec, "participation_lam": 0.05}
    kitchen_sink = {
        **base_ec,
        # Spectral entropy with KL-feedback schedule.
        "entropy_lam0": 0.01, "entropy_target": 0.5, "entropy_eta": 5.0,
        # R2 participation reg.
        "participation_lam": 0.05,
        # Multi-layer L=3 with LayerNorm + weight sharing + JK-concat.
        "n_layers": 3,
        "spline_kinds": ["bspline"] * 3,
        "pool_mode": "sum",
        "jk_mode": "concat",
        "layer_norm_between": True,
        "share_weights": True,
        # Heterogeneous skip placement (CV head-neck-spine analogy).
        "inner_skip": "residual",
        "outer_skip": "none",
    }

    configs = [
        ("EC",            base_ec),
        ("EC+R2",         base_r2),
        ("kitchen-sink",  kitchen_sink),
    ]

    results = []
    t0 = time.time()
    for dataset in datasets:
        ds_cfg = _cfg_for(dataset)
        for tag, kw in configs:
            for seed in args.seeds:
                kwargs = {**kw, **{k: v for k, v in ds_cfg.items()
                                    if k not in ("hidden",)}}
                r = run_one("signedkan", dataset,
                             hidden=ds_cfg["hidden"], seed=seed,
                             n_epochs=args.n_epochs, lr=5e-2, **kwargs)
                r["cfg"] = tag
                print(f"  {tag:14s} {dataset:14s} "
                      f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"params={r['n_params']:,}  "
                      f"{r['elapsed_s']:.1f}s")
                results.append(r)
                # Snapshot after each run so a crash mid-overnight
                # doesn't lose everything.
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\noverall elapsed: {time.time()-t0:.0f}s  ({len(results)} runs)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
