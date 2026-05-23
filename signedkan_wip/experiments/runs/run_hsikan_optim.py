"""HSiKAN-CR optimization sweep.

Tests four variants against the canonical HSiKAN-CR recipe:
  1. baseline                — HSiKAN-CR canonical
  2. + attention             — replace mean-pool aggregation with
                               signed-triad tanh-attention + entropy reg
  3. + AdamW + grad-clip     — optimizer/regularization upgrades
  4. + attention + AdamW + grad-clip — full stack
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run_compare import run_one
from signedkan_wip.src.core.highway_signedkan import HighwaySignedKAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/hsikan_optim.json")
    args = ap.parse_args()

    base = HighwaySignedKAN.recommended_training_recipe()

    configs = [
        ("HSiKAN-CR-base",                 {**base, "spline_kind": "catmull_rom"}),
        ("HSiKAN-CR+attn",                 {**base, "spline_kind": "catmull_rom",
                                            "use_attention": True,
                                            "attention_entropy_lam": 0.01}),
        ("HSiKAN-CR+AdamW+clip",           {**base, "spline_kind": "catmull_rom",
                                            "optimizer_kind": "adamw",
                                            "weight_decay": 1e-4,
                                            "grad_clip": 1.0}),
        ("HSiKAN-CR+attn+AdamW+clip",      {**base, "spline_kind": "catmull_rom",
                                            "use_attention": True,
                                            "attention_entropy_lam": 0.01,
                                            "optimizer_kind": "adamw",
                                            "weight_decay": 1e-4,
                                            "grad_clip": 1.0}),
    ]

    results = []
    for tag, kwargs in configs:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one("signedkan", dataset, hidden=32, seed=seed,
                             n_epochs=200, lr=5e-2, **kwargs)
                r["cfg"] = tag
                print(f"  {tag:30s} {dataset:14s} "
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
