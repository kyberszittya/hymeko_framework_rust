"""HSiKAN-CR high-sensitivity hyperparameter sweep.

Tests the four highest-sensitivity unswept knobs around the canonical
recipe:

  - Learning rate: lr ∈ {1e-2, 3e-2, 5e-2 [base], 1e-1}
  - Embedding init scale: {0.05, 0.1 [base], 0.2}
  - Entropy schedule sensitivity: eta ∈ {1.0, 5.0 [base], 10.0}
  - Highway gate bias initialisation: {-3.0, -2.0 [base], 0.0}

One-knob-at-a-time around the canonical (lr=5e-2, init=0.1, eta=5.0,
gate_bias=-2.0). 11 configs × 2 datasets × 3 seeds = 66 runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from .run_compare import run_one
from signedkan_wip.src.core.highway_signedkan import HighwaySignedKAN


def _patch_gate_bias(bias: float):
    """Monkey-patch SignedKANLayer's highway gate bias-init at the
    module level for this sweep cell. Dependent on the existing
    `bias.fill_(-2.0)` line in SignedKANLayer.__init__."""
    import signedkan_wip.src.core.signedkan as sk
    orig = sk.SignedKANLayer.__init__
    def patched(self, cfg):
        orig(self, cfg)
        if self.gate_inner is not None:
            with torch.no_grad():
                self.gate_inner.bias.fill_(bias)
        if self.gate_outer is not None:
            with torch.no_grad():
                self.gate_outer.bias.fill_(bias)
    sk.SignedKANLayer.__init__ = patched
    return orig


def _restore_gate(orig):
    import signedkan_wip.src.core.signedkan as sk
    sk.SignedKANLayer.__init__ = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/hsikan_hpsweep.json")
    args = ap.parse_args()

    base = HighwaySignedKAN.recommended_training_recipe()
    base = {**base, "spline_kind": "catmull_rom"}

    # (tag, lr, extra_kwargs override, gate_bias override or None)
    sweep = [
        # Learning rate sweep (other knobs at base values)
        ("lr=1e-2",        1e-2, {}, None),
        ("lr=3e-2",        3e-2, {}, None),
        ("lr=5e-2 (base)", 5e-2, {}, None),
        ("lr=1e-1",        1e-1, {}, None),
        # Init scale sweep
        ("init=0.05",      5e-2, {"init_scale": 0.05}, None),
        ("init=0.20",      5e-2, {"init_scale": 0.20}, None),
        # Entropy eta sweep
        ("eta=1.0",        5e-2, {"entropy_eta": 1.0}, None),
        ("eta=10.0",       5e-2, {"entropy_eta": 10.0}, None),
        # Highway gate bias sweep
        ("gate_bias=-3",   5e-2, {}, -3.0),
        ("gate_bias=0",    5e-2, {}, 0.0),
    ]

    results = []
    for tag, lr, extra, gate_bias in sweep:
        if gate_bias is not None:
            orig = _patch_gate_bias(gate_bias)
        try:
            kwargs = {**base, **extra}
            for dataset in args.datasets:
                for seed in args.seeds:
                    r = run_one("signedkan", dataset, hidden=32, seed=seed,
                                 n_epochs=200, lr=lr, **kwargs)
                    r["cfg"] = tag
                    print(f"  {tag:20s} {dataset:14s} "
                          f"seed={seed}  best_ep={r['best_epoch']:3d}  "
                          f"AUC={r['test_auc']:.4f}  "
                          f"F1m={r['test_f1_macro']:.4f}  "
                          f"{r['elapsed_s']:.1f}s")
                    results.append(r)
        finally:
            if gate_bias is not None:
                _restore_gate(orig)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
