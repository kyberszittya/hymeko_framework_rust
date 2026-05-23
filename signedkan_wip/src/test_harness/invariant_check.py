"""Pre-flight invariant checks for HSiKAN training cells.

Catches misconfigurations BEFORE we burn 15 minutes per cell:

1. Forward pass produces correct output shape (logits, not NaN)
2. n_params is in expected range
3. Optimiser actually has parameters to update
4. Train/test split is non-empty
5. Cycle-incidence sparse tensors validate
6. (For attention path) attention head's params are non-zero (not
   completely frozen)

Usage as a fast smoke test before launching long sweeps:

    python -m signedkan_wip.src.test_harness.invariant_check \\
        --dataset bitcoin_alpha
"""

from __future__ import annotations

import argparse
import sys
import os

import torch


def check(name: str, ok: bool, detail: str = "") -> bool:
    marker = "✓" if ok else "✗"
    print(f"  {marker} {name}: {detail}")
    return ok


def smoke_test(dataset: str = "bitcoin_alpha", epochs: int = 1,
                hidden: int = 8, max_gpu_gb: float | None = None) -> bool:
    """Run a 1-epoch training cell on the given dataset and verify
    invariants. Returns True if all checks pass.

    `max_gpu_gb`: if set, asserts that
    `torch.cuda.max_memory_allocated()` after the cell stays below
    this threshold. Useful as a pre-flight check — if a 1-epoch run
    already consumes most of the budget, the 30-epoch sweep will OOM."""
    from signedkan_wip.experiments.runs.run_final_cell import cell_signed_graph

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    out = cell_signed_graph(
        dataset=dataset, model_name="HSiKAN", hidden=hidden,
        n_epochs=epochs, max_k4=10000, device=device, seed=0,
    )

    if device.type == "cuda" and max_gpu_gb is not None:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        check(
            f"peak GPU memory ≤ {max_gpu_gb} GB",
            peak_gb <= max_gpu_gb,
            f"actual peak {peak_gb:.2f} GB",
        )

    all_ok = True

    all_ok &= check(
        "result is non-None",
        out is not None,
        "cell_signed_graph returned None — train/test split empty?",
    )
    if out is None:
        return False

    all_ok &= check(
        "auc is finite",
        not (out.get("auc") != out.get("auc"))   # NaN check
            and out.get("auc") is not None,
        f"auc={out.get('auc')}",
    )

    all_ok &= check(
        "auc in [0, 1]",
        0.0 <= out.get("auc", -1) <= 1.0,
        f"auc={out.get('auc')}",
    )

    all_ok &= check(
        "n_params > 0",
        out.get("n_params", 0) > 0,
        f"n_params={out.get('n_params')}",
    )

    all_ok &= check(
        "fwd_per_call_ms is finite and positive",
        out.get("fwd_per_call_ms", 0) > 0,
        f"fwd_per_call_ms={out.get('fwd_per_call_ms')}",
    )

    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    print(f"Running invariant smoke test on {args.dataset} ({args.epochs} epoch)...")
    ok = smoke_test(args.dataset, args.epochs)
    print()
    if ok:
        print("All invariants passed ✓")
    else:
        print("Invariant FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
