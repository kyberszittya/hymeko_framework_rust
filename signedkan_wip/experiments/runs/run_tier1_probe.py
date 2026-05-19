"""Tier 1 (B+C) probe.

Two parts:
  1. Numerical equivalence + speedup of the eigvalsh swap in
     `_spectral_distribution` (no run_one needed — just compare the
     pure function on a representative tensor and time both).
  2. End-to-end canonical-recipe sanity: HSiKAN canonical at seed 0
     under the new code path; confirms `last_h_norm`, `last_lam_eff`,
     and the test metrics are sane.

The previous svdvals form is regenerated locally for the equivalence
check; the production module is already on eigvalsh.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from signedkan_wip.src.datasets import load
from signedkan_wip.src.entropy_reg import _spectral_distribution
from signedkan_wip.src.highway_signedkan import HighwaySignedKAN
from .run_compare import run_one


def _spectral_distribution_svd(A: torch.Tensor, eps: float = 1e-12):
    """Reference (old) implementation, kept here for equivalence test."""
    s = torch.linalg.svdvals(A)
    p = s.pow(2)
    return p / (p.sum() + eps)


def equivalence_probe(n_iter: int = 200) -> dict:
    """Compare svdvals vs eigvalsh on a representative shape."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load("bitcoin_alpha")
    n_nodes, d = g.n_nodes, 32

    torch.manual_seed(0)
    A = torch.randn(n_nodes, d, device=device) * 0.05
    A.requires_grad_(False)

    # Numerical equivalence — must match to FP-tolerance after sort.
    p_svd  = _spectral_distribution_svd(A).detach().cpu().sort(descending=True).values
    p_eigh = _spectral_distribution(A, 1e-12).detach().cpu().sort(descending=True).values
    absdiff = float((p_svd - p_eigh).abs().max().item())
    assert absdiff < 1e-5, (
        f"FAIL equivalence: max|Δp| = {absdiff:.3e} (expected < 1e-5)"
    )

    # Timing — warm-up then n_iter calls each.
    if device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(3):
        _spectral_distribution_svd(A)
        _spectral_distribution(A, 1e-12)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _spectral_distribution_svd(A)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_svd = (time.perf_counter() - t0) / n_iter * 1e3   # ms/call

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _spectral_distribution(A, 1e-12)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_eigh = (time.perf_counter() - t0) / n_iter * 1e3

    return dict(
        n_nodes=n_nodes, d=d, device=str(device),
        ms_per_call_svd=round(t_svd, 4),
        ms_per_call_eigh=round(t_eigh, 4),
        speedup_x=round(t_svd / max(t_eigh, 1e-9), 2),
        max_abs_diff=absdiff,
    )


def canonical_seed0(n_epochs: int = 120) -> dict:
    """End-to-end canonical-recipe sanity at seed 0 on bitcoin_alpha."""
    base = HighwaySignedKAN.recommended_training_recipe()
    base = {**base, "spline_kind": "catmull_rom"}
    r = run_one(
        "signedkan", "bitcoin_alpha",
        hidden=32, seed=0, n_epochs=n_epochs, lr=5e-2, **base,
    )
    return dict(
        test_auc=float(r["test_auc"]),
        test_f1_macro=float(r["test_f1_macro"]),
        elapsed_s=float(r["elapsed_s"]),
        last_h_norm=float(r.get("last_h_norm", float("nan"))),
        last_lam_eff=float(r.get("last_lam_eff", float("nan"))),
        n_params=int(r["n_params"]),
        best_epoch=int(r["best_epoch"]),
    )


def main():
    print("[1/2] Equivalence + speedup probe ...")
    eq = equivalence_probe(n_iter=200)
    print(f"  {eq}")

    print("[2/2] Canonical-recipe sanity (HSiKAN-CR, alpha, seed=0, 120ep) ...")
    s0 = canonical_seed0(n_epochs=120)
    print(f"  {s0}")

    out = {
        "equivalence_probe": eq,
        "canonical_seed0": s0,
    }
    out_path = Path(
        "signedkan_wip/experiments/results/tier1_probe.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
