"""Auto-split from signedkan_wip/src/triton_kernels.py 2026-05-11.
CLAUDE.md §6.5 #4 (no monstrosity > 300 LOC).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

from .catmull_rom import catmull_rom_triton
from .inner import signedkan_inner_triton

def _parity_check_catmull_rom(C: int = 16, G: int = 8,
                                B: int = 256, atol: float = 1e-5,
                                rtol: float = 1e-4) -> dict:
    """Runs the Triton kernel on random inputs and compares against
    `splines._catmull_rom_eval`.  Returns a dict with max abs / rel
    error and pass/fail.
    """
    from ..splines import _catmull_rom_eval
    if not torch.cuda.is_available():
        raise RuntimeError("parity test needs CUDA")
    torch.manual_seed(0)
    coef = torch.randn(C, G, device="cuda", dtype=torch.float32)
    x = torch.empty(B, C, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    out_triton = catmull_rom_triton(coef, x, grid=G)
    out_torch = _catmull_rom_eval(coef, x, G)
    abs_err = (out_triton - out_torch).abs().max().item()
    rel_err = ((out_triton - out_torch).abs()
               / (out_torch.abs() + 1e-9)).max().item()
    return dict(
        C=C, G=G, B=B,
        max_abs_err=abs_err,
        max_rel_err=rel_err,
        passed=(abs_err < atol),
    )


def _benchmark_catmull_rom(C: int = 16, G: int = 8,
                            B: int = 200_000, n_warmup: int = 3,
                            n_runs: int = 30) -> dict:
    """Compare Triton vs PyTorch reference at HSiKAN-realistic
    cycle-batch shapes."""
    from ..splines import _catmull_rom_eval
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark needs CUDA")
    torch.manual_seed(0)
    coef = torch.randn(C, G, device="cuda", dtype=torch.float32)
    x = torch.empty(B, C, device="cuda", dtype=torch.float32).uniform_(-1, 1)

    # Warmup both implementations.
    for _ in range(n_warmup):
        _ = _catmull_rom_eval(coef, x, G)
        _ = catmull_rom_triton(coef, x, grid=G)
    torch.cuda.synchronize()

    # PyTorch ref timing.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    for i in range(n_runs):
        starts[i].record()
        _ = _catmull_rom_eval(coef, x, G)
        ends[i].record()
    torch.cuda.synchronize()
    torch_times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    torch_ms = sum(torch_times) / len(torch_times)

    # Triton timing.
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_runs)]
    for i in range(n_runs):
        starts[i].record()
        _ = catmull_rom_triton(coef, x, grid=G)
        ends[i].record()
    torch.cuda.synchronize()
    triton_times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    triton_ms = sum(triton_times) / len(triton_times)

    return dict(
        B=B, C=C, G=G,
        torch_ms=torch_ms,
        triton_ms=triton_ms,
        speedup=torch_ms / triton_ms if triton_ms > 0 else float("inf"),
    )


def _parity_check_signedkan_inner(
    V: int = 1000, d: int = 16, T: int = 5000, k: int = 4, G: int = 8,
    atol: float = 1e-4,
) -> dict:
    """Parity check the fused inner kernel against an explicit PyTorch
    reference implementation."""
    from ..splines import _catmull_rom_eval
    if not torch.cuda.is_available():
        raise RuntimeError("parity test needs CUDA")
    torch.manual_seed(0)
    x = torch.randn(V, d, device="cuda", dtype=torch.float32) * 0.5
    triad_v = torch.randint(0, V, (T, k), device="cuda", dtype=torch.int64)
    triad_sigma = torch.randint(0, 2, (T, k), device="cuda",
                                  dtype=torch.int64) * 2 - 1
    coef_pos = torch.randn(d, G, device="cuda", dtype=torch.float32)
    coef_neg = torch.randn(d, G, device="cuda", dtype=torch.float32)

    # PyTorch reference: explicit per-sign CR + mask + mean.
    # Matches production SignedKANLayer._forward_impl: no tanh wrap,
    # CR input clamped to [-1, 1] (the CR domain) but residual uses
    # raw h_v.  This reference path is no-skip so no residual term.
    h_v = x[triad_v]                                       # (T, k, d)
    inner_pos = _catmull_rom_eval(coef_pos, h_v, G)
    inner_neg = _catmull_rom_eval(coef_neg, h_v, G)
    inner_all = torch.stack([inner_pos, inner_neg], dim=2)  # (T, k, 2, d)
    sign_vals = torch.tensor([1, -1], device="cuda", dtype=torch.int64)
    masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
    masks_e = masks.unsqueeze(-1)                           # (T, k, 2, 1)
    counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)    # (T, 2, 1)
    agg_torch = (inner_all * masks_e).sum(dim=1) / counts   # (T, 2, d)

    # Triton.
    agg_triton = signedkan_inner_triton(
        x, triad_v, triad_sigma, coef_pos, coef_neg, G,
    )

    abs_err = (agg_torch - agg_triton).abs().max().item()
    rel_err = ((agg_torch - agg_triton).abs()
               / (agg_torch.abs() + 1e-9)).max().item()
    return dict(
        V=V, d=d, T=T, k=k, G=G,
        max_abs_err=abs_err,
        max_rel_err=rel_err,
        passed=abs_err < atol,
    )

