"""Parameter-sensitivity sweep for the Triton kernels.

Measures how speedup, runtime, and memory scale with the kernel's
configuration parameters (BLOCK_T, BLOCK_D, hidden dim d, cycle
count T, arity k, grid size G).  Output is a JSONL file plus a
human-readable summary on stdout.

Usage:
    python -m signedkan_wip.src.triton_kernels_sensitivity > sensitivity.txt

The sweeps are designed to map the kernel's performance landscape
so an autotuner can pick optimal block sizes per (d, T) combination
without running the full grid every time.
"""
from __future__ import annotations

import json
import statistics
import time

import torch

from .splines import _catmull_rom_eval
from ..triton_kernels import signedkan_inner_triton


def _torch_signedkan_inner(x, triad_v, triad_sigma, coef_pos, coef_neg, G):
    """Production-equivalent ref: no tanh wrap, CR clamps internally."""
    h_v = x[triad_v]
    inner_pos = _catmull_rom_eval(coef_pos, h_v, G)
    inner_neg = _catmull_rom_eval(coef_neg, h_v, G)
    inner_all = torch.stack([inner_pos, inner_neg], dim=2)
    sign_vals = torch.tensor([1, -1], device=x.device, dtype=torch.int64)
    masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
    counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (inner_all * masks.unsqueeze(-1)).sum(dim=1) / counts


def _bench(fn, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_runs):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_runs


def _make_inputs(V, d, T, k, G, device="cuda"):
    torch.manual_seed(0)
    x = torch.randn(V, d, device=device, dtype=torch.float32) * 0.5
    triad_v = torch.randint(0, V, (T, k), device=device, dtype=torch.int64)
    triad_sigma = (torch.randint(0, 2, (T, k), device=device,
                                   dtype=torch.int64) * 2 - 1)
    coef_pos = torch.randn(d, G, device=device, dtype=torch.float32)
    coef_neg = torch.randn(d, G, device=device, dtype=torch.float32)
    return x, triad_v, triad_sigma, coef_pos, coef_neg


def sweep_T(d=4, k=4, G=8, V=10_000):
    """How does speedup scale with cycle count T?"""
    rows = []
    for T in (1_000, 10_000, 50_000, 100_000, 200_000, 500_000):
        V_use = max(V, T // 10)
        x, tv, ts, cp, cn = _make_inputs(V_use, d, T, k, G)
        t_torch = _bench(lambda: _torch_signedkan_inner(x, tv, ts, cp, cn, G))
        t_triton = _bench(lambda: signedkan_inner_triton(x, tv, ts, cp, cn, G))
        rows.append(dict(
            sweep="T", T=T, d=d, k=k, G=G,
            torch_ms=t_torch, triton_ms=t_triton,
            speedup=t_torch / t_triton,
        ))
    return rows


def sweep_d(T=100_000, k=4, G=8, V=10_000):
    """How does speedup scale with hidden dim d?"""
    rows = []
    for d in (2, 4, 8, 16, 32, 64):
        x, tv, ts, cp, cn = _make_inputs(V, d, T, k, G)
        t_torch = _bench(lambda: _torch_signedkan_inner(x, tv, ts, cp, cn, G))
        t_triton = _bench(lambda: signedkan_inner_triton(x, tv, ts, cp, cn, G))
        rows.append(dict(
            sweep="d", T=T, d=d, k=k, G=G,
            torch_ms=t_torch, triton_ms=t_triton,
            speedup=t_torch / t_triton,
        ))
    return rows


def sweep_k(T=100_000, d=4, G=8, V=10_000):
    """How does speedup scale with arity k?"""
    rows = []
    for k in (2, 3, 4, 5, 6, 8):
        x, tv, ts, cp, cn = _make_inputs(V, d, T, k, G)
        t_torch = _bench(lambda: _torch_signedkan_inner(x, tv, ts, cp, cn, G))
        t_triton = _bench(lambda: signedkan_inner_triton(x, tv, ts, cp, cn, G))
        rows.append(dict(
            sweep="k", T=T, d=d, k=k, G=G,
            torch_ms=t_torch, triton_ms=t_triton,
            speedup=t_torch / t_triton,
        ))
    return rows


def sweep_G(T=100_000, d=4, k=4, V=10_000):
    """How does speedup scale with grid (control point) count G?"""
    rows = []
    for G in (4, 8, 16, 32):
        x, tv, ts, cp, cn = _make_inputs(V, d, T, k, G)
        t_torch = _bench(lambda: _torch_signedkan_inner(x, tv, ts, cp, cn, G))
        t_triton = _bench(lambda: signedkan_inner_triton(x, tv, ts, cp, cn, G))
        rows.append(dict(
            sweep="G", T=T, d=d, k=k, G=G,
            torch_ms=t_torch, triton_ms=t_triton,
            speedup=t_torch / t_triton,
        ))
    return rows


def sweep_block_T(T=100_000, d=4, k=4, G=8, V=10_000):
    """Find the optimal BLOCK_T for the kernel."""
    rows = []
    x, tv, ts, cp, cn = _make_inputs(V, d, T, k, G)
    for BLOCK_T in (8, 16, 32, 64, 128, 256):
        t_triton = _bench(lambda: signedkan_inner_triton(
            x, tv, ts, cp, cn, G, BLOCK_T=BLOCK_T,
        ))
        rows.append(dict(
            sweep="BLOCK_T", T=T, d=d, k=k, G=G,
            BLOCK_T=BLOCK_T, triton_ms=t_triton,
        ))
    return rows


def sweep_block_D(T=100_000, d=16, k=4, G=8, V=10_000):
    """Find the optimal BLOCK_D for the kernel.  d=16 chosen so the
    sweep can cover BLOCK_D values up to d itself."""
    rows = []
    x, tv, ts, cp, cn = _make_inputs(V, d, T, k, G)
    for BLOCK_D in (4, 8, 16, 32, 64):
        if BLOCK_D > d:
            BLOCK_D = d
        t_triton = _bench(lambda: signedkan_inner_triton(
            x, tv, ts, cp, cn, G, BLOCK_D=BLOCK_D,
        ))
        rows.append(dict(
            sweep="BLOCK_D", T=T, d=d, k=k, G=G,
            BLOCK_D=BLOCK_D, triton_ms=t_triton,
        ))
    return rows


def measure_memory(T=200_000, d=4, k=4, G=8, V=10_000):
    """Compare peak memory: PyTorch path vs Triton kernel."""
    x, tv, ts, cp, cn = _make_inputs(V, d, T, k, G)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        _ = _torch_signedkan_inner(x, tv, ts, cp, cn, G)
    torch.cuda.synchronize()
    peak_torch = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        _ = signedkan_inner_triton(x, tv, ts, cp, cn, G)
    torch.cuda.synchronize()
    peak_triton = torch.cuda.max_memory_allocated() / 1024**2

    return dict(
        T=T, d=d, k=k, G=G,
        torch_peak_mb=peak_torch,
        triton_peak_mb=peak_triton,
        memory_savings=(peak_torch - peak_triton) / peak_torch * 100,
    )


def main():
    print("=" * 60)
    print("HSiKAN Triton kernel — parameter sensitivity sweep")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    print("\n[1] Sweep over T (cycle count) — d=4, k=4, G=8")
    print(f"{'T':>8}  {'PyTorch (ms)':>14}  {'Triton (ms)':>14}  "
          f"{'speedup':>10}")
    for r in sweep_T():
        print(f"{r['T']:>8}  {r['torch_ms']:>14.3f}  "
              f"{r['triton_ms']:>14.3f}  {r['speedup']:>10.1f}×")

    print("\n[2] Sweep over d (hidden dim) — T=100K, k=4, G=8")
    print(f"{'d':>8}  {'PyTorch (ms)':>14}  {'Triton (ms)':>14}  "
          f"{'speedup':>10}")
    for r in sweep_d():
        print(f"{r['d']:>8}  {r['torch_ms']:>14.3f}  "
              f"{r['triton_ms']:>14.3f}  {r['speedup']:>10.1f}×")

    print("\n[3] Sweep over k (arity) — T=100K, d=4, G=8")
    print(f"{'k':>8}  {'PyTorch (ms)':>14}  {'Triton (ms)':>14}  "
          f"{'speedup':>10}")
    for r in sweep_k():
        print(f"{r['k']:>8}  {r['torch_ms']:>14.3f}  "
              f"{r['triton_ms']:>14.3f}  {r['speedup']:>10.1f}×")

    print("\n[4] Sweep over G (grid size) — T=100K, d=4, k=4")
    print(f"{'G':>8}  {'PyTorch (ms)':>14}  {'Triton (ms)':>14}  "
          f"{'speedup':>10}")
    for r in sweep_G():
        print(f"{r['G']:>8}  {r['torch_ms']:>14.3f}  "
              f"{r['triton_ms']:>14.3f}  {r['speedup']:>10.1f}×")

    print("\n[5] Sweep over BLOCK_T — T=100K, d=4, k=4, G=8")
    print(f"{'BLOCK_T':>8}  {'Triton (ms)':>14}")
    for r in sweep_block_T():
        print(f"{r['BLOCK_T']:>8}  {r['triton_ms']:>14.3f}")

    print("\n[6] Sweep over BLOCK_D — T=100K, d=16, k=4, G=8")
    print(f"{'BLOCK_D':>8}  {'Triton (ms)':>14}")
    for r in sweep_block_D():
        print(f"{r['BLOCK_D']:>8}  {r['triton_ms']:>14.3f}")

    print("\n[7] Peak memory comparison — T=200K, d=4, k=4, G=8")
    m = measure_memory()
    print(f"  PyTorch: {m['torch_peak_mb']:>10.2f} MB")
    print(f"  Triton : {m['triton_peak_mb']:>10.2f} MB")
    print(f"  Savings: {m['memory_savings']:>10.2f}%")


if __name__ == "__main__":
    main()
