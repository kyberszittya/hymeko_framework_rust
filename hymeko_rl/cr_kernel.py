"""Fused Triton kernel for the learnable Catmull-Rom (CR) activation.

The CR activation is a piecewise *cubic polynomial* per channel (no transcendental). In eager PyTorch it
costs ~18x ``tanh`` because it is ~15 separate op-launches; here the whole forward (segment + cubic + 4
control-point reads + dot) is **one** Triton kernel, and the backward fuses the spline derivative
(``grad_x``) with the control-point gradient scatter (``grad_coef``, via atomic-add). The math is identical
to :func:`hymeko_rl.policy._catmull_rom` (a parity test pins them together).

GPU only — Triton targets CUDA. On CPU this transparently falls back to the eager spline, so the same
``FusedCatmullRom`` module is usable everywhere; the speed win is on the GPU/compute-bound path. See the plan
``docs/plans/2026-06-22-fast-polynomial-activation-kernel/`` and the report.
"""
from __future__ import annotations

# Triton @jit kernels are untyped by construction, and torch.autograd.Function's ctx/override signatures
# can't satisfy --strict; suppress those inherent codes for this kernel-wrapper file only. Correctness is
# pinned by the parity tests (test_cr_kernel.py), not by mypy here.
# mypy: disable-error-code="untyped-decorator, no-untyped-def, no-untyped-call, override"

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:  # pragma: no cover - triton is a dev/GPU dependency
    _HAVE_TRITON = False

_BLOCK = 1024


if _HAVE_TRITON:

    @triton.jit
    def _cr_fwd_kernel(x_ptr, coef_ptr, out_ptr, n, C, G, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        x = tl.load(x_ptr + off, mask=m, other=0.0)
        c = off % C
        xc = tl.minimum(tl.maximum(x, -1.0), 1.0)
        u = (xc + 1.0) * 0.5 * (G - 1)                 # u >= 0, so trunc == floor
        i = tl.minimum((u).to(tl.int32), G - 2)
        fi = i.to(tl.float32)
        t = u - fi
        t2 = t * t
        t3 = t2 * t
        w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
        w_0 = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
        w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
        w_p2 = 0.5 * (t3 - t2)
        base = c * G
        im1 = tl.maximum(i - 1, 0)
        ip1 = tl.minimum(i + 1, G - 1)
        ip2 = tl.minimum(i + 2, G - 1)
        p_m1 = tl.load(coef_ptr + base + im1, mask=m, other=0.0)
        p_0 = tl.load(coef_ptr + base + i, mask=m, other=0.0)
        p_p1 = tl.load(coef_ptr + base + ip1, mask=m, other=0.0)
        p_p2 = tl.load(coef_ptr + base + ip2, mask=m, other=0.0)
        tl.store(out_ptr + off, w_m1 * p_m1 + w_0 * p_0 + w_p1 * p_p1 + w_p2 * p_p2, mask=m)

    @triton.jit
    def _cr_gradx_kernel(go_ptr, x_ptr, coef_ptr, gx_ptr, n, C, G, BLOCK: tl.constexpr):
        """Per-element ``grad_x`` only (no atomics): the spline derivative scaled by ``du/dx``."""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        go = tl.load(go_ptr + off, mask=m, other=0.0)
        x = tl.load(x_ptr + off, mask=m, other=0.0)
        c = off % C
        in_range = (x > -1.0) & (x < 1.0)              # clamp boundary -> zero grad
        xc = tl.minimum(tl.maximum(x, -1.0), 1.0)
        u = (xc + 1.0) * 0.5 * (G - 1)
        i = tl.minimum((u).to(tl.int32), G - 2)
        t = u - i.to(tl.float32)
        t2 = t * t
        dw_m1 = 0.5 * (-3.0 * t2 + 4.0 * t - 1.0)
        dw_0 = 0.5 * (9.0 * t2 - 10.0 * t)
        dw_p1 = 0.5 * (-9.0 * t2 + 8.0 * t + 1.0)
        dw_p2 = 0.5 * (3.0 * t2 - 2.0 * t)
        base = c * G
        p_m1 = tl.load(coef_ptr + base + tl.maximum(i - 1, 0), mask=m, other=0.0)
        p_0 = tl.load(coef_ptr + base + i, mask=m, other=0.0)
        p_p1 = tl.load(coef_ptr + base + tl.minimum(i + 1, G - 1), mask=m, other=0.0)
        p_p2 = tl.load(coef_ptr + base + tl.minimum(i + 2, G - 1), mask=m, other=0.0)
        dout_dt = dw_m1 * p_m1 + dw_0 * p_0 + dw_p1 * p_p1 + dw_p2 * p_p2
        gx = tl.where(in_range, go * dout_dt * (0.5 * (G - 1)), 0.0)
        tl.store(gx_ptr + off, gx, mask=m)

    @triton.jit
    def _cr_gradcoef_kernel(go_ptr, x_ptr, gc_ptr, M, C, G,  # one program per channel; no atomics
                            BLOCK_M: tl.constexpr, GC: tl.constexpr):
        """``grad_coef[c,:]`` by **register accumulation**: program ``c`` streams its channel's ``M`` rows,
        accumulating a ``(G,)`` register vector, then writes the row once. No atomics, no contention."""
        c = tl.program_id(0)
        knots = tl.arange(0, GC)                       # (GC,), GC = next pow2 >= G (Triton requirement)
        acc = tl.zeros((GC,), dtype=tl.float32)
        for m0 in range(0, M, BLOCK_M):
            mm = m0 + tl.arange(0, BLOCK_M)
            mask = mm < M
            off = mm * C + c                           # row-major (M, C): element (mm, c)
            x = tl.load(x_ptr + off, mask=mask, other=0.0)
            go = tl.load(go_ptr + off, mask=mask, other=0.0)
            xc = tl.minimum(tl.maximum(x, -1.0), 1.0)
            u = (xc + 1.0) * 0.5 * (G - 1)
            i = tl.minimum((u).to(tl.int32), G - 2)
            t = u - i.to(tl.float32)
            t2 = t * t
            t3 = t2 * t
            gw_m1 = go * 0.5 * (-t3 + 2.0 * t2 - t)
            gw_0 = go * 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
            gw_p1 = go * 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
            gw_p2 = go * 0.5 * (t3 - t2)
            im1 = tl.maximum(i - 1, 0)
            ip1 = tl.minimum(i + 1, G - 1)
            ip2 = tl.minimum(i + 2, G - 1)
            kk = knots[None, :]                        # (1, G)
            contrib = ((im1[:, None] == kk).to(tl.float32) * gw_m1[:, None]
                       + (i[:, None] == kk).to(tl.float32) * gw_0[:, None]
                       + (ip1[:, None] == kk).to(tl.float32) * gw_p1[:, None]
                       + (ip2[:, None] == kk).to(tl.float32) * gw_p2[:, None])
            contrib = tl.where(mask[:, None], contrib, 0.0)
            acc += tl.sum(contrib, axis=0)             # reduce the block of rows -> (GC,)
        tl.store(gc_ptr + c * G + knots, acc, mask=knots < G)   # write row c (real G stride)


class _FusedCatmullRomFn(torch.autograd.Function):
    """autograd glue: Triton fwd/bwd over a flattened ``(..., C)`` input against ``coef (C, G)``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        n_ch, grid_g = coef.shape
        if x.shape[-1] != n_ch:
            raise ValueError(f"x last dim {x.shape[-1]} != coef channels {n_ch}")
        xf = x.contiguous().reshape(-1)
        coef_c = coef.contiguous()
        out = torch.empty_like(xf)
        n = xf.numel()
        grid = (triton.cdiv(n, _BLOCK),)
        _cr_fwd_kernel[grid](xf, coef_c, out, n, n_ch, grid_g, BLOCK=_BLOCK)
        ctx.save_for_backward(xf, coef_c)
        ctx.n_ch, ctx.grid_g = n_ch, grid_g
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        xf, coef_c = ctx.saved_tensors
        gof = grad_out.contiguous().reshape(-1)
        gx = torch.empty_like(gof)
        n_ch, grid_g = ctx.n_ch, ctx.grid_g
        n = gof.numel()
        # grad_x: per-element (no atomics).
        _cr_gradx_kernel[(triton.cdiv(n, _BLOCK),)](gof, xf, coef_c, gx, n, n_ch, grid_g, BLOCK=_BLOCK)
        # grad_coef: one program per channel, register accumulation (no atomics, no contention).
        gc = torch.empty(n_ch, grid_g, device=coef_c.device, dtype=coef_c.dtype)
        gc_pow2 = 1 << (grid_g - 1).bit_length()        # tl.arange needs a power-of-2 length
        _cr_gradcoef_kernel[(n_ch,)](gof, xf, gc, n // n_ch, n_ch, grid_g, BLOCK_M=_BLOCK, GC=gc_pow2)
        return gx.reshape(grad_out.shape), gc


def fused_catmull_rom(x: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """Fused CR activation. CUDA -> Triton kernel; otherwise the eager spline (same values).

    # Preconditions ``coef`` is ``(C, G)``; ``x`` is ``(..., C)``.
    # Postconditions returns ``(..., C)``, bit-close to :func:`hymeko_rl.policy._catmull_rom`.
    """
    if _HAVE_TRITON and x.is_cuda:
        out: torch.Tensor = _FusedCatmullRomFn.apply(x, coef)
        return out
    from hymeko_rl.policy import _catmull_rom
    return _catmull_rom(coef, x, int(coef.shape[1]))


class FusedCatmullRom(nn.Module):
    """Drop-in learnable CR activation backed by the fused Triton kernel (CPU eager fallback).

    # Preconditions ``n_channels >= 1``, ``grid >= 4``."""

    def __init__(self, n_channels: int, *, grid: int = 5, init_scale: float = 0.1) -> None:
        super().__init__()
        if n_channels < 1 or grid < 4:
            raise ValueError(f"n_channels >= 1 and grid >= 4 required; got {n_channels}, {grid}")
        self.grid = grid
        self.coef = nn.Parameter(torch.randn(n_channels, grid) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_catmull_rom(x, self.coef)
