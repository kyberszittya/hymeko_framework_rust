"""Triton kernel for CliffordFIR -- fused causal multivector FIR.

The PyTorch reference at hymeko_neuro/experiments/sequence/clifford_fir.py
materialises an unfold window tensor of shape (B, L, c_in, K, 4) and
then a geometric-product intermediate of shape (B, L, c_out, c_in, K, 4)
before reducing to (B, L, c_out, 4). At c_in=c_out=4, K=8, L=256, B=32:

  window      memory ≈  B·L·c·K·4·4 B  ≈  4.2 MB
  gp_inter    memory ≈  B·L·c·c·K·4·4 B ≈ 16.8 MB

A single Triton kernel collapses both intermediates into a per-output
(B, L, c_out, 4) accumulator. Forward only here; backward is via
PyTorch autograd through a custom-op stub that calls into the fused
kernel.

Algebra: Cl(2,0), 4 components (scalar, e₁, e₂, e₁₂).
Geometric product (a·b):
    s    = a0·b0 + a1·b1 + a2·b2 − a12·b12
    e1   = a0·b1 + a1·b0 − a2·b12 + a12·b2
    e2   = a0·b2 + a1·b12 + a2·b0 − a12·b1
    e12  = a0·b12 + a1·b2 − a2·b1 + a12·b0
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ── Forward kernel ──────────────────────────────────────────────────

@triton.jit
def _clifford_fir_forward_kernel(
    x_ptr,            # (B, L, C_in, 4)        input multivector stream
    taps_ptr,         # (C_out, C_in, K, 4)    learnable filter taps
    y_ptr,            # (B, L, C_out, 4)       output multivector stream
    B, L, C_in, C_out, K,
    sB_x, sL_x, sC_x,   # x strides (last stride = 1 since contiguous)
    sCout_t, sCin_t, sK_t,  # taps strides
    sB_y, sL_y, sC_y,   # y strides
    BLOCK_L: tl.constexpr,
):
    """One program per (b, c_out, l-block). Loops c_in × K, accumulates."""
    pid_b   = tl.program_id(0)
    pid_co  = tl.program_id(1)
    pid_l   = tl.program_id(2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L

    acc0  = tl.zeros((BLOCK_L,), dtype=tl.float32)
    acc1  = tl.zeros((BLOCK_L,), dtype=tl.float32)
    acc2  = tl.zeros((BLOCK_L,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_L,), dtype=tl.float32)

    for ci in range(C_in):
        for k in range(K):
            # x position is offs_l - k; bounds-check for the causal pad.
            pos = offs_l - k
            valid = (pos >= 0) & mask_l
            base = pid_b * sB_x + pos * sL_x + ci * sC_x
            x0  = tl.load(x_ptr + base + 0, mask=valid, other=0.0)
            x1  = tl.load(x_ptr + base + 1, mask=valid, other=0.0)
            x2  = tl.load(x_ptr + base + 2, mask=valid, other=0.0)
            x12 = tl.load(x_ptr + base + 3, mask=valid, other=0.0)
            tbase = pid_co * sCout_t + ci * sCin_t + k * sK_t
            t0  = tl.load(taps_ptr + tbase + 0)
            t1  = tl.load(taps_ptr + tbase + 1)
            t2  = tl.load(taps_ptr + tbase + 2)
            t12 = tl.load(taps_ptr + tbase + 3)
            # Geometric product taps ⊗ x (taps first, x second).
            acc0  += t0  * x0  + t1  * x1  + t2  * x2  - t12 * x12
            acc1  += t0  * x1  + t1  * x0  - t2  * x12 + t12 * x2
            acc2  += t0  * x2  + t1  * x12 + t2  * x0  - t12 * x1
            acc12 += t0  * x12 + t1  * x2  - t2  * x1  + t12 * x0

    # Write output (B, L, C_out, 4) for this (b, c_out, l-block).
    out_base = pid_b * sB_y + offs_l * sL_y + pid_co * sC_y
    tl.store(y_ptr + out_base + 0, acc0,  mask=mask_l)
    tl.store(y_ptr + out_base + 1, acc1,  mask=mask_l)
    tl.store(y_ptr + out_base + 2, acc2,  mask=mask_l)
    tl.store(y_ptr + out_base + 3, acc12, mask=mask_l)


# ── Backward kernels ────────────────────────────────────────────────
# Forward output:  y[b, t, co, :] = Σ_{ci, k} taps[co, ci, k, :] ⊗ x[b, t-k, ci, :]
#
# Backward derivation (Cl(2,0) geometric product is non-commutative;
# treat tap a, input b: components y_j = Σ_p Σ_q M_{j,p,q} a_p b_q
# where M is the Cl(2,0) structure tensor below). Then
#   dL/db_q = Σ_j Σ_p M_{j,p,q} a_p dL/dy_j        (input gradient)
#   dL/da_p = Σ_j Σ_q M_{j,p,q} b_q dL/dy_j        (tap gradient)
#
# The Cl(2,0) M tensor (indexing by (j, p, q) over {0=s, 1=e1, 2=e2, 3=e12}):
#   from clifford.py geometric_product expansion. Closed-form: only 16
#   non-zero entries per j, expressed inline in the kernel below.

@triton.jit
def _clifford_fir_backward_dx_kernel(
    grad_y_ptr,       # (B, L, C_out, 4)
    taps_ptr,         # (C_out, C_in, K, 4)
    grad_x_ptr,       # (B, L, C_in, 4)
    B, L, C_in, C_out, K,
    sB_gy, sL_gy, sC_gy,
    sCout_t, sCin_t, sK_t,
    sB_gx, sL_gx, sC_gx,
    BLOCK_L: tl.constexpr,
):
    """Per (b, c_in, l-block): dL/dx[b, s, ci, :] = Σ_{co, k} (tap[co, ci, k])^T ⊗ dy[b, s+k, co, :]

    where T is the appropriate "reverse" geometric-product map for the
    Cl(2,0) algebra (derivative of (a ⊗ x) wrt x).
    """
    pid_b   = tl.program_id(0)
    pid_ci  = tl.program_id(1)
    pid_l   = tl.program_id(2)

    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L

    acc0  = tl.zeros((BLOCK_L,), dtype=tl.float32)
    acc1  = tl.zeros((BLOCK_L,), dtype=tl.float32)
    acc2  = tl.zeros((BLOCK_L,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_L,), dtype=tl.float32)

    for co in range(C_out):
        for k in range(K):
            # dy position is offs_l + k; valid if < L.
            pos = offs_l + k
            valid = (pos < L) & mask_l
            base = pid_b * sB_gy + pos * sL_gy + co * sC_gy
            dy0  = tl.load(grad_y_ptr + base + 0, mask=valid, other=0.0)
            dy1  = tl.load(grad_y_ptr + base + 1, mask=valid, other=0.0)
            dy2  = tl.load(grad_y_ptr + base + 2, mask=valid, other=0.0)
            dy12 = tl.load(grad_y_ptr + base + 3, mask=valid, other=0.0)
            tbase = co * sCout_t + pid_ci * sCin_t + k * sK_t
            t0  = tl.load(taps_ptr + tbase + 0)
            t1  = tl.load(taps_ptr + tbase + 1)
            t2  = tl.load(taps_ptr + tbase + 2)
            t12 = tl.load(taps_ptr + tbase + 3)
            # dL/dx_q = Σ_{j,p} M_{j,p,q} a_p dL/dy_j   (closed form below)
            #   q=0   : t0*dy0 + t1*dy1 + t2*dy2 + t12*dy12
            #   q=1   : t0*dy1 + t1*dy0 - t2*dy12 - t12*dy2
            #   q=2   : t0*dy2 + t1*dy12 + t2*dy0 + t12*dy1
            #   q=12  : t0*dy12 + t1*dy2 - t2*dy1 - t12*dy0
            acc0  += t0  * dy0  + t1  * dy1  + t2  * dy2  + t12 * dy12
            acc1  += t0  * dy1  + t1  * dy0  - t2  * dy12 - t12 * dy2
            acc2  += t0  * dy2  + t1  * dy12 + t2  * dy0  + t12 * dy1
            acc12 += t0  * dy12 + t1  * dy2  - t2  * dy1  - t12 * dy0

    out_base = pid_b * sB_gx + offs_l * sL_gx + pid_ci * sC_gx
    tl.store(grad_x_ptr + out_base + 0, acc0,  mask=mask_l)
    tl.store(grad_x_ptr + out_base + 1, acc1,  mask=mask_l)
    tl.store(grad_x_ptr + out_base + 2, acc2,  mask=mask_l)
    tl.store(grad_x_ptr + out_base + 3, acc12, mask=mask_l)


@triton.jit
def _clifford_fir_backward_dtaps_kernel(
    grad_y_ptr,       # (B, L, C_out, 4)
    x_ptr,            # (B, L, C_in, 4)
    grad_taps_ptr,    # (C_out, C_in, K, 4)
    B, L, C_in, C_out, K,
    sB_gy, sL_gy, sC_gy,
    sB_x, sL_x, sC_x,
    sCout_gt, sCin_gt, sK_gt,
    BLOCK_L: tl.constexpr,
):
    """Per (b, c_out, c_in, k, l-block): partial dL/dtaps reduced
    over the L-block, then atomic-added to the global grad_taps.

    Each program handles a single (b, c_out, c_in, k) and a chunk of
    L of size BLOCK_L. Grid: (B, C_out * C_in * K, L_blocks).
    The flattened middle axis is decoded into (co, ci, k) inside.
    """
    pid_b   = tl.program_id(0)
    pid_x   = tl.program_id(1)
    pid_lb  = tl.program_id(2)

    pid_co = pid_x // (C_in * K)
    pid_ci = (pid_x // K) % C_in
    pid_k  = pid_x % K

    offs_l = pid_lb * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L
    # x position is offs_l - k; valid if >= 0 AND mask_l.
    pos = offs_l - pid_k
    valid = (pos >= 0) & mask_l

    gy_base = pid_b * sB_gy + offs_l * sL_gy + pid_co * sC_gy
    dy0  = tl.load(grad_y_ptr + gy_base + 0, mask=mask_l, other=0.0)
    dy1  = tl.load(grad_y_ptr + gy_base + 1, mask=mask_l, other=0.0)
    dy2  = tl.load(grad_y_ptr + gy_base + 2, mask=mask_l, other=0.0)
    dy12 = tl.load(grad_y_ptr + gy_base + 3, mask=mask_l, other=0.0)
    x_base = pid_b * sB_x + pos * sL_x + pid_ci * sC_x
    x0  = tl.load(x_ptr + x_base + 0, mask=valid, other=0.0)
    x1  = tl.load(x_ptr + x_base + 1, mask=valid, other=0.0)
    x2  = tl.load(x_ptr + x_base + 2, mask=valid, other=0.0)
    x12 = tl.load(x_ptr + x_base + 3, mask=valid, other=0.0)

    # Per-pos contributions then sum over BLOCK_L.
    p0  = dy0  * x0  + dy1  * x1  + dy2  * x2  + dy12 * x12
    p1  = dy0  * x1  + dy1  * x0  + dy2  * x12 + dy12 * x2
    p2  = dy0  * x2  - dy1  * x12 + dy2  * x0  - dy12 * x1
    p12 = -dy0 * x12 + dy1  * x2  - dy2  * x1  + dy12 * x0
    s0  = tl.sum(p0,  axis=0)
    s1  = tl.sum(p1,  axis=0)
    s2  = tl.sum(p2,  axis=0)
    s12 = tl.sum(p12, axis=0)

    out_base = pid_co * sCout_gt + pid_ci * sCin_gt + pid_k * sK_gt
    tl.atomic_add(grad_taps_ptr + out_base + 0, s0)
    tl.atomic_add(grad_taps_ptr + out_base + 1, s1)
    tl.atomic_add(grad_taps_ptr + out_base + 2, s2)
    tl.atomic_add(grad_taps_ptr + out_base + 3, s12)


def clifford_fir_backward_triton(
    grad_y: torch.Tensor,    # (B, L, C_out, 4)
    x: torch.Tensor,         # (B, L, C_in, 4)
    taps: torch.Tensor,      # (C_out, C_in, K, 4)
    block_l: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused backward: returns (grad_x, grad_taps).

    For correctness during development we compare against the PyTorch
    autograd-of-reference path; see ``_CliffordFIRAutograd.backward``
    which can be switched between the two via ``USE_TRITON_BACKWARD``.
    """
    assert grad_y.is_cuda and x.is_cuda and taps.is_cuda
    grad_y = grad_y.contiguous(); x = x.contiguous(); taps = taps.contiguous()
    B, L, C_in, _ = x.shape
    C_out, _, K, _ = taps.shape

    grad_x = torch.empty_like(x)
    grid_x = (B, C_in, triton.cdiv(L, block_l))
    sB_gy, sL_gy, sC_gy, _ = grad_y.stride()
    sCout_t, sCin_t, sK_t, _ = taps.stride()
    sB_gx, sL_gx, sC_gx, _ = grad_x.stride()
    _clifford_fir_backward_dx_kernel[grid_x](
        grad_y, taps, grad_x,
        B, L, C_in, C_out, K,
        sB_gy, sL_gy, sC_gy,
        sCout_t, sCin_t, sK_t,
        sB_gx, sL_gx, sC_gx,
        BLOCK_L=block_l,
    )

    # dtaps must be zero-initialised because the kernel uses atomic_add.
    grad_taps = torch.zeros_like(taps)
    sB_x, sL_x, sC_x, _ = x.stride()
    sCout_gt, sCin_gt, sK_gt, _ = grad_taps.stride()
    grid_t = (B, C_out * C_in * K, triton.cdiv(L, block_l))
    _clifford_fir_backward_dtaps_kernel[grid_t](
        grad_y, x, grad_taps,
        B, L, C_in, C_out, K,
        sB_gy, sL_gy, sC_gy,
        sB_x, sL_x, sC_x,
        sCout_gt, sCin_gt, sK_gt,
        BLOCK_L=block_l,
    )
    return grad_x, grad_taps


# Toggle: True = use Triton bwd; False = fall back to PyTorch reference.
USE_TRITON_BACKWARD = True


# ── Python launchers ────────────────────────────────────────────────

def clifford_fir_forward_triton(
    x: torch.Tensor,         # (B, L, C_in, 4)
    taps: torch.Tensor,      # (C_out, C_in, K, 4)
    block_l: int = 32,
) -> torch.Tensor:
    """Triton-fused causal CliffordFIR forward.

    Same semantics as ``hymeko_neuro.experiments.sequence.clifford_fir.CliffordFIR
    .forward`` -- causal, zero-padded for t < k. Returns (B, L, C_out, 4).
    """
    assert x.is_cuda and taps.is_cuda, "Triton kernel requires CUDA tensors"
    assert x.dim() == 4 and x.shape[-1] == 4, f"x shape: {tuple(x.shape)}"
    assert taps.dim() == 4 and taps.shape[-1] == 4, f"taps: {tuple(taps.shape)}"
    B, L, C_in, _ = x.shape
    C_out, C_in_t, K, _ = taps.shape
    assert C_in == C_in_t, f"C_in mismatch: {C_in} vs {C_in_t}"
    x = x.contiguous()
    taps = taps.contiguous()
    y = torch.empty((B, L, C_out, 4), device=x.device, dtype=x.dtype)
    grid = (B, C_out, triton.cdiv(L, block_l))
    sB_x, sL_x, sC_x, _ = x.stride()
    sCout_t, sCin_t, sK_t, _ = taps.stride()
    sB_y, sL_y, sC_y, _ = y.stride()
    _clifford_fir_forward_kernel[grid](
        x, taps, y,
        B, L, C_in, C_out, K,
        sB_x, sL_x, sC_x,
        sCout_t, sCin_t, sK_t,
        sB_y, sL_y, sC_y,
        BLOCK_L=block_l,
    )
    return y


# ── Autograd wrapper: Triton forward, PyTorch backward ──────────────

class _CliffordFIRAutograd(torch.autograd.Function):
    """Forward via Triton; backward via PyTorch reference for now.

    The forward saves the inputs so the PyTorch reference (in clifford_fir.py)
    can be re-run during backward to compute gradients. This is correctness-
    over-perf. A fused backward kernel is the natural v2 follow-up.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, taps: torch.Tensor):
        ctx.save_for_backward(x, taps)
        return clifford_fir_forward_triton(x, taps)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, taps = ctx.saved_tensors
        if USE_TRITON_BACKWARD and x.is_cuda:
            return clifford_fir_backward_triton(grad_y.contiguous(), x, taps)
        # Reference reverse path (PyTorch autograd of the inline reference).
        from hymeko_neuro.experiments.sequence.clifford import geometric_product
        with torch.enable_grad():
            x_g = x.detach().clone().requires_grad_(True)
            t_g = taps.detach().clone().requires_grad_(True)
            B, L, C_in, _ = x_g.shape
            C_out, _, K, _ = t_g.shape
            pad = torch.zeros(B, K - 1, C_in, 4,
                              dtype=x_g.dtype, device=x_g.device)
            x_pad = torch.cat([pad, x_g], dim=1)
            windows = x_pad.unfold(dimension=1, size=K, step=1)
            windows = windows.permute(0, 1, 2, 4, 3).contiguous()
            windows = windows.flip(dims=(3,))
            windows_e = windows.unsqueeze(2)
            taps_e = t_g.view(1, 1, C_out, C_in, K, 4)
            prods = geometric_product(taps_e, windows_e)
            y = prods.sum(dim=(3, 4))
            grads = torch.autograd.grad(
                outputs=y, inputs=(x_g, t_g),
                grad_outputs=grad_y, retain_graph=False,
            )
        return grads[0], grads[1]


def clifford_fir_triton(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """User-facing autograd-friendly entry point for the Triton kernel.

    Falls back to the PyTorch reference if inputs are not on CUDA.
    """
    if not (x.is_cuda and taps.is_cuda):
        # CPU fallback: call the reference module.
        from hymeko_neuro.experiments.sequence.clifford_fir import CliffordFIR
        mod = CliffordFIR(K=taps.shape[2], c_in=taps.shape[1],
                            c_out=taps.shape[0])
        with torch.no_grad():
            mod.taps.copy_(taps)
        return mod(x)
    return _CliffordFIRAutograd.apply(x, taps)
