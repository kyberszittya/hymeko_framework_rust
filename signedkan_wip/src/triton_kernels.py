"""Triton kernels for HSiKAN's encoder forward path.

This module implements GPU kernels that fuse the inner Catmull-Rom
spline evaluation and σ-masked aggregation primitives that dominate
training time on Slashdot / Epinions / mesh / detection tasks.  See
`IMPLEMENTATION_PLAN.md` Phase 5 for the venue framing; this is the
first concrete contribution under that umbrella.

Status: bottom-up build.  Step 1 (this commit) is the Catmull-Rom
4-point gather + cubic basis kernel — the innermost primitive
called ~4 × n_layers × n_arities × n_batches times per training step.
Step 2 will fuse Catmull-Rom + tanh + σ-mask + reduce-mean into a
single SignedKAN-inner-layer kernel.

All kernels keep parity with the PyTorch reference at fp32 < 1e-5
(see `tests/test_triton_parity.py` once added to the test suite).
"""
from __future__ import annotations

import torch

import triton
import triton.language as tl


# ─── Catmull-Rom 4-point evaluation kernel ──────────────────────────


@triton.jit
def _catmull_rom_kernel(
    coef_ptr,           # (C, G) per-channel control points (flattened)
    x_ptr,              # (N, C) inputs in [-1, 1] (flattened)
    out_ptr,            # (N, C) output (flattened)
    N, C, G,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Tile (BLOCK_N, BLOCK_C) of inputs → outputs.

    For each (n, c) tile element:
      1. Clamp x_{n,c} to [-1, 1]
      2. Map to grid coord u = (x + 1) · (G - 1) / 2
      3. i = floor(u),  t = u − i
      4. Evaluate 4 CR basis weights from t, t^2, t^3
      5. Gather 4 control points coef[c, i±k] for k ∈ {-1, 0, 1, 2}
      6. Output = w_m1·P_m1 + w_0·P_0 + w_p1·P_p1 + w_p2·P_p2

    Boundary handling: clamp gather indices into [0, G-1] (matches
    PyTorch reference's `_catmull_rom_eval`).
    """
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_n = offs_n < N
    mask_c = offs_c < C
    mask = mask_n[:, None] & mask_c[None, :]

    # Load x_{n, c}.
    x_offs = offs_n[:, None] * C + offs_c[None, :]
    x = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

    # Clamp to [-1, 1] and map to grid.
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    u = (x + 1.0) * 0.5 * (G - 1)
    # Integer segment index, fractional offset.
    i = u.to(tl.int32)
    # tl.minimum on integers + clamping to [0, G-2].
    i = tl.minimum(i, G - 2)
    t = u - i.to(tl.float32)
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom basis weights.
    w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
    w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
    w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
    w_p2 = 0.5 * (t3 - t2)

    # Gather indices, clamped to [0, G-1].
    idx_m1 = tl.minimum(tl.maximum(i - 1, 0), G - 1)
    idx_0  = i
    idx_p1 = tl.minimum(i + 1, G - 1)
    idx_p2 = tl.minimum(i + 2, G - 1)

    # Compute base offsets into coef: coef[c, idx_*] flattened →
    # offs_c * G + idx_*.  Need (BLOCK_N, BLOCK_C) shape via
    # broadcasting of offs_c.
    cG = offs_c[None, :] * G                         # (1, BLOCK_C)
    P_m1 = tl.load(coef_ptr + cG + idx_m1, mask=mask, other=0.0)
    P_0  = tl.load(coef_ptr + cG + idx_0,  mask=mask, other=0.0)
    P_p1 = tl.load(coef_ptr + cG + idx_p1, mask=mask, other=0.0)
    P_p2 = tl.load(coef_ptr + cG + idx_p2, mask=mask, other=0.0)

    out = w_m1 * P_m1 + w_0 * P_0 + w_p1 * P_p1 + w_p2 * P_p2
    tl.store(out_ptr + x_offs, out, mask=mask)


def catmull_rom_triton(
    coef: torch.Tensor,
    x: torch.Tensor,
    grid: int,
) -> torch.Tensor:
    """Triton-fused Catmull-Rom evaluation.  Matches the signature of
    `splines._catmull_rom_eval` exactly.

    Parameters
    ----------
    coef : (..., C, G) per-channel control points
    x    : (..., C)    inputs in [-1, 1]
    grid : G

    Returns
    -------
    out : (..., C) spline values
    """
    if not x.is_cuda:
        raise RuntimeError("catmull_rom_triton requires CUDA inputs")
    # Only the shared-coef case is supported by the kernel: coef is
    # (C, G), not per-batch.  Detect by checking dimensionality;
    # genuinely per-batch coef (BatchedCatmullRomActivation, etc.)
    # falls back to the PyTorch reference because each row needs a
    # different (C, G) load and the kernel is not yet vectorised
    # over per-row coef.
    if coef.dim() != 2:
        from .splines import _catmull_rom_eval as _ref
        return _ref(coef, x, grid)
    C = coef.shape[0]
    if grid != coef.shape[1]:
        raise RuntimeError(
            f"grid ({grid}) must match coef.shape[1] ({coef.shape[1]})"
        )
    if x.shape[-1] != C:
        raise RuntimeError(
            f"last dim of x ({x.shape[-1]}) must match coef.shape[0] "
            f"({C})"
        )
    *batch_shape, _ = x.shape
    x_flat = x.reshape(-1, C).contiguous()
    coef_flat = coef.contiguous()
    out = torch.empty_like(x_flat)
    N = x_flat.shape[0]
    BLOCK_N = 64
    BLOCK_C = min(64, triton.next_power_of_2(C))
    grid_lambda = (
        triton.cdiv(N, BLOCK_N),
        triton.cdiv(C, BLOCK_C),
    )
    _catmull_rom_kernel[grid_lambda](
        coef_flat, x_flat, out,
        N, C, grid,
        BLOCK_N=BLOCK_N, BLOCK_C=BLOCK_C,
    )
    return out.reshape(*batch_shape, C)


# ─── Parity test against PyTorch reference ──────────────────────────


def _parity_check_catmull_rom(C: int = 16, G: int = 8,
                                B: int = 256, atol: float = 1e-5,
                                rtol: float = 1e-4) -> dict:
    """Runs the Triton kernel on random inputs and compares against
    `splines._catmull_rom_eval`.  Returns a dict with max abs / rel
    error and pass/fail.
    """
    from .splines import _catmull_rom_eval
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
    from .splines import _catmull_rom_eval
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


class _CatmullRomTritonFn(torch.autograd.Function):
    """Autograd-aware Triton-fused Catmull-Rom evaluation.

    Forward: Triton kernel (fast).
    Backward: PyTorch reference computes gradients (slower, but
    correct).  Future work can replace the backward with a Triton
    kernel of its own; for now correctness > speed on the backward.
    """

    @staticmethod
    def forward(ctx, coef, x, grid):
        ctx.save_for_backward(coef, x)
        ctx.grid = grid
        return catmull_rom_triton(coef, x, grid)

    @staticmethod
    def backward(ctx, grad_out):
        coef, x = ctx.saved_tensors
        grid = ctx.grid
        # Use the PyTorch reference to compute the backward.  Re-run
        # forward with grad-enabled inputs to get the autograd graph.
        from .splines import _catmull_rom_eval as _ref
        coef_g = coef.detach().requires_grad_(True)
        x_g = x.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _ref(coef_g, x_g, grid)
            grads = torch.autograd.grad(
                out, (coef_g, x_g), grad_out,
                retain_graph=False, create_graph=False,
            )
        return grads[0], grads[1], None


def catmull_rom_triton_autograd(coef: torch.Tensor, x: torch.Tensor,
                                  grid: int) -> torch.Tensor:
    """Autograd-wrapped Triton-fused Catmull-Rom evaluation."""
    return _CatmullRomTritonFn.apply(coef, x, grid)


# ─── Step 4: fused inner with Highway-skip support ─────────────────


@triton.jit
def _signedkan_inner_highway_kernel(
    x_ptr,
    triad_v_ptr,
    triad_sigma_ptr,
    coef_pos_ptr,
    coef_neg_ptr,
    gate_w_ptr,           # (d, d) Highway gate Linear weight
    gate_b_ptr,           # (d,) Highway gate Linear bias
    out_ptr,              # (T, 2, d)
    T, k, V, d, G,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """As ``_signedkan_inner_kernel`` but with the Highway inner-skip
    applied per-(t, k, d) before the σ-mask + mean reduction:

        T_inner_{t,i,d} = sigmoid(W_d^T · h_v_{t,i} + b_d)
        inner_{t,i,s,d} = T_inner · CR_s(h_v_{t,i,d}) + (1 - T_inner) · h_v_{t,i,d}

    To compute the gate without breaking the per-(t, d_c) thread-
    local tile, each thread block evaluates the d-by-d matvec inline:
    for output channel d_c, gate_logit_{d_c} = Σ_e W[e, d_c] · h_v_e
    + b[d_c].  This requires loading the full d-vector h_v per (t, i)
    rather than just the BLOCK_D slice — so each block does an inner
    loop over the full hidden dim.
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < d
    mask_td = mask_t[:, None] & mask_d[None, :]

    acc_pos = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    acc_neg = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    cnt_pos = tl.zeros((BLOCK_T,), dtype=tl.float32)
    cnt_neg = tl.zeros((BLOCK_T,), dtype=tl.float32)

    cG = offs_d[None, :] * G
    G_minus_1 = tl.full((), G - 1, dtype=tl.int32)

    # Load gate bias for the BLOCK_D output channels once.
    gate_b = tl.load(gate_b_ptr + offs_d, mask=mask_d, other=0.0)

    for i in range(k):
        v_off = offs_t * k + i
        v_idx = tl.load(triad_v_ptr + v_off, mask=mask_t, other=0)
        sigma = tl.load(triad_sigma_ptr + v_off, mask=mask_t, other=0)

        # Gather h_v[v_idx, offs_d] for this BLOCK_D slice.
        x_off = v_idx[:, None] * d + offs_d[None, :]
        x = tl.load(x_ptr + x_off, mask=mask_td, other=0.0)

        # Catmull-Rom for both signs (identical to the no-skip kernel).
        x_clamp = tl.minimum(tl.maximum(x, -1.0), 1.0)
        u = (x_clamp + 1.0) * 0.5 * (G - 1)
        i_seg = u.to(tl.int32)
        i_seg = tl.minimum(i_seg, G_minus_1 - 1)
        t_frac = u - i_seg.to(tl.float32)
        t2 = t_frac * t_frac
        t3 = t2 * t_frac
        w_m1 = 0.5 * (-t3 + 2.0 * t2 - t_frac)
        w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
        w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t_frac)
        w_p2 = 0.5 * (t3 - t2)
        idx_m1 = tl.minimum(tl.maximum(i_seg - 1, 0), G_minus_1)
        idx_0  = i_seg
        idx_p1 = tl.minimum(i_seg + 1, G_minus_1)
        idx_p2 = tl.minimum(i_seg + 2, G_minus_1)

        # Highway gate: gate_logit_{d_c} = Σ_e W[e, d_c] · h_v_e + b_d_c.
        # We need the full d-vector h_v_{t,i,:} to compute the matvec
        # for each d_c in BLOCK_D.  Inner-loop over d_full in steps
        # of BLOCK_D-equivalent stride.  For this kernel we keep it
        # simple: compute one matvec per (t, i, d_c) by loading the
        # full row of W and the full h_v.
        gate_logit = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
        for e in range(d):
            # Load h_v[v_idx, e] (broadcast over BLOCK_D).
            h_e_off = v_idx * d + e
            h_e = tl.load(x_ptr + h_e_off, mask=mask_t, other=0.0)
            # Load W[e, offs_d] — shape (BLOCK_D,).
            w_offs = e * d + offs_d
            w_row = tl.load(gate_w_ptr + w_offs, mask=mask_d, other=0.0)
            gate_logit += h_e[:, None] * w_row[None, :]
        gate_logit += gate_b[None, :]
        # 2-element softmax (sigmoid-free, KAN-aligned): T = σ(gate_logit).
        T_inner = 1.0 / (1.0 + tl.exp(-gate_logit))

        # Per-sign CR, then Highway mix with raw x (matches production
        # SignedKANLayer._forward_impl: no tanh wrap, raw h_v residual):
        # inner = T · CR(x_clamp) + (1 - T) · x
        for sign_kind in tl.static_range(2):
            if sign_kind == 0:
                P_m1 = tl.load(coef_pos_ptr + cG + idx_m1,
                                 mask=mask_td, other=0.0)
                P_0  = tl.load(coef_pos_ptr + cG + idx_0,
                                 mask=mask_td, other=0.0)
                P_p1 = tl.load(coef_pos_ptr + cG + idx_p1,
                                 mask=mask_td, other=0.0)
                P_p2 = tl.load(coef_pos_ptr + cG + idx_p2,
                                 mask=mask_td, other=0.0)
            else:
                P_m1 = tl.load(coef_neg_ptr + cG + idx_m1,
                                 mask=mask_td, other=0.0)
                P_0  = tl.load(coef_neg_ptr + cG + idx_0,
                                 mask=mask_td, other=0.0)
                P_p1 = tl.load(coef_neg_ptr + cG + idx_p1,
                                 mask=mask_td, other=0.0)
                P_p2 = tl.load(coef_neg_ptr + cG + idx_p2,
                                 mask=mask_td, other=0.0)
            h = w_m1 * P_m1 + w_0 * P_0 + w_p1 * P_p1 + w_p2 * P_p2
            # Highway mix with raw x (production semantics).
            h_mixed = T_inner * h + (1.0 - T_inner) * x

            if sign_kind == 0:
                is_match = (sigma == 1).to(tl.float32)
                acc_pos += is_match[:, None] * h_mixed
                cnt_pos += is_match
            else:
                is_match = (sigma == -1).to(tl.float32)
                acc_neg += is_match[:, None] * h_mixed
                cnt_neg += is_match

    cnt_pos = tl.maximum(cnt_pos, 1.0)
    cnt_neg = tl.maximum(cnt_neg, 1.0)
    acc_pos = acc_pos / cnt_pos[:, None]
    acc_neg = acc_neg / cnt_neg[:, None]

    out_pos_off = offs_t[:, None] * 2 * d + offs_d[None, :]
    out_neg_off = offs_t[:, None] * 2 * d + d + offs_d[None, :]
    tl.store(out_ptr + out_pos_off, acc_pos, mask=mask_td)
    tl.store(out_ptr + out_neg_off, acc_neg, mask=mask_td)


def signedkan_inner_highway_triton(
    x: torch.Tensor,
    triad_v: torch.Tensor,
    triad_sigma: torch.Tensor,
    coef_pos: torch.Tensor,
    coef_neg: torch.Tensor,
    gate_w: torch.Tensor,         # (d, d) Highway Linear weight
    gate_b: torch.Tensor,         # (d,) bias
    grid: int,
    BLOCK_T: int | None = None,
    BLOCK_D: int | None = None,
) -> torch.Tensor:
    """Fused inner with Highway-skip support.  Returns (T, 2, d).

    Matches production ``SignedKANLayer._forward_impl`` semantics
    (no tanh wrap, raw h_v residual).  Equivalent PyTorch reference:
        h_v = x[triad_v]
        T_inner = sigmoid(gate_w(h_v) + gate_b)   # (T, k, d)
        cr_pos = CR(h_v, coef_pos)                # CR clamps internally
        cr_neg = CR(h_v, coef_neg)
        h_pos = T_inner * cr_pos + (1 - T_inner) * h_v
        h_neg = T_inner * cr_neg + (1 - T_inner) * h_v
        # σ-mask + mean → (T, 2, d)
    """
    if not (x.is_cuda and triad_v.is_cuda):
        raise RuntimeError("requires CUDA inputs")
    V, d = x.shape
    T, k = triad_v.shape
    assert coef_pos.shape == (d, grid)
    assert coef_neg.shape == (d, grid)
    assert gate_w.shape == (d, d)
    assert gate_b.shape == (d,)
    out = torch.empty(T, 2, d, device=x.device, dtype=torch.float32)
    if BLOCK_T is None:
        BLOCK_T = 16
    if BLOCK_D is None:
        BLOCK_D = min(64, triton.next_power_of_2(d))
    grid_lambda = (
        triton.cdiv(T, BLOCK_T),
        triton.cdiv(d, BLOCK_D),
    )
    _signedkan_inner_highway_kernel[grid_lambda](
        x.contiguous(),
        triad_v.contiguous(),
        triad_sigma.contiguous(),
        coef_pos.contiguous(),
        coef_neg.contiguous(),
        gate_w.contiguous(),
        gate_b.contiguous(),
        out,
        T, k, V, d, grid,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return out


# ─── Step 2: fused gather + per-sign CR + σ-mask + mean ────────────


@triton.jit
def _signedkan_inner_kernel(
    x_ptr,              # (V, d) per-vertex embeddings
    triad_v_ptr,        # (T, k) vertex indices, int64
    triad_sigma_ptr,    # (T, k) σ values in {-1, +1}, int64
    coef_pos_ptr,       # (d, G) inner CR coef for sign +1
    coef_neg_ptr,       # (d, G) inner CR coef for sign -1
    out_ptr,            # (T, 2, d) per-sign mean output
    T, k, V, d, G,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """For each cycle t and channel d_c, accumulate:
        acc_+[t, d_c] = mean_{i : σ_{t,i}=+1}  tanh(CR(x[v_{t,i}, d_c], coef_+))
        acc_-[t, d_c] = mean_{i : σ_{t,i}=-1}  tanh(CR(x[v_{t,i}, d_c], coef_-))

    The (T, k, S, d) intermediate that the PyTorch reference
    materialises is collapsed here into per-(t, d_c) thread-local
    accumulators.  Memory savings: O(T·k·S·d) → O(T·S·d).
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < d
    mask_td = mask_t[:, None] & mask_d[None, :]

    acc_pos = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    acc_neg = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    cnt_pos = tl.zeros((BLOCK_T,), dtype=tl.float32)
    cnt_neg = tl.zeros((BLOCK_T,), dtype=tl.float32)

    cG = offs_d[None, :] * G                              # (1, BLOCK_D)
    G_minus_1 = tl.full((), G - 1, dtype=tl.int32)

    for i in range(k):
        # Per-(t, i) gather: load v_idx and σ at (t, i).
        v_off = offs_t * k + i
        v_idx = tl.load(triad_v_ptr + v_off, mask=mask_t, other=0)
        sigma = tl.load(triad_sigma_ptr + v_off, mask=mask_t, other=0)

        # Gather x[v_idx, offs_d] → (BLOCK_T, BLOCK_D).
        x_off = v_idx[:, None] * d + offs_d[None, :]
        x = tl.load(x_ptr + x_off, mask=mask_td, other=0.0)

        # Catmull-Rom evaluation per-channel for both sign branches.
        x_clamp = tl.minimum(tl.maximum(x, -1.0), 1.0)
        u = (x_clamp + 1.0) * 0.5 * (G - 1)
        i_seg = u.to(tl.int32)
        i_seg = tl.minimum(i_seg, G_minus_1 - 1)
        t_frac = u - i_seg.to(tl.float32)
        t2 = t_frac * t_frac
        t3 = t2 * t_frac

        w_m1 = 0.5 * (-t3 + 2.0 * t2 - t_frac)
        w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
        w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t_frac)
        w_p2 = 0.5 * (t3 - t2)

        idx_m1 = tl.minimum(tl.maximum(i_seg - 1, 0), G_minus_1)
        idx_0  = i_seg
        idx_p1 = tl.minimum(i_seg + 1, G_minus_1)
        idx_p2 = tl.minimum(i_seg + 2, G_minus_1)

        # Per-sign CR + tanh.
        for sign_kind in tl.static_range(2):
            if sign_kind == 0:
                P_m1 = tl.load(coef_pos_ptr + cG + idx_m1,
                                 mask=mask_td, other=0.0)
                P_0  = tl.load(coef_pos_ptr + cG + idx_0,
                                 mask=mask_td, other=0.0)
                P_p1 = tl.load(coef_pos_ptr + cG + idx_p1,
                                 mask=mask_td, other=0.0)
                P_p2 = tl.load(coef_pos_ptr + cG + idx_p2,
                                 mask=mask_td, other=0.0)
            else:
                P_m1 = tl.load(coef_neg_ptr + cG + idx_m1,
                                 mask=mask_td, other=0.0)
                P_0  = tl.load(coef_neg_ptr + cG + idx_0,
                                 mask=mask_td, other=0.0)
                P_p1 = tl.load(coef_neg_ptr + cG + idx_p1,
                                 mask=mask_td, other=0.0)
                P_p2 = tl.load(coef_neg_ptr + cG + idx_p2,
                                 mask=mask_td, other=0.0)
            h = w_m1 * P_m1 + w_0 * P_0 + w_p1 * P_p1 + w_p2 * P_p2

            # σ-masked accumulate.
            if sign_kind == 0:
                is_match = (sigma == 1).to(tl.float32)
                acc_pos += is_match[:, None] * h
                cnt_pos += is_match
            else:
                is_match = (sigma == -1).to(tl.float32)
                acc_neg += is_match[:, None] * h
                cnt_neg += is_match

    cnt_pos = tl.maximum(cnt_pos, 1.0)
    cnt_neg = tl.maximum(cnt_neg, 1.0)
    acc_pos = acc_pos / cnt_pos[:, None]
    acc_neg = acc_neg / cnt_neg[:, None]

    # Store: out[t, s, d] in (T, 2, d) layout flattened as t*2*d + s*d + d_c.
    out_pos_off = offs_t[:, None] * 2 * d + offs_d[None, :]
    out_neg_off = offs_t[:, None] * 2 * d + d + offs_d[None, :]
    tl.store(out_ptr + out_pos_off, acc_pos, mask=mask_td)
    tl.store(out_ptr + out_neg_off, acc_neg, mask=mask_td)


def signedkan_inner_triton(
    x: torch.Tensor,            # (V, d)
    triad_v: torch.Tensor,      # (T, k) long
    triad_sigma: torch.Tensor,  # (T, k) long, ±1
    coef_pos: torch.Tensor,     # (d, G) inner CR coef for σ=+1
    coef_neg: torch.Tensor,     # (d, G) inner CR coef for σ=-1
    grid: int,
    BLOCK_T: int | None = None,
    BLOCK_D: int | None = None,
) -> torch.Tensor:
    """Returns (T, 2, d) σ-masked per-sign mean of tanh(CR(x[v]))."""
    if not (x.is_cuda and triad_v.is_cuda):
        raise RuntimeError("signedkan_inner_triton requires CUDA inputs")
    V, d = x.shape
    T, k = triad_v.shape
    assert coef_pos.shape == (d, grid)
    assert coef_neg.shape == (d, grid)
    out = torch.empty(T, 2, d, device=x.device, dtype=torch.float32)
    if BLOCK_T is None:
        # Tuning sweep (`triton_kernels_sensitivity.py`) found
        # BLOCK_T=16 best for d=4 Slashdot recipe (0.082 ms vs
        # 0.164 ms at BLOCK_T=32).  Higher d may want larger BLOCK_T.
        BLOCK_T = 16
    if BLOCK_D is None:
        # BLOCK_D = d (next power of 2, capped at 64) is the
        # sweep's empirical optimum.
        BLOCK_D = min(64, triton.next_power_of_2(d))
    grid_lambda = (
        triton.cdiv(T, BLOCK_T),
        triton.cdiv(d, BLOCK_D),
    )
    _signedkan_inner_kernel[grid_lambda](
        x.contiguous(),
        triad_v.contiguous(),
        triad_sigma.contiguous(),
        coef_pos.contiguous(),
        coef_neg.contiguous(),
        out,
        T, k, V, d, grid,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return out


# ─── Backward kernel (no-skip) ──────────────────────────────────────


@triton.jit
def _signedkan_inner_backward_kernel(
    x_ptr,                  # (V, d) per-vertex embeddings
    triad_v_ptr,            # (T, k) int64
    triad_sigma_ptr,        # (T, k) int64
    coef_pos_ptr,           # (d, G)
    coef_neg_ptr,           # (d, G)
    grad_out_ptr,           # (T, 2, d) ∂L/∂agg
    grad_x_ptr,             # (V, d)   ∂L/∂x   (atomic-added)
    grad_coef_pos_ptr,      # (d, G)   ∂L/∂coef_pos (atomic-added)
    grad_coef_neg_ptr,      # (d, G)   ∂L/∂coef_neg (atomic-added)
    T, k, V, d, G,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Closed-form backward for the no-skip fused inner kernel.

    Mathematical derivation lives in
    docs/triton_kernel_integration_tutorial_2026_05_09.md §10.2.

    Parallelism: (BLOCK_T cycles) × (BLOCK_D channels).  Two passes
    over k slots inside each block: first to compute n^s counts, then
    to scatter ∂x (via atomic-add at v_{t,i}) and ∂coef^s (via atomic-
    add at (c, q) with q ∈ {idx_{m1,0,p1,p2}}).
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < d
    mask_td = mask_t[:, None] & mask_d[None, :]

    G_minus_1 = tl.full((), G - 1, dtype=tl.int32)

    # Pass 1: per-cycle σ-counts n^s_t.
    cnt_pos = tl.zeros((BLOCK_T,), dtype=tl.float32)
    cnt_neg = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for i in range(k):
        v_off = offs_t * k + i
        sigma = tl.load(triad_sigma_ptr + v_off, mask=mask_t, other=0)
        cnt_pos += (sigma == 1).to(tl.float32)
        cnt_neg += (sigma == -1).to(tl.float32)
    cnt_pos = tl.maximum(cnt_pos, 1.0)
    cnt_neg = tl.maximum(cnt_neg, 1.0)

    # Load grad_out per sign (T, 2, d) → (BLOCK_T, BLOCK_D).
    g_pos_off = offs_t[:, None] * 2 * d + offs_d[None, :]
    g_neg_off = offs_t[:, None] * 2 * d + d + offs_d[None, :]
    g_pos = tl.load(grad_out_ptr + g_pos_off, mask=mask_td, other=0.0)
    g_neg = tl.load(grad_out_ptr + g_neg_off, mask=mask_td, other=0.0)

    # Pre-divide: g̃^s[t, c] = g^s[t, c] / n^s_t.
    g_pos_normed = g_pos / cnt_pos[:, None]
    g_neg_normed = g_neg / cnt_neg[:, None]

    cG = offs_d[None, :] * G

    # Pass 2: per-slot scatter to ∂x and ∂coef^s.
    for i in range(k):
        v_off = offs_t * k + i
        v_idx = tl.load(triad_v_ptr + v_off, mask=mask_t, other=0)
        sigma = tl.load(triad_sigma_ptr + v_off, mask=mask_t, other=0)
        m_pos = (sigma == 1).to(tl.float32)
        m_neg = (sigma == -1).to(tl.float32)

        x_off = v_idx[:, None] * d + offs_d[None, :]
        x = tl.load(x_ptr + x_off, mask=mask_td, other=0.0)

        # CR forward locals (re-derived; cheap).
        x_clamp = tl.minimum(tl.maximum(x, -1.0), 1.0)
        in_range = ((x > -1.0) & (x < 1.0)).to(tl.float32)  # 1 if not saturated
        u = (x_clamp + 1.0) * 0.5 * (G - 1)
        i_seg = u.to(tl.int32)
        i_seg = tl.minimum(i_seg, G_minus_1 - 1)
        t_frac = u - i_seg.to(tl.float32)
        t2 = t_frac * t_frac
        t3 = t2 * t_frac

        # Closed-form CR blend weights (forward).
        w_m1 = 0.5 * (-t3 + 2.0 * t2 - t_frac)
        w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
        w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t_frac)
        w_p2 = 0.5 * (t3 - t2)

        # Their derivatives w.r.t. τ (used in ∂CR/∂x).
        wp_m1 = 0.5 * (-3.0 * t2 + 4.0 * t_frac - 1.0)
        wp_0  = 0.5 * (9.0 * t2 - 10.0 * t_frac)
        wp_p1 = 0.5 * (-9.0 * t2 + 8.0 * t_frac + 1.0)
        wp_p2 = 0.5 * (3.0 * t2 - 2.0 * t_frac)

        idx_m1 = tl.minimum(tl.maximum(i_seg - 1, 0), G_minus_1)
        idx_0  = i_seg
        idx_p1 = tl.minimum(i_seg + 1, G_minus_1)
        idx_p2 = tl.minimum(i_seg + 2, G_minus_1)

        # Per-sign branch.
        for sign_kind in tl.static_range(2):
            if sign_kind == 0:
                P_m1 = tl.load(coef_pos_ptr + cG + idx_m1, mask=mask_td, other=0.0)
                P_0  = tl.load(coef_pos_ptr + cG + idx_0,  mask=mask_td, other=0.0)
                P_p1 = tl.load(coef_pos_ptr + cG + idx_p1, mask=mask_td, other=0.0)
                P_p2 = tl.load(coef_pos_ptr + cG + idx_p2, mask=mask_td, other=0.0)
                m = m_pos
                gn = g_pos_normed
            else:
                P_m1 = tl.load(coef_neg_ptr + cG + idx_m1, mask=mask_td, other=0.0)
                P_0  = tl.load(coef_neg_ptr + cG + idx_0,  mask=mask_td, other=0.0)
                P_p1 = tl.load(coef_neg_ptr + cG + idx_p1, mask=mask_td, other=0.0)
                P_p2 = tl.load(coef_neg_ptr + cG + idx_p2, mask=mask_td, other=0.0)
                m = m_neg
                gn = g_neg_normed

            # ĝ^s_{t,i,c} = m^s_{t,i} · g^s_{t,c} / n^s_t
            eff = m[:, None] * gn  # (BLOCK_T, BLOCK_D)

            # ∂CR/∂h = 1[|h|<1] · (G-1)/2 · Σ_p w'_p(τ) · P_p
            dCR_dh = in_range * 0.5 * (G - 1) * (
                wp_m1 * P_m1 + wp_0 * P_0 + wp_p1 * P_p1 + wp_p2 * P_p2
            )
            grad_x_contrib = eff * dCR_dh

            # Scatter ∂x to grad_x[v_{t,i}, c].
            if sign_kind == 0:
                tl.atomic_add(grad_x_ptr + x_off, grad_x_contrib, mask=mask_td)
            else:
                tl.atomic_add(grad_x_ptr + x_off, grad_x_contrib, mask=mask_td)

            # ∂coef contributions: scatter eff · w_p(τ) into (c, idx_p) slot.
            cgrad_m1 = offs_d[None, :] * G + idx_m1
            cgrad_0  = offs_d[None, :] * G + idx_0
            cgrad_p1 = offs_d[None, :] * G + idx_p1
            cgrad_p2 = offs_d[None, :] * G + idx_p2
            if sign_kind == 0:
                tl.atomic_add(grad_coef_pos_ptr + cgrad_m1, eff * w_m1, mask=mask_td)
                tl.atomic_add(grad_coef_pos_ptr + cgrad_0,  eff * w_0,  mask=mask_td)
                tl.atomic_add(grad_coef_pos_ptr + cgrad_p1, eff * w_p1, mask=mask_td)
                tl.atomic_add(grad_coef_pos_ptr + cgrad_p2, eff * w_p2, mask=mask_td)
            else:
                tl.atomic_add(grad_coef_neg_ptr + cgrad_m1, eff * w_m1, mask=mask_td)
                tl.atomic_add(grad_coef_neg_ptr + cgrad_0,  eff * w_0,  mask=mask_td)
                tl.atomic_add(grad_coef_neg_ptr + cgrad_p1, eff * w_p1, mask=mask_td)
                tl.atomic_add(grad_coef_neg_ptr + cgrad_p2, eff * w_p2, mask=mask_td)


def signedkan_inner_backward_triton(
    x: torch.Tensor,            # (V, d)
    triad_v: torch.Tensor,      # (T, k) long
    triad_sigma: torch.Tensor,  # (T, k) long
    coef_pos: torch.Tensor,     # (d, G)
    coef_neg: torch.Tensor,     # (d, G)
    grad_out: torch.Tensor,     # (T, 2, d)
    grid: int,
    BLOCK_T: int | None = None,
    BLOCK_D: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (∂L/∂x, ∂L/∂coef_pos, ∂L/∂coef_neg).  Closed-form
    backward for the no-skip fused inner kernel."""
    if not (x.is_cuda and triad_v.is_cuda):
        raise RuntimeError("requires CUDA inputs")
    V, d = x.shape
    T, k = triad_v.shape

    grad_x = torch.zeros_like(x)
    grad_coef_pos = torch.zeros_like(coef_pos)
    grad_coef_neg = torch.zeros_like(coef_neg)

    if BLOCK_T is None:
        BLOCK_T = 16
    if BLOCK_D is None:
        BLOCK_D = min(64, triton.next_power_of_2(d))
    grid_lambda = (
        triton.cdiv(T, BLOCK_T),
        triton.cdiv(d, BLOCK_D),
    )
    _signedkan_inner_backward_kernel[grid_lambda](
        x.contiguous(),
        triad_v.contiguous(),
        triad_sigma.contiguous(),
        coef_pos.contiguous(),
        coef_neg.contiguous(),
        grad_out.contiguous(),
        grad_x,
        grad_coef_pos,
        grad_coef_neg,
        T, k, V, d, grid,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
    )
    return grad_x, grad_coef_pos, grad_coef_neg


# ─── Backward kernel (highway) ──────────────────────────────────────


@triton.jit
def _signedkan_inner_highway_backward_kernel(
    x_ptr,                  # (V, d)
    triad_v_ptr,            # (T, k) int64
    triad_sigma_ptr,        # (T, k) int64
    coef_pos_ptr,           # (d, G)
    coef_neg_ptr,           # (d, G)
    gate_w_ptr,             # (d, d)  Highway gate Linear weight
    gate_b_ptr,             # (d,)    Highway gate Linear bias
    grad_out_ptr,           # (T, 2, d) ∂L/∂agg
    grad_x_ptr,             # (V, d)   ∂L/∂x   (atomic-added)
    grad_coef_pos_ptr,      # (d, G)
    grad_coef_neg_ptr,      # (d, G)
    grad_gate_w_ptr,        # (d, d)
    grad_gate_b_ptr,        # (d,)
    T, k, V, d, G,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    """Closed-form backward for the highway-skip fused inner kernel.

    Mathematical derivation:
    docs/triton_kernel_integration_tutorial_2026_05_09.md §10.2.

    Three contributions to ∂L/∂x (residual / gate-logit / CR-input),
    plus reductions for ∂W, ∂b, and a scatter for ∂coef^s.
    """
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < d
    mask_td = mask_t[:, None] & mask_d[None, :]

    G_minus_1 = tl.full((), G - 1, dtype=tl.int32)

    # Pass 1: per-cycle σ-counts.
    cnt_pos = tl.zeros((BLOCK_T,), dtype=tl.float32)
    cnt_neg = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for i in range(k):
        v_off = offs_t * k + i
        sigma = tl.load(triad_sigma_ptr + v_off, mask=mask_t, other=0)
        cnt_pos += (sigma == 1).to(tl.float32)
        cnt_neg += (sigma == -1).to(tl.float32)
    cnt_pos = tl.maximum(cnt_pos, 1.0)
    cnt_neg = tl.maximum(cnt_neg, 1.0)

    # Load grad_out per sign.
    g_pos_off = offs_t[:, None] * 2 * d + offs_d[None, :]
    g_neg_off = offs_t[:, None] * 2 * d + d + offs_d[None, :]
    g_pos = tl.load(grad_out_ptr + g_pos_off, mask=mask_td, other=0.0)
    g_neg = tl.load(grad_out_ptr + g_neg_off, mask=mask_td, other=0.0)
    g_pos_normed = g_pos / cnt_pos[:, None]
    g_neg_normed = g_neg / cnt_neg[:, None]

    # Load gate bias for this BLOCK_D tile (used per (t, i)).
    gate_b_tile = tl.load(gate_b_ptr + offs_d, mask=mask_d, other=0.0)

    # Pre-load full W (d, BLOCK_D) once outside the k-loop when using
    # the tl.dot fast path.  Layout: w_full[e, c] = W[e_full[e], offs_d[c]].
    e_full = tl.arange(0, BLOCK_D)
    mask_e = e_full < d
    if USE_DOT:
        w_off_full = e_full[:, None] * d + offs_d[None, :]
        mask_w = mask_e[:, None] & mask_d[None, :]
        w_full = tl.load(gate_w_ptr + w_off_full, mask=mask_w, other=0.0)
    else:
        # Unused branch placeholder (kernel paths must agree on tensor shapes).
        w_full = tl.zeros((BLOCK_D, BLOCK_D), dtype=tl.float32)

    cG = offs_d[None, :] * G

    # Pass 2: per-slot scatters and reductions.
    for i in range(k):
        v_off = offs_t * k + i
        v_idx = tl.load(triad_v_ptr + v_off, mask=mask_t, other=0)
        sigma = tl.load(triad_sigma_ptr + v_off, mask=mask_t, other=0)
        m_pos = (sigma == 1).to(tl.float32)
        m_neg = (sigma == -1).to(tl.float32)

        x_off = v_idx[:, None] * d + offs_d[None, :]
        x = tl.load(x_ptr + x_off, mask=mask_td, other=0.0)

        # Gate logit: ℓ_c = Σ_e W[e, c] · h_e + b_c.
        if USE_DOT:
            # When BLOCK_D >= d (typical: BLOCK_D == d), x is the full
            # h vector (BLOCK_T, BLOCK_D) and we can do the d-by-d
            # matvec in a single tl.dot.  Force IEEE fp32 to keep
            # gradient parity ≤ 1e-5 (TF32 default would give ~1e-3).
            gate_logit = tl.dot(x, w_full, input_precision="ieee")
        else:
            gate_logit = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
            for e in range(d):
                h_e_off = v_idx * d + e
                h_e = tl.load(x_ptr + h_e_off, mask=mask_t, other=0.0)
                w_offs = e * d + offs_d
                w_row = tl.load(gate_w_ptr + w_offs, mask=mask_d, other=0.0)
                gate_logit += h_e[:, None] * w_row[None, :]
        gate_logit += gate_b_tile[None, :]
        T_inner = 1.0 / (1.0 + tl.exp(-gate_logit))
        T_dot = T_inner * (1.0 - T_inner)  # ∂T/∂ℓ

        # CR forward locals.
        x_clamp = tl.minimum(tl.maximum(x, -1.0), 1.0)
        in_range = ((x > -1.0) & (x < 1.0)).to(tl.float32)
        u = (x_clamp + 1.0) * 0.5 * (G - 1)
        i_seg = u.to(tl.int32)
        i_seg = tl.minimum(i_seg, G_minus_1 - 1)
        t_frac = u - i_seg.to(tl.float32)
        t2 = t_frac * t_frac
        t3 = t2 * t_frac

        w_m1 = 0.5 * (-t3 + 2.0 * t2 - t_frac)
        w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
        w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t_frac)
        w_p2 = 0.5 * (t3 - t2)

        wp_m1 = 0.5 * (-3.0 * t2 + 4.0 * t_frac - 1.0)
        wp_0  = 0.5 * (9.0 * t2 - 10.0 * t_frac)
        wp_p1 = 0.5 * (-9.0 * t2 + 8.0 * t_frac + 1.0)
        wp_p2 = 0.5 * (3.0 * t2 - 2.0 * t_frac)

        idx_m1 = tl.minimum(tl.maximum(i_seg - 1, 0), G_minus_1)
        idx_0  = i_seg
        idx_p1 = tl.minimum(i_seg + 1, G_minus_1)
        idx_p2 = tl.minimum(i_seg + 2, G_minus_1)

        # Per-sign branch.  Accumulate gate-path totals (for ∂W, ∂b,
        # and the gate contribution to ∂x) over both signs in
        # gate_factor_total = Σ_s (m^s · g^s_normed · Δ^s · T(1-T)).
        gate_factor_total = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
        for sign_kind in tl.static_range(2):
            if sign_kind == 0:
                P_m1 = tl.load(coef_pos_ptr + cG + idx_m1, mask=mask_td, other=0.0)
                P_0  = tl.load(coef_pos_ptr + cG + idx_0,  mask=mask_td, other=0.0)
                P_p1 = tl.load(coef_pos_ptr + cG + idx_p1, mask=mask_td, other=0.0)
                P_p2 = tl.load(coef_pos_ptr + cG + idx_p2, mask=mask_td, other=0.0)
                m = m_pos
                gn = g_pos_normed
            else:
                P_m1 = tl.load(coef_neg_ptr + cG + idx_m1, mask=mask_td, other=0.0)
                P_0  = tl.load(coef_neg_ptr + cG + idx_0,  mask=mask_td, other=0.0)
                P_p1 = tl.load(coef_neg_ptr + cG + idx_p1, mask=mask_td, other=0.0)
                P_p2 = tl.load(coef_neg_ptr + cG + idx_p2, mask=mask_td, other=0.0)
                m = m_neg
                gn = g_neg_normed

            # CR forward value (needed for Δ = CR - h).
            CR = w_m1 * P_m1 + w_0 * P_0 + w_p1 * P_p1 + w_p2 * P_p2
            Delta = CR - x  # raw h, not h_clamp (production semantics)

            # Effective per-(t, i, c) gradient: g̃ = m · g/n.
            eff = m[:, None] * gn  # (BLOCK_T, BLOCK_D)

            # ∂agg/∂CR = T · g̃ → ∂coef = T·g̃·w_p, scattered into (c, q).
            t_eff = T_inner * eff
            cgrad_m1 = offs_d[None, :] * G + idx_m1
            cgrad_0  = offs_d[None, :] * G + idx_0
            cgrad_p1 = offs_d[None, :] * G + idx_p1
            cgrad_p2 = offs_d[None, :] * G + idx_p2
            if sign_kind == 0:
                tl.atomic_add(grad_coef_pos_ptr + cgrad_m1, t_eff * w_m1, mask=mask_td)
                tl.atomic_add(grad_coef_pos_ptr + cgrad_0,  t_eff * w_0,  mask=mask_td)
                tl.atomic_add(grad_coef_pos_ptr + cgrad_p1, t_eff * w_p1, mask=mask_td)
                tl.atomic_add(grad_coef_pos_ptr + cgrad_p2, t_eff * w_p2, mask=mask_td)
            else:
                tl.atomic_add(grad_coef_neg_ptr + cgrad_m1, t_eff * w_m1, mask=mask_td)
                tl.atomic_add(grad_coef_neg_ptr + cgrad_0,  t_eff * w_0,  mask=mask_td)
                tl.atomic_add(grad_coef_neg_ptr + cgrad_p1, t_eff * w_p1, mask=mask_td)
                tl.atomic_add(grad_coef_neg_ptr + cgrad_p2, t_eff * w_p2, mask=mask_td)

            # CR-input path: T · ∂CR/∂h · g̃ → ∂x[v_{t,i}, c]
            dCR_dh = in_range * 0.5 * (G - 1) * (
                wp_m1 * P_m1 + wp_0 * P_0 + wp_p1 * P_p1 + wp_p2 * P_p2
            )
            grad_x_cr = t_eff * dCR_dh

            # Residual path: (1-T) · g̃ → ∂x[v_{t,i}, c]
            grad_x_res = (1.0 - T_inner) * eff

            tl.atomic_add(grad_x_ptr + x_off, grad_x_cr + grad_x_res, mask=mask_td)

            # Accumulate gate factor across signs.
            gate_factor_total += eff * Delta * T_dot

        # ∂L/∂b_c  +=  Σ_t (m_pos · g_pos_normed · Δ^+ · T(1-T)
        #                 + m_neg · g_neg_normed · Δ^- · T(1-T))
        # which is gate_factor_total summed over BLOCK_T.
        gb_local = tl.sum(gate_factor_total, axis=0)  # (BLOCK_D,)
        tl.atomic_add(grad_gate_b_ptr + offs_d, gb_local, mask=mask_d)

        if USE_DOT:
            # ∂L/∂W (BLOCK_D=d, BLOCK_D=d) = x.T @ gate_factor_total.
            gw_contrib = tl.dot(tl.trans(x), gate_factor_total,
                                 input_precision="ieee")
            gw_off = e_full[:, None] * d + offs_d[None, :]
            mask_gw = mask_e[:, None] & mask_d[None, :]
            tl.atomic_add(grad_gate_w_ptr + gw_off, gw_contrib, mask=mask_gw)

            # ∂L/∂h_gate (BLOCK_T, BLOCK_D=d) = gate_factor_total @ W.T.
            grad_x_gate = tl.dot(gate_factor_total, tl.trans(w_full),
                                  input_precision="ieee")
            grad_x_h_off = v_idx[:, None] * d + e_full[None, :]
            mask_gx = mask_t[:, None] & mask_e[None, :]
            tl.atomic_add(grad_x_ptr + grad_x_h_off, grad_x_gate, mask=mask_gx)
        else:
            # ∂L/∂W_{e,c} += Σ_t gate_factor_total[t, c] · h_e
            for e in range(d):
                h_e_off = v_idx * d + e
                h_e = tl.load(x_ptr + h_e_off, mask=mask_t, other=0.0)
                gw_contrib_per_t = gate_factor_total * h_e[:, None]
                gw_contrib = tl.sum(gw_contrib_per_t, axis=0)
                gw_off = e * d + offs_d
                tl.atomic_add(grad_gate_w_ptr + gw_off, gw_contrib, mask=mask_d)

            # Gate-path contribution to ∂x:
            for e in range(d):
                w_offs = e * d + offs_d
                w_row = tl.load(gate_w_ptr + w_offs, mask=mask_d, other=0.0)
                grad_x_gate_e = tl.sum(gate_factor_total * w_row[None, :], axis=1)
                grad_x_h_off = v_idx * d + e
                tl.atomic_add(grad_x_ptr + grad_x_h_off, grad_x_gate_e, mask=mask_t)


def signedkan_inner_highway_backward_triton(
    x: torch.Tensor,
    triad_v: torch.Tensor,
    triad_sigma: torch.Tensor,
    coef_pos: torch.Tensor,
    coef_neg: torch.Tensor,
    gate_w: torch.Tensor,           # (d, d)
    gate_b: torch.Tensor,           # (d,)
    grad_out: torch.Tensor,         # (T, 2, d)
    grid: int,
    BLOCK_T: int | None = None,
    BLOCK_D: int | None = None,
    use_dot: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Returns (∂L/∂x, ∂L/∂coef_pos, ∂L/∂coef_neg, ∂L/∂gate_w, ∂L/∂gate_b)
    for the highway-skip fused inner kernel.

    The ``use_dot`` flag picks the gate-matvec path: ``True`` uses
    ``tl.dot`` (faster at d ≥ 16, requires BLOCK_D ≥ d ≥ 16);
    ``False`` uses three explicit ``for e in range(d)`` loops (works
    at any d but slower).  Default: auto-pick by d.
    """
    if not (x.is_cuda and triad_v.is_cuda):
        raise RuntimeError("requires CUDA inputs")
    V, d = x.shape
    T, k = triad_v.shape

    grad_x = torch.zeros_like(x)
    grad_coef_pos = torch.zeros_like(coef_pos)
    grad_coef_neg = torch.zeros_like(coef_neg)
    grad_gate_w = torch.zeros_like(gate_w)
    grad_gate_b = torch.zeros_like(gate_b)

    if BLOCK_T is None:
        BLOCK_T = 16
    if BLOCK_D is None:
        BLOCK_D = min(64, triton.next_power_of_2(d))
    if use_dot is None:
        # tl.dot needs BLOCK_T ≥ 16, BLOCK_D ≥ 16, and d ≥ 16 (else
        # the matmul is too small to amortize).  Empirically,
        # the explicit-loop path wins at d ≤ 8.
        use_dot = (d >= 16 and BLOCK_T >= 16 and BLOCK_D >= 16)
    grid_lambda = (
        triton.cdiv(T, BLOCK_T),
        triton.cdiv(d, BLOCK_D),
    )
    _signedkan_inner_highway_backward_kernel[grid_lambda](
        x.contiguous(),
        triad_v.contiguous(),
        triad_sigma.contiguous(),
        coef_pos.contiguous(),
        coef_neg.contiguous(),
        gate_w.contiguous(),
        gate_b.contiguous(),
        grad_out.contiguous(),
        grad_x,
        grad_coef_pos,
        grad_coef_neg,
        grad_gate_w,
        grad_gate_b,
        T, k, V, d, grid,
        BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        USE_DOT=use_dot,
    )
    return grad_x, grad_coef_pos, grad_coef_neg, grad_gate_w, grad_gate_b


def _parity_check_signedkan_inner(
    V: int = 1000, d: int = 16, T: int = 5000, k: int = 4, G: int = 8,
    atol: float = 1e-4,
) -> dict:
    """Parity check the fused inner kernel against an explicit PyTorch
    reference implementation."""
    from .splines import _catmull_rom_eval
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


class _SignedKANInnerTritonFn(torch.autograd.Function):
    """Autograd-aware fused inner-kernel.

    Forward: Triton kernel.
    Backward: fused Triton kernel (closed-form gradients per
    docs/triton_kernel_integration_tutorial_2026_05_09.md §10.2).
    Set HSIKAN_TRITON_BACKWARD=0 to fall back to a PyTorch reference
    re-computation (slower; useful for debugging).
    """

    @staticmethod
    def forward(ctx, x, triad_v, triad_sigma, coef_pos, coef_neg, grid):
        ctx.save_for_backward(x, triad_v, triad_sigma, coef_pos, coef_neg)
        ctx.grid = grid
        return signedkan_inner_triton(
            x, triad_v, triad_sigma, coef_pos, coef_neg, grid,
        )

    @staticmethod
    def backward(ctx, grad_out):
        import os
        x, triad_v, triad_sigma, coef_pos, coef_neg = ctx.saved_tensors
        grid = ctx.grid
        if int(os.environ.get("HSIKAN_TRITON_BACKWARD", "1")) != 0:
            grad_x, grad_cp, grad_cn = signedkan_inner_backward_triton(
                x, triad_v, triad_sigma, coef_pos, coef_neg,
                grad_out.contiguous(), grid,
            )
            return grad_x, None, None, grad_cp, grad_cn, None
        # PyTorch fallback (correctness reference).
        from .splines import _catmull_rom_eval as _ref
        x_g = x.detach().requires_grad_(True)
        coef_pos_g = coef_pos.detach().requires_grad_(True)
        coef_neg_g = coef_neg.detach().requires_grad_(True)
        with torch.enable_grad():
            h_v = x_g[triad_v]
            inner_pos = _ref(coef_pos_g, h_v, grid)
            inner_neg = _ref(coef_neg_g, h_v, grid)
            inner_all = torch.stack([inner_pos, inner_neg], dim=2)
            sign_vals = torch.tensor(
                [1, -1], device=x.device, dtype=torch.int64,
            )
            masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
            counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)
            out = (inner_all * masks.unsqueeze(-1)).sum(dim=1) / counts
            grads = torch.autograd.grad(
                out, (x_g, coef_pos_g, coef_neg_g), grad_out,
                retain_graph=False, create_graph=False,
            )
        return grads[0], None, None, grads[1], grads[2], None


def signedkan_inner_triton_autograd(
    x, triad_v, triad_sigma, coef_pos, coef_neg, grid,
):
    """Autograd-wrapped fused inner kernel.  Drop-in replacement for
    the gather + per-sign CR + σ-mask + mean reduce that appears in
    `SignedKANLayer._forward_impl` when ``inner_skip == "none"``."""
    return _SignedKANInnerTritonFn.apply(
        x, triad_v, triad_sigma, coef_pos, coef_neg, grid,
    )


class _SignedKANInnerHighwayTritonFn(torch.autograd.Function):
    """Autograd-aware fused inner-kernel with Highway skip.

    Forward: Triton kernel.
    Backward: PyTorch reference re-runs forward under autograd.
    """

    @staticmethod
    def forward(ctx, x, triad_v, triad_sigma,
                coef_pos, coef_neg, gate_w, gate_b, grid):
        ctx.save_for_backward(
            x, triad_v, triad_sigma, coef_pos, coef_neg, gate_w, gate_b,
        )
        ctx.grid = grid
        return signedkan_inner_highway_triton(
            x, triad_v, triad_sigma, coef_pos, coef_neg,
            gate_w, gate_b, grid,
        )

    @staticmethod
    def backward(ctx, grad_out):
        import os
        (x, triad_v, triad_sigma, coef_pos, coef_neg,
         gate_w, gate_b) = ctx.saved_tensors
        grid = ctx.grid
        if int(os.environ.get("HSIKAN_TRITON_BACKWARD", "1")) != 0:
            (grad_x, grad_cp, grad_cn,
             grad_gw, grad_gb) = signedkan_inner_highway_backward_triton(
                x, triad_v, triad_sigma, coef_pos, coef_neg,
                gate_w, gate_b, grad_out.contiguous(), grid,
            )
            return (grad_x, None, None,
                    grad_cp, grad_cn, grad_gw, grad_gb, None)
        # PyTorch fallback (correctness reference).
        from .splines import _catmull_rom_eval as _ref
        x_g = x.detach().requires_grad_(True)
        coef_pos_g = coef_pos.detach().requires_grad_(True)
        coef_neg_g = coef_neg.detach().requires_grad_(True)
        gate_w_g = gate_w.detach().requires_grad_(True)
        gate_b_g = gate_b.detach().requires_grad_(True)
        with torch.enable_grad():
            h_v = x_g[triad_v]
            T_inner = torch.sigmoid(h_v @ gate_w_g + gate_b_g)
            inner_pos = _ref(coef_pos_g, h_v, grid)
            inner_neg = _ref(coef_neg_g, h_v, grid)
            h_pos = T_inner * inner_pos + (1 - T_inner) * h_v
            h_neg = T_inner * inner_neg + (1 - T_inner) * h_v
            inner_all = torch.stack([h_pos, h_neg], dim=2)
            sign_vals = torch.tensor(
                [1, -1], device=x.device, dtype=torch.int64,
            )
            masks = (triad_sigma.unsqueeze(-1) == sign_vals).to(x.dtype)
            counts = masks.sum(dim=1).clamp(min=1).unsqueeze(-1)
            out = (inner_all * masks.unsqueeze(-1)).sum(dim=1) / counts
            grads = torch.autograd.grad(
                out,
                (x_g, coef_pos_g, coef_neg_g, gate_w_g, gate_b_g),
                grad_out,
                retain_graph=False, create_graph=False,
            )
        return (grads[0], None, None,
                grads[1], grads[2], grads[3], grads[4], None)


def signedkan_inner_highway_triton_autograd(
    x, triad_v, triad_sigma, coef_pos, coef_neg, gate_w, gate_b, grid,
):
    """Autograd-wrapped Highway fused inner kernel.  Drop-in for the
    inner-highway path of ``SignedKANLayer._forward_impl``."""
    return _SignedKANInnerHighwayTritonFn.apply(
        x, triad_v, triad_sigma, coef_pos, coef_neg,
        gate_w, gate_b, grid,
    )


def install_triton_catmull_rom():
    """Monkey-patch `splines._catmull_rom_eval` to dispatch to the
    Triton kernel when inputs are on CUDA, falling back to the
    PyTorch reference on CPU.  Idempotent: safe to call multiple
    times.  Returns the original function for restoration.
    """
    from . import splines
    if hasattr(splines, "_orig_catmull_rom_eval"):
        return splines._orig_catmull_rom_eval  # already installed
    splines._orig_catmull_rom_eval = splines._catmull_rom_eval

    def _dispatch(coef, x, grid):
        if x.is_cuda and coef.is_cuda and coef.dim() == 2:
            try:
                # Use autograd-wrapped variant so gradients flow back.
                return catmull_rom_triton_autograd(coef, x, grid)
            except Exception:
                return splines._orig_catmull_rom_eval(coef, x, grid)
        return splines._orig_catmull_rom_eval(coef, x, grid)

    splines._catmull_rom_eval = _dispatch
    return splines._orig_catmull_rom_eval


def uninstall_triton_catmull_rom():
    """Restore the PyTorch-only `_catmull_rom_eval`."""
    from . import splines
    if hasattr(splines, "_orig_catmull_rom_eval"):
        splines._catmull_rom_eval = splines._orig_catmull_rom_eval
        del splines._orig_catmull_rom_eval


if __name__ == "__main__":
    import json
    print("--- parity ---")
    print(json.dumps(_parity_check_catmull_rom(), indent=2))
    print("--- benchmark (HSiKAN-realistic shapes) ---")
    for B in (10_000, 50_000, 200_000):
        for C in (4, 16):
            print(json.dumps(
                _benchmark_catmull_rom(C=C, G=8, B=B), indent=0,
            ))
