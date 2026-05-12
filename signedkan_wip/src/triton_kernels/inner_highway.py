"""Auto-split from signedkan_wip/src/triton_kernels.py 2026-05-11.
CLAUDE.md §6.5 #4 (no monstrosity > 300 LOC).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

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

