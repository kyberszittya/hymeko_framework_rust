"""Auto-split from signedkan_wip/src/triton_kernels.py 2026-05-11.
CLAUDE.md §6.5 #4 (no monstrosity > 300 LOC).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

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

