"""Auto-split from signedkan_wip/src/triton_kernels.py 2026-05-11.
CLAUDE.md §6.5 #4 (no monstrosity > 300 LOC).
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

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

