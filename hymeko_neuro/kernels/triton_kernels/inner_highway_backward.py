"""Inner-highway backward kernel — auto-split 2026-05-11 (CLAUDE.md §6.5 #4)."""
from __future__ import annotations
import torch
import triton
import triton.language as tl  # noqa: E702

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

