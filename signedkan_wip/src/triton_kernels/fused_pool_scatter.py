"""Triton fused pool-scatter forward kernel with vector-CR activation.

Pre-computes Q, K, V outside the kernel (cheap matmul, well-optimised
by cuBLAS), then the kernel does the per-pair Hamilton-real-part dot
+ vector-CR activation + pool-accumulator + atomic-scatter in a
single launch. Inputs:

    Q     : (B, L, h)   pre-Hamilton-signed
    K     : (B, L, h)
    V     : (B, L, h)
    coef_pos, coef_neg : (h, G)  per-channel CR control points
    local : (L, K_total) int64    candidate indices per anchor

Outputs:

    pool_h    : (B, L, h)  anchor ← candidate sum-weighted
    scatter_h : (B, L, h)  anchor → candidate sum-weighted

The user then computes ``y = x + (pool_h + scatter_h) @ W_back``
outside.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_cr_branch(
    R, coef_pos_ptr, coef_neg_ptr, hG, mask, G,
):
    """Inline per-channel CR with sign-branch.

    R: (BLOCK_L, BLOCK_H); hG: (BLOCK_H,) = offs_h * G.
    Returns S of the same shape.
    """
    x_abs = tl.minimum(tl.abs(R), 1.0)
    u = x_abs * (G - 1)
    i = u.to(tl.int32)
    i = tl.minimum(i, G - 2)
    t = u - i.to(tl.float32)
    t2 = t * t
    t3 = t2 * t
    w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
    w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
    w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
    w_p2 = 0.5 * (t3 - t2)
    idx_m1 = tl.maximum(i - 1, 0)
    idx_0  = i
    idx_p1 = tl.minimum(i + 1, G - 1)
    idx_p2 = tl.minimum(i + 2, G - 1)
    hG_b = hG[None, :]
    P_m1_p = tl.load(coef_pos_ptr + hG_b + idx_m1, mask=mask, other=0.0)
    P_0_p  = tl.load(coef_pos_ptr + hG_b + idx_0,  mask=mask, other=0.0)
    P_p1_p = tl.load(coef_pos_ptr + hG_b + idx_p1, mask=mask, other=0.0)
    P_p2_p = tl.load(coef_pos_ptr + hG_b + idx_p2, mask=mask, other=0.0)
    out_pos = w_m1 * P_m1_p + w_0 * P_0_p + w_p1 * P_p1_p + w_p2 * P_p2_p
    P_m1_n = tl.load(coef_neg_ptr + hG_b + idx_m1, mask=mask, other=0.0)
    P_0_n  = tl.load(coef_neg_ptr + hG_b + idx_0,  mask=mask, other=0.0)
    P_p1_n = tl.load(coef_neg_ptr + hG_b + idx_p1, mask=mask, other=0.0)
    P_p2_n = tl.load(coef_neg_ptr + hG_b + idx_p2, mask=mask, other=0.0)
    out_neg = w_m1 * P_m1_n + w_0 * P_0_n + w_p1 * P_p1_n + w_p2 * P_p2_n
    return tl.where(R >= 0, out_pos, out_neg)


@triton.jit
def _fused_pool_scatter_forward_kernel(
    Q_ptr, K_ptr, V_ptr,
    coef_pos_ptr, coef_neg_ptr,
    local_ptr,
    pool_out_ptr, scatter_out_ptr,
    B, L, h, G, K_total,
    scale,
    sB_q, sL_q,         # Q strides
    sB_k, sL_k,         # K strides
    sB_v, sL_v,
    sL_loc, sK_loc,
    sB_po, sL_po,
    sB_so, sL_so,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b  = tl.program_id(0)
    pid_lb = tl.program_id(1)

    offs_l = pid_lb * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < h
    mask = mask_l[:, None] & mask_h[None, :]
    hG = offs_h * G

    # Load anchor Q, K, V for the BLOCK_L anchors.
    q_off = pid_b * sB_q + offs_l[:, None] * sL_q + offs_h[None, :]
    k_off = pid_b * sB_k + offs_l[:, None] * sL_k + offs_h[None, :]
    v_off = pid_b * sB_v + offs_l[:, None] * sL_v + offs_h[None, :]
    Q_a = tl.load(Q_ptr + q_off, mask=mask, other=0.0)
    K_a = tl.load(K_ptr + k_off, mask=mask, other=0.0)
    V_a = tl.load(V_ptr + v_off, mask=mask, other=0.0)

    pool_acc = tl.zeros((BLOCK_L, BLOCK_H), dtype=tl.float32)

    for k_c in range(K_total):
        # Gather j = local[offs_l, k_c].
        loc_off = offs_l * sL_loc + k_c * sK_loc
        j = tl.load(local_ptr + loc_off, mask=mask_l, other=0).to(tl.int32)
        # Load Q_j, K_j, V_j.
        qj_off = pid_b * sB_q + j[:, None] * sL_q + offs_h[None, :]
        kj_off = pid_b * sB_k + j[:, None] * sL_k + offs_h[None, :]
        vj_off = pid_b * sB_v + j[:, None] * sL_v + offs_h[None, :]
        Q_j = tl.load(Q_ptr + qj_off, mask=mask, other=0.0)
        K_j = tl.load(K_ptr + kj_off, mask=mask, other=0.0)
        V_j = tl.load(V_ptr + vj_off, mask=mask, other=0.0)

        # POOL direction: R[i, k_c, h] = Q_a * K_j * scale  (element-wise)
        R_pool = Q_a * K_j * scale
        S_pool = _vector_cr_branch(R_pool, coef_pos_ptr, coef_neg_ptr,
                                    hG, mask, G)
        pool_acc += S_pool * V_j

        # SCATTER direction: R[j, i, h] = Q_j * K_a * scale
        R_scatter = Q_j * K_a * scale
        S_scatter = _vector_cr_branch(R_scatter, coef_pos_ptr,
                                       coef_neg_ptr, hG, mask, G)
        scatter_contrib = S_scatter * V_a

        # Atomic-add to scatter_out[pid_b, j, :].
        sct_off = pid_b * sB_so + j[:, None] * sL_so + offs_h[None, :]
        tl.atomic_add(scatter_out_ptr + sct_off, scatter_contrib,
                       mask=mask)

    # Write pool_acc.
    po_off = pid_b * sB_po + offs_l[:, None] * sL_po + offs_h[None, :]
    tl.store(pool_out_ptr + po_off, pool_acc, mask=mask)


def fused_pool_scatter_forward_triton(
    Q: torch.Tensor,         # (B, L, h)  -- Hamilton-coeffs baked in
    K: torch.Tensor,         # (B, L, h)
    V: torch.Tensor,         # (B, L, h)
    coef_pos: torch.Tensor,  # (h, G)
    coef_neg: torch.Tensor,  # (h, G)
    local: torch.Tensor,     # (L, K_total)
    scale: float,
    block_l: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pool_h, scatter_h), both (B, L, h)."""
    assert Q.is_cuda
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
    coef_pos = coef_pos.contiguous(); coef_neg = coef_neg.contiguous()
    local = local.contiguous().to(torch.int64)
    B, L, h = Q.shape
    _, G = coef_pos.shape
    K_total = local.shape[1]

    pool_h = torch.empty_like(Q)
    scatter_h = torch.zeros_like(Q)   # atomic-add accumulator

    BLOCK_H = max(triton.next_power_of_2(h), 8)
    grid = (B, triton.cdiv(L, block_l))
    sB_q, sL_q, _ = Q.stride()
    sB_k, sL_k, _ = K.stride()
    sB_v, sL_v, _ = V.stride()
    sL_loc, sK_loc = local.stride()
    sB_po, sL_po, _ = pool_h.stride()
    sB_so, sL_so, _ = scatter_h.stride()

    _fused_pool_scatter_forward_kernel[grid](
        Q, K, V,
        coef_pos, coef_neg,
        local,
        pool_h, scatter_h,
        B, L, h, G, K_total,
        scale,
        sB_q, sL_q,
        sB_k, sL_k,
        sB_v, sL_v,
        sL_loc, sK_loc,
        sB_po, sL_po,
        sB_so, sL_so,
        BLOCK_L=block_l,
        BLOCK_H=BLOCK_H,
    )
    return pool_h, scatter_h


# ── Backward kernel (projektive conjugate identity) ─────────────────
# The Hamilton-bilinear backward identity:
#     y = a ⊗ x  =>  dL/dx = a* ⊗ dL/dy ,  dL/da = dL/dy ⊗ x*
# In the pool-scatter context the Hamilton-coefficients (+,-,-,-) are
# baked into Q via Q_signed = Q · coeffs. R = Q_signed · K (element-
# wise per quaternion blade), so the backward through R is element-
# wise too -- the "conjugate" reduces to the same coefficient pattern.
# Element-wise: dR/dQ_signed = K, dR/dK = Q_signed, dR/dscale = R/scale.

@triton.jit
def _vector_cr_branch_and_grad(
    R, coef_pos_ptr, coef_neg_ptr, hG, mask, G,
):
    """Forward + d/dR of vector-CR. Returns (S, dSdR).

    The CR is C^1; the absolute-value branch makes dS/dR sign-aware:
        dS/dR = sign(R) · (G-1) · (dS/du)
    """
    x_abs = tl.minimum(tl.abs(R), 1.0)
    u = x_abs * (G - 1)
    i = u.to(tl.int32)
    i = tl.minimum(i, G - 2)
    t = u - i.to(tl.float32)
    t2 = t * t
    t3 = t2 * t
    w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
    w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
    w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
    w_p2 = 0.5 * (t3 - t2)
    # Derivative w.r.t. t.
    dw_m1 = 0.5 * (-3.0 * t2 + 4.0 * t - 1.0)
    dw_0  = 0.5 * (9.0 * t2 - 10.0 * t)
    dw_p1 = 0.5 * (-9.0 * t2 + 8.0 * t + 1.0)
    dw_p2 = 0.5 * (3.0 * t2 - 2.0 * t)
    idx_m1 = tl.maximum(i - 1, 0)
    idx_0  = i
    idx_p1 = tl.minimum(i + 1, G - 1)
    idx_p2 = tl.minimum(i + 2, G - 1)
    hG_b = hG[None, :]
    # Sign branch: pick coef_pos if R >= 0, else coef_neg.
    P_m1_p = tl.load(coef_pos_ptr + hG_b + idx_m1, mask=mask, other=0.0)
    P_0_p  = tl.load(coef_pos_ptr + hG_b + idx_0,  mask=mask, other=0.0)
    P_p1_p = tl.load(coef_pos_ptr + hG_b + idx_p1, mask=mask, other=0.0)
    P_p2_p = tl.load(coef_pos_ptr + hG_b + idx_p2, mask=mask, other=0.0)
    out_pos = w_m1 * P_m1_p + w_0 * P_0_p + w_p1 * P_p1_p + w_p2 * P_p2_p
    P_m1_n = tl.load(coef_neg_ptr + hG_b + idx_m1, mask=mask, other=0.0)
    P_0_n  = tl.load(coef_neg_ptr + hG_b + idx_0,  mask=mask, other=0.0)
    P_p1_n = tl.load(coef_neg_ptr + hG_b + idx_p1, mask=mask, other=0.0)
    P_p2_n = tl.load(coef_neg_ptr + hG_b + idx_p2, mask=mask, other=0.0)
    out_neg = w_m1 * P_m1_n + w_0 * P_0_n + w_p1 * P_p1_n + w_p2 * P_p2_n
    is_pos = R >= 0
    S = tl.where(is_pos, out_pos, out_neg)
    P_m1 = tl.where(is_pos, P_m1_p, P_m1_n)
    P_0  = tl.where(is_pos, P_0_p,  P_0_n)
    P_p1 = tl.where(is_pos, P_p1_p, P_p1_n)
    P_p2 = tl.where(is_pos, P_p2_p, P_p2_n)
    dSdu = dw_m1 * P_m1 + dw_0 * P_0 + dw_p1 * P_p1 + dw_p2 * P_p2
    sign_R = tl.where(R >= 0, 1.0, -1.0)
    dSdR = sign_R * (G - 1) * dSdu
    return S, dSdR


@triton.jit
def _fused_pool_scatter_backward_kernel(
    Q_ptr, K_ptr, V_ptr,            # forward inputs
    coef_pos_ptr, coef_neg_ptr,
    local_ptr,
    grad_pool_h_ptr,                # (B, L, h)
    grad_scatter_h_ptr,             # (B, L, h)
    grad_Q_ptr, grad_K_ptr, grad_V_ptr,  # accumulators (zero-init outside)
    B, L, h, G, K_total,
    scale,
    sB_q, sL_q, sB_k, sL_k, sB_v, sL_v,
    sL_loc, sK_loc,
    sB_gp, sL_gp, sB_gs, sL_gs,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Backward: produces dQ, dK, dV via per-pair Hamilton-bilinear
    backprop. Coef gradients (dcoef_pos / dcoef_neg) are computed
    in a separate kernel below to keep this one inside the constexpr
    register budget."""
    pid_b  = tl.program_id(0)
    pid_lb = tl.program_id(1)

    offs_l = pid_lb * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < h
    mask = mask_l[:, None] & mask_h[None, :]
    hG = offs_h * G

    q_off = pid_b * sB_q + offs_l[:, None] * sL_q + offs_h[None, :]
    k_off = pid_b * sB_k + offs_l[:, None] * sL_k + offs_h[None, :]
    v_off = pid_b * sB_v + offs_l[:, None] * sL_v + offs_h[None, :]
    Q_a = tl.load(Q_ptr + q_off, mask=mask, other=0.0)
    K_a = tl.load(K_ptr + k_off, mask=mask, other=0.0)
    V_a = tl.load(V_ptr + v_off, mask=mask, other=0.0)
    # Anchor incoming grads.
    gp_off = pid_b * sB_gp + offs_l[:, None] * sL_gp + offs_h[None, :]
    grad_pool_a = tl.load(grad_pool_h_ptr + gp_off, mask=mask, other=0.0)

    # Anchor accumulators.
    dQ_acc = tl.zeros((BLOCK_L, BLOCK_H), dtype=tl.float32)
    dK_a_acc = tl.zeros((BLOCK_L, BLOCK_H), dtype=tl.float32)
    dV_a_acc = tl.zeros((BLOCK_L, BLOCK_H), dtype=tl.float32)

    for k_c in range(K_total):
        loc_off = offs_l * sL_loc + k_c * sK_loc
        j = tl.load(local_ptr + loc_off, mask=mask_l, other=0).to(tl.int32)
        qj_off = pid_b * sB_q + j[:, None] * sL_q + offs_h[None, :]
        kj_off = pid_b * sB_k + j[:, None] * sL_k + offs_h[None, :]
        vj_off = pid_b * sB_v + j[:, None] * sL_v + offs_h[None, :]
        Q_j = tl.load(Q_ptr + qj_off, mask=mask, other=0.0)
        K_j = tl.load(K_ptr + kj_off, mask=mask, other=0.0)
        V_j = tl.load(V_ptr + vj_off, mask=mask, other=0.0)

        # POOL direction backward.
        R_pool = Q_a * K_j * scale
        S_pool, dSdR_pool = _vector_cr_branch_and_grad(
            R_pool, coef_pos_ptr, coef_neg_ptr, hG, mask, G,
        )
        # forward: pool[b, i, h] += S_pool * V_j
        # dS_pool[b, i, k_c, h] = grad_pool_a * V_j
        dS_pool = grad_pool_a * V_j
        # dV_j += S_pool * grad_pool_a  -> atomic into V[b, j]
        dV_j_pool = S_pool * grad_pool_a
        # dR_pool = dS_pool * dSdR_pool
        dR_pool = dS_pool * dSdR_pool
        # dQ_a += dR_pool * K_j * scale  (anchor)
        dQ_acc += dR_pool * K_j * scale
        # dK_j += dR_pool * Q_a * scale  -> atomic
        dK_j_pool = dR_pool * Q_a * scale
        # SCATTER direction backward.
        gs_j_off = pid_b * sB_gs + j[:, None] * sL_gs + offs_h[None, :]
        grad_scatter_j = tl.load(
            grad_scatter_h_ptr + gs_j_off, mask=mask, other=0.0,
        )
        R_scatter = Q_j * K_a * scale
        S_scatter, dSdR_scatter = _vector_cr_branch_and_grad(
            R_scatter, coef_pos_ptr, coef_neg_ptr, hG, mask, G,
        )
        # forward: scatter[b, j, h] += S_scatter * V_a
        dS_scatter = grad_scatter_j * V_a
        # dV_a += S_scatter * grad_scatter_j  (anchor)
        dV_a_acc += S_scatter * grad_scatter_j
        dR_scatter = dS_scatter * dSdR_scatter
        # dQ_j += dR_scatter * K_a * scale  -> atomic
        dQ_j_scatter = dR_scatter * K_a * scale
        # dK_a += dR_scatter * Q_j * scale  (anchor)
        dK_a_acc += dR_scatter * Q_j * scale

        # Atomic adds to j positions.
        tl.atomic_add(grad_V_ptr + vj_off, dV_j_pool, mask=mask)
        tl.atomic_add(grad_K_ptr + kj_off, dK_j_pool, mask=mask)
        tl.atomic_add(grad_Q_ptr + qj_off, dQ_j_scatter, mask=mask)

    # Write anchor accumulators (atomic because other (b, l) programs
    # could also target this anchor via their scatter direction, even
    # though for a single batch the anchor is unique to one program;
    # atomic is just safe).
    tl.atomic_add(grad_Q_ptr + q_off, dQ_acc, mask=mask)
    tl.atomic_add(grad_K_ptr + k_off, dK_a_acc, mask=mask)
    tl.atomic_add(grad_V_ptr + v_off, dV_a_acc, mask=mask)


# ── Hamilton rotor kernel (forward + reverse via M-conjugate) ───────────

@triton.jit
def _hamilton_rotate_kernel(
    field_ptr, M_ptr, out_ptr,
    BL_total,
    n_quat: tl.constexpr,
    BLOCK_BL: tl.constexpr,
):
    """Compute ``out = M ⊗ field`` per ``(b, l, q)`` for the dual-path
    Hamilton rotor. The cat-based PyTorch reference was costing ~0.65 ms
    per call in the AC-HSiKAN classifier profile; this kernel drops it
    to ~0.02 ms (≈ 11× win) by issuing a single load/compute/store pass.

    Layout: ``field``, ``out`` are flattened to ``(BL_total, n_quat·4)``;
    ``M`` is ``(n_quat, 4)``. The quaternion-block loop is unrolled via
    ``tl.static_range`` so the compiler keeps ``m_q`` in registers.
    """
    pid = tl.program_id(0)
    offs_bl = pid * BLOCK_BL + tl.arange(0, BLOCK_BL)
    mask = offs_bl < BL_total
    h = n_quat * 4
    for q in tl.static_range(n_quat):
        m0 = tl.load(M_ptr + q * 4 + 0)
        m1 = tl.load(M_ptr + q * 4 + 1)
        m2 = tl.load(M_ptr + q * 4 + 2)
        m3 = tl.load(M_ptr + q * 4 + 3)
        base = offs_bl * h + q * 4
        f0 = tl.load(field_ptr + base + 0, mask=mask, other=0.0)
        f1 = tl.load(field_ptr + base + 1, mask=mask, other=0.0)
        f2 = tl.load(field_ptr + base + 2, mask=mask, other=0.0)
        f3 = tl.load(field_ptr + base + 3, mask=mask, other=0.0)
        out0 = m0 * f0 - m1 * f1 - m2 * f2 - m3 * f3
        out1 = m0 * f1 + m1 * f0 + m2 * f3 - m3 * f2
        out2 = m0 * f2 - m1 * f3 + m2 * f0 + m3 * f1
        out3 = m0 * f3 + m1 * f2 - m2 * f1 + m3 * f0
        tl.store(out_ptr + base + 0, out0, mask=mask)
        tl.store(out_ptr + base + 1, out1, mask=mask)
        tl.store(out_ptr + base + 2, out2, mask=mask)
        tl.store(out_ptr + base + 3, out3, mask=mask)


def hamilton_rotate_triton(
    field: torch.Tensor,        # (B, L, h)
    M: torch.Tensor,            # (n_quat, 4)
    n_quat: int,
) -> torch.Tensor:
    """Triton fast-path for the Hamilton rotor; CUDA-only.

    The kernel is forward-only: a working context-init exists for the
    autograd backward worker thread (``torch.zeros(1).sum()`` activates
    the CUDA driver primary context), but enabling Triton on the bwd
    side empirically introduced ~0.005 val_acc drift on IMDB smoke
    even though parity tests pass at 1e-7 abs. The forward path saves
    ~0.05 ms/step at no measurable accuracy cost; the backward path
    keeps the PyTorch fallback. Documented in
    reports/2026-06-06-pool-scatter-final-triton-stack.md.
    """
    import threading
    if threading.current_thread() is not threading.main_thread():
        return _hamilton_rotate_pytorch(field, M, n_quat)
    assert field.is_cuda
    B, L, h = field.shape
    BL = B * L
    field_flat = field.contiguous().view(BL, h)
    out = torch.empty_like(field_flat)
    BLOCK_BL = 256
    grid = (triton.cdiv(BL, BLOCK_BL),)
    _hamilton_rotate_kernel[grid](
        field_flat, M.contiguous(), out,
        BL,
        n_quat=n_quat,
        BLOCK_BL=BLOCK_BL,
    )
    return out.view(B, L, h)


def _hamilton_rotate_pytorch(field, M, n_quat):
    """PyTorch fallback (autograd-traceable, thread-safe).

    Numerically bit-identical to the original 16-mul + cat formulation
    (the reference baseline). einsum + Hamilton-matrix-stack was
    measured 1.08× faster standalone but drifted IMDB val_acc by
    ~0.002 due to different FMA ordering -- reverted in favour of
    the verified-stable cat path.
    """
    B, L, h = field.shape
    f = field.view(B, L, n_quat, 4)
    M_b = M.view(1, 1, n_quat, 4)
    m0, m1, m2, m3 = M_b[..., 0:1], M_b[..., 1:2], M_b[..., 2:3], M_b[..., 3:4]
    f0, f1, f2, f3 = f[..., 0:1], f[..., 1:2], f[..., 2:3], f[..., 3:4]
    out0 = m0 * f0 - m1 * f1 - m2 * f2 - m3 * f3
    out1 = m0 * f1 + m1 * f0 + m2 * f3 - m3 * f2
    out2 = m0 * f2 - m1 * f3 + m2 * f0 + m3 * f1
    out3 = m0 * f3 + m1 * f2 - m2 * f1 + m3 * f0
    return torch.cat([out0, out1, out2, out3], dim=-1).view(B, L, h)


# ── CR-coef backward kernel ────────────────────────────────────────────────

@triton.jit
def _cr_coef_backward_kernel(
    Q_ptr, K_ptr, V_ptr,
    local_ptr,
    grad_pool_h_ptr, grad_scatter_h_ptr,
    grad_coef_combined_ptr,             # (2 * h * G,) flat target
    B, L, h, G: tl.constexpr, K_total,
    scale,
    sB_q, sL_q, sB_k, sL_k, sB_v, sL_v,
    sL_loc, sK_loc,
    sB_gp, sL_gp, sB_gs, sL_gs,
    BLOCK_L: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused R-reconstruction + CR-weight + atomic-scatter into (2·h·G,).

    Replaces the closed-form PyTorch path that was the post-optimisation
    bottleneck (3.6 ms / call, ~86 % of bwd). Per (b, l, k_c, c) thread:
      - reconstruct R_pool = Q_a · K_j · scale, R_scatter = Q_j · K_a · scale
      - compute Catmull-Rom weights + indices from R
      - compute dL_dS_pool = grad_pool_a · V_j
      -        dL_dS_scatter = grad_scatter_j · V_a
      - 4 atomic_adds into ``grad_coef_combined[branch·hG + c·G + idx_x]``

    The combined target packs both sign branches (pos at [0:hG], neg at
    [hG:2hG]) and both forward paths (pool + scatter), so all 8 + 8 = 16
    contributions per (b, l, k_c, c) thread land in one tensor.
    """
    pid_b  = tl.program_id(0)
    pid_lb = tl.program_id(1)

    offs_l = pid_lb * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = offs_l < L
    offs_h = tl.arange(0, BLOCK_H)
    mask_h = offs_h < h
    mask = mask_l[:, None] & mask_h[None, :]
    hG = h * G

    # Anchor positions: load Q_signed, K, V, grad_pool_h
    q_off = pid_b * sB_q + offs_l[:, None] * sL_q + offs_h[None, :]
    k_off = pid_b * sB_k + offs_l[:, None] * sL_k + offs_h[None, :]
    v_off = pid_b * sB_v + offs_l[:, None] * sL_v + offs_h[None, :]
    Q_a = tl.load(Q_ptr + q_off, mask=mask, other=0.0)
    K_a = tl.load(K_ptr + k_off, mask=mask, other=0.0)
    V_a = tl.load(V_ptr + v_off, mask=mask, other=0.0)
    gp_off = pid_b * sB_gp + offs_l[:, None] * sL_gp + offs_h[None, :]
    grad_pool_a = tl.load(grad_pool_h_ptr + gp_off, mask=mask, other=0.0)

    c_off = offs_h[None, :] * G   # (1, BLOCK_H)

    for k_c in range(K_total):
        loc_off = offs_l * sL_loc + k_c * sK_loc
        j = tl.load(local_ptr + loc_off, mask=mask_l, other=0).to(tl.int32)
        qj_off = pid_b * sB_q + j[:, None] * sL_q + offs_h[None, :]
        kj_off = pid_b * sB_k + j[:, None] * sL_k + offs_h[None, :]
        vj_off = pid_b * sB_v + j[:, None] * sL_v + offs_h[None, :]
        gs_off = pid_b * sB_gs + j[:, None] * sL_gs + offs_h[None, :]
        Q_j = tl.load(Q_ptr + qj_off, mask=mask, other=0.0)
        K_j = tl.load(K_ptr + kj_off, mask=mask, other=0.0)
        V_j = tl.load(V_ptr + vj_off, mask=mask, other=0.0)
        grad_scatter_j = tl.load(grad_scatter_h_ptr + gs_off,
                                  mask=mask, other=0.0)

        # ── POOL direction ────────────────────────────────────────
        R_pool = Q_a * K_j * scale
        x_abs = tl.minimum(tl.abs(R_pool), 1.0)
        u = x_abs * (G - 1)
        i = u.to(tl.int32)
        i = tl.minimum(i, G - 2)
        t = u - i.to(tl.float32)
        t2 = t * t; t3 = t2 * t
        w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
        w_0  = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
        w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
        w_p2 = 0.5 * (t3 - t2)
        idx_m1 = tl.maximum(i - 1, 0)
        idx_0  = i
        idx_p1 = tl.minimum(i + 1, G - 1)
        idx_p2 = tl.minimum(i + 2, G - 1)
        dL_dS_pool = grad_pool_a * V_j
        is_neg_pool = R_pool < 0
        branch_off = tl.where(is_neg_pool, hG, 0)
        base = branch_off + c_off
        tl.atomic_add(grad_coef_combined_ptr + base + idx_m1,
                       dL_dS_pool * w_m1, mask=mask)
        tl.atomic_add(grad_coef_combined_ptr + base + idx_0,
                       dL_dS_pool * w_0,  mask=mask)
        tl.atomic_add(grad_coef_combined_ptr + base + idx_p1,
                       dL_dS_pool * w_p1, mask=mask)
        tl.atomic_add(grad_coef_combined_ptr + base + idx_p2,
                       dL_dS_pool * w_p2, mask=mask)

        # ── SCATTER direction ─────────────────────────────────────
        R_scatter = Q_j * K_a * scale
        x_abs_s = tl.minimum(tl.abs(R_scatter), 1.0)
        u_s = x_abs_s * (G - 1)
        i_s = u_s.to(tl.int32)
        i_s = tl.minimum(i_s, G - 2)
        t_s = u_s - i_s.to(tl.float32)
        t2_s = t_s * t_s; t3_s = t2_s * t_s
        w_m1_s = 0.5 * (-t3_s + 2.0 * t2_s - t_s)
        w_0_s  = 0.5 * (3.0 * t3_s - 5.0 * t2_s + 2.0)
        w_p1_s = 0.5 * (-3.0 * t3_s + 4.0 * t2_s + t_s)
        w_p2_s = 0.5 * (t3_s - t2_s)
        idx_m1_s = tl.maximum(i_s - 1, 0)
        idx_0_s  = i_s
        idx_p1_s = tl.minimum(i_s + 1, G - 1)
        idx_p2_s = tl.minimum(i_s + 2, G - 1)
        dL_dS_scatter = grad_scatter_j * V_a
        is_neg_s = R_scatter < 0
        branch_off_s = tl.where(is_neg_s, hG, 0)
        base_s = branch_off_s + c_off
        tl.atomic_add(grad_coef_combined_ptr + base_s + idx_m1_s,
                       dL_dS_scatter * w_m1_s, mask=mask)
        tl.atomic_add(grad_coef_combined_ptr + base_s + idx_0_s,
                       dL_dS_scatter * w_0_s,  mask=mask)
        tl.atomic_add(grad_coef_combined_ptr + base_s + idx_p1_s,
                       dL_dS_scatter * w_p1_s, mask=mask)
        tl.atomic_add(grad_coef_combined_ptr + base_s + idx_p2_s,
                       dL_dS_scatter * w_p2_s, mask=mask)


def cr_coef_backward_triton(
    Q_signed: torch.Tensor,             # (B, L, h)
    K: torch.Tensor, V: torch.Tensor,
    local: torch.Tensor,                # (L, K_total)
    grad_pool_h: torch.Tensor,
    grad_scatter_h: torch.Tensor,
    scale: float, G: int,
    block_l: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (grad_coef_pos, grad_coef_neg), each (h, G)."""
    assert Q_signed.is_cuda
    Q_signed = Q_signed.contiguous(); K = K.contiguous(); V = V.contiguous()
    local = local.contiguous().to(torch.int64)
    grad_pool_h = grad_pool_h.contiguous()
    grad_scatter_h = grad_scatter_h.contiguous()
    B, L, h = Q_signed.shape
    K_total = local.shape[1]
    BLOCK_H = max(triton.next_power_of_2(h), 8)
    grid = (B, triton.cdiv(L, block_l))

    combined = torch.zeros(2 * h * G, device=Q_signed.device,
                            dtype=Q_signed.dtype)

    sB_q, sL_q, _ = Q_signed.stride()
    sB_k, sL_k, _ = K.stride()
    sB_v, sL_v, _ = V.stride()
    sL_loc, sK_loc = local.stride()
    sB_gp, sL_gp, _ = grad_pool_h.stride()
    sB_gs, sL_gs, _ = grad_scatter_h.stride()

    _cr_coef_backward_kernel[grid](
        Q_signed, K, V,
        local,
        grad_pool_h, grad_scatter_h,
        combined,
        B, L, h, G, K_total,
        scale,
        sB_q, sL_q, sB_k, sL_k, sB_v, sL_v,
        sL_loc, sK_loc,
        sB_gp, sL_gp, sB_gs, sL_gs,
        BLOCK_L=block_l, BLOCK_H=BLOCK_H,
    )
    return combined[:h*G].view(h, G), combined[h*G:].view(h, G)


def fused_pool_scatter_backward_triton(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    coef_pos: torch.Tensor, coef_neg: torch.Tensor,
    local: torch.Tensor,
    grad_pool_h: torch.Tensor, grad_scatter_h: torch.Tensor,
    scale: float, block_l: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (grad_Q, grad_K, grad_V), all (B, L, h).

    Coef gradients (dcoef_pos, dcoef_neg) are computed via PyTorch
    autograd on the reference CR eval in the Python wrapper; the
    heavy gradient flow (Q, K, V) is fused here.
    """
    assert Q.is_cuda
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous()
    coef_pos = coef_pos.contiguous(); coef_neg = coef_neg.contiguous()
    local = local.contiguous().to(torch.int64)
    grad_pool_h = grad_pool_h.contiguous()
    grad_scatter_h = grad_scatter_h.contiguous()
    B, L, h = Q.shape
    _, G = coef_pos.shape
    K_total = local.shape[1]
    BLOCK_H = max(triton.next_power_of_2(h), 8)
    grid = (B, triton.cdiv(L, block_l))

    grad_Q = torch.zeros_like(Q)
    grad_K = torch.zeros_like(K)
    grad_V = torch.zeros_like(V)

    sB_q, sL_q, _ = Q.stride()
    sB_k, sL_k, _ = K.stride()
    sB_v, sL_v, _ = V.stride()
    sL_loc, sK_loc = local.stride()
    sB_gp, sL_gp, _ = grad_pool_h.stride()
    sB_gs, sL_gs, _ = grad_scatter_h.stride()

    _fused_pool_scatter_backward_kernel[grid](
        Q, K, V,
        coef_pos, coef_neg,
        local,
        grad_pool_h, grad_scatter_h,
        grad_Q, grad_K, grad_V,
        B, L, h, G, K_total,
        scale,
        sB_q, sL_q, sB_k, sL_k, sB_v, sL_v,
        sL_loc, sK_loc,
        sB_gp, sL_gp, sB_gs, sL_gs,
        BLOCK_L=block_l, BLOCK_H=BLOCK_H,
    )
    return grad_Q, grad_K, grad_V
