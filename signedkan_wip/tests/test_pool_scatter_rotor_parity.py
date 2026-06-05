"""Parity test: FusedPoolScatter Triton fwd+bwd with entropy-rotor vs
a full-autograd PyTorch reference that applies the rotor identically.

This validates two things at once:
  1. The Triton forward + closed-form backward in
     ``_FusedPoolScatterTritonFn`` agree with a PyTorch reference
     where the rotor is applied as M ⊗ scatter_h between the
     pool-scatter primitive and the W_back projection.
  2. The closed-form ``_compute_rotor_grad`` agrees with the autograd
     gradient on ``entropy_axis`` and ``entropy_beta`` through that
     same reference.

CUDA-only -- the autograd Function dispatches to Triton on CUDA and
falls back to a no-rotor CPU path otherwise.
"""
from __future__ import annotations

import pytest
import torch

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernel",
)

from signedkan_wip.src.ac_hsikan.components.pool_scatter import (  # noqa: E402
    FusedPoolScatter,
    fused_pool_scatter_reference,
    _hamilton_rotate_static,
)


# ---- Reference path with rotor injected -----------------------------------

def _reference_with_rotor(
    x, W_q, W_k, W_v, W_back, coef_pos, coef_neg, local,
    n_quat, temperature, G, rotor_M,
):
    """PyTorch reference that applies M ⊗ scatter_h before W_back.

    Mirrors :func:`fused_pool_scatter_reference` but cracks the pool +
    scatter sum so the rotor can be injected on scatter_h alone.
    """
    from signedkan_wip.src.ac_hsikan.components.pool_scatter import (
        _vector_cr_eval,
    )
    B, L, d = x.shape
    h = W_q.shape[1]
    K_total = local.shape[1]
    scale = (n_quat ** -0.5) / temperature

    Q = x @ W_q
    K_ = x @ W_k
    V = x @ W_v
    coeffs = torch.tensor([1.0, -1.0, -1.0, -1.0] * n_quat,
                          device=x.device, dtype=x.dtype)
    Q_signed = Q * coeffs

    # POOL: anchor i ← candidates
    Q_a = Q_signed.unsqueeze(2).expand(B, L, K_total, h)
    K_c = K_[:, local, :]
    R_pool = Q_a * K_c * scale
    S_pool = _vector_cr_eval(R_pool, coef_pos, coef_neg, G)
    V_c = V[:, local, :]
    pool_h = (S_pool * V_c).sum(dim=2)

    # SCATTER: anchor i → candidates
    Q_j = Q_signed[:, local, :]
    K_i = K_.unsqueeze(2).expand(B, L, K_total, h)
    R_scatter = Q_j * K_i * scale
    S_scatter = _vector_cr_eval(R_scatter, coef_pos, coef_neg, G)
    V_a = V.unsqueeze(2).expand(B, L, K_total, h)
    scatter_contrib = S_scatter * V_a
    scatter_h = torch.zeros(B, L, h, device=x.device, dtype=x.dtype)
    j_flat = (local.unsqueeze(0).expand(B, L, K_total)
                    .reshape(B, L * K_total))
    j_full = j_flat.unsqueeze(-1).expand(B, L * K_total, h)
    contrib_flat = scatter_contrib.reshape(B, L * K_total, h)
    scatter_h = scatter_h.scatter_add(dim=1, index=j_full, src=contrib_flat)

    # Apply rotor on scatter_h only (matches the Triton path).
    if rotor_M is not None:
        scatter_h = _hamilton_rotate_static(scatter_h, rotor_M, n_quat)

    out = x + (pool_h + scatter_h) @ W_back
    return out


# ---- Fixture builder ------------------------------------------------------

def _build_inputs(seed=0, B=2, L=8, K_total=4, d=16, h=8, n_quat=2, G=8,
                  device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, L, d, device=device, generator=g, requires_grad=False)
    local = torch.randint(0, L, (L, K_total), generator=torch.Generator()
                                              .manual_seed(seed + 1),
                          device="cpu").to(device)
    return x, local, dict(B=B, L=L, K_total=K_total, d=d, h=h, n_quat=n_quat, G=G)


# ---- Tests ----------------------------------------------------------------

@cuda_only
def test_forward_parity_with_rotor():
    """Triton+closed-form forward (with rotor) == PyTorch reference forward."""
    device = "cuda"
    x, local, dims = _build_inputs(device=device)
    mod = FusedPoolScatter(
        d_model=dims["d"], h=dims["h"], n_quat=dims["n_quat"], G=dims["G"],
        entropy_feedback=True,
    ).to(device)
    # Modest non-zero β + entropy so rotor != identity.
    with torch.no_grad():
        mod.entropy_beta.fill_(0.3)
    H = torch.tensor(0.7, device=device)
    M = mod._build_rotor(H)

    y_kernel = mod(x, local, entropy_scalar=H)

    y_ref = _reference_with_rotor(
        x, mod.W_q.weight.T, mod.W_k.weight.T, mod.W_v.weight.T,
        mod.W_back.weight.T, mod.coef_pos, mod.coef_neg, local,
        dims["n_quat"], 1.0, dims["G"], M,
    )

    diff = (y_kernel - y_ref).abs().max().item()
    assert diff < 1e-4, f"forward parity broken: max abs diff = {diff:.3e}"


@cuda_only
def test_backward_parity_with_rotor():
    """Closed-form backward (with rotor) agrees with autograd-through-reference
    on every parameter and on x."""
    device = "cuda"
    x, local, dims = _build_inputs(device=device)

    # Build two parallel modules with identical params.
    mod = FusedPoolScatter(
        d_model=dims["d"], h=dims["h"], n_quat=dims["n_quat"], G=dims["G"],
        entropy_feedback=True,
    ).to(device)
    with torch.no_grad():
        mod.entropy_beta.fill_(0.3)

    # Inputs require grad on both paths.
    x_kernel = x.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)
    H_kernel = torch.tensor(0.7, device=device, requires_grad=True)
    H_ref = torch.tensor(0.7, device=device, requires_grad=True)

    # Build rotor on each side from same entropy + same params.
    M_ref = mod._build_rotor(H_ref)

    # ── Kernel path ────────────────────────────────────────────────
    y_kernel = mod(x_kernel, local, entropy_scalar=H_kernel)
    grad_out = torch.randn_like(y_kernel)
    y_kernel.backward(grad_out)
    g_x_k = x_kernel.grad.clone()
    g_Wq_k = mod.W_q.weight.grad.clone()
    g_Wk_k = mod.W_k.weight.grad.clone()
    g_Wv_k = mod.W_v.weight.grad.clone()
    g_Wb_k = mod.W_back.weight.grad.clone()
    g_cp_k = mod.coef_pos.grad.clone()
    g_cn_k = mod.coef_neg.grad.clone()
    g_axis_k = mod.entropy_axis.grad.clone()
    g_beta_k = mod.entropy_beta.grad.clone()

    # Zero grads for reference path.
    for p in mod.parameters():
        p.grad = None

    # ── Reference path ─────────────────────────────────────────────
    y_ref = _reference_with_rotor(
        x_ref, mod.W_q.weight.T, mod.W_k.weight.T, mod.W_v.weight.T,
        mod.W_back.weight.T, mod.coef_pos, mod.coef_neg, local,
        dims["n_quat"], 1.0, dims["G"], M_ref,
    )
    y_ref.backward(grad_out)
    g_x_r = x_ref.grad.clone()
    g_Wq_r = mod.W_q.weight.grad.clone()
    g_Wk_r = mod.W_k.weight.grad.clone()
    g_Wv_r = mod.W_v.weight.grad.clone()
    g_Wb_r = mod.W_back.weight.grad.clone()
    g_cp_r = mod.coef_pos.grad.clone()
    g_cn_r = mod.coef_neg.grad.clone()
    g_axis_r = mod.entropy_axis.grad.clone()
    g_beta_r = mod.entropy_beta.grad.clone()

    # ── Parity ────────────────────────────────────────────────────
    def _diff(a, b, name):
        d_abs = (a - b).abs().max().item()
        d_rel = d_abs / (b.abs().max().item() + 1e-12)
        return name, d_abs, d_rel

    rows = [
        _diff(g_x_k, g_x_r, "∂L/∂x"),
        _diff(g_Wq_k, g_Wq_r, "∂L/∂W_q"),
        _diff(g_Wk_k, g_Wk_r, "∂L/∂W_k"),
        _diff(g_Wv_k, g_Wv_r, "∂L/∂W_v"),
        _diff(g_Wb_k, g_Wb_r, "∂L/∂W_back"),
        _diff(g_cp_k, g_cp_r, "∂L/∂coef_pos"),
        _diff(g_cn_k, g_cn_r, "∂L/∂coef_neg"),
        _diff(g_axis_k, g_axis_r, "∂L/∂entropy_axis"),
        _diff(g_beta_k, g_beta_r, "∂L/∂entropy_beta"),
    ]
    msg = "\n".join(f"  {n:>18}: abs={a:.3e}  rel={r:.3e}"
                    for n, a, r in rows)
    # Tight abs tolerance for kernel-vs-reference numerical agreement.
    # CR-coef and rotor grads use slightly different code paths so
    # allow a touch more slack there.
    tol = {
        "∂L/∂x":             5e-4,
        "∂L/∂W_q":           5e-4,
        "∂L/∂W_k":           5e-4,
        "∂L/∂W_v":           5e-4,
        "∂L/∂W_back":        5e-4,
        "∂L/∂coef_pos":      5e-4,
        "∂L/∂coef_neg":      5e-4,
        "∂L/∂entropy_axis":  5e-4,
        "∂L/∂entropy_beta":  5e-4,
    }
    failures = [(n, a) for n, a, _ in rows if a > tol[n]]
    assert not failures, f"backward parity broken:\n{msg}"


@cuda_only
def test_rotor_identity_when_beta_zero():
    """When β=0 the rotor reduces to identity; output must match no-rotor path."""
    device = "cuda"
    x, local, dims = _build_inputs(device=device)
    mod = FusedPoolScatter(
        d_model=dims["d"], h=dims["h"], n_quat=dims["n_quat"], G=dims["G"],
        entropy_feedback=True,
    ).to(device)
    with torch.no_grad():
        mod.entropy_beta.zero_()
    H = torch.tensor(2.0, device=device)  # any entropy

    y_with_rotor = mod(x, local, entropy_scalar=H)

    # No-rotor path: build a parallel module sharing the same params.
    mod_off = FusedPoolScatter(
        d_model=dims["d"], h=dims["h"], n_quat=dims["n_quat"], G=dims["G"],
        entropy_feedback=False,
    ).to(device)
    with torch.no_grad():
        mod_off.W_q.weight.copy_(mod.W_q.weight)
        mod_off.W_k.weight.copy_(mod.W_k.weight)
        mod_off.W_v.weight.copy_(mod.W_v.weight)
        mod_off.W_back.weight.copy_(mod.W_back.weight)
        mod_off.coef_pos.copy_(mod.coef_pos)
        mod_off.coef_neg.copy_(mod.coef_neg)
    y_no_rotor = mod_off(x, local)

    diff = (y_with_rotor - y_no_rotor).abs().max().item()
    assert diff < 1e-5, f"β=0 should be identity: max diff = {diff:.3e}"
