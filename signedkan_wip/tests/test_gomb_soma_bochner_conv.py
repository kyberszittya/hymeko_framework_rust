"""Tests for BochnerHypergraphConv (Ricci-Stim phase 4).

The headline test is `test_alpha_beta_zero_reproduces_inner_exactly` —
with both Bochner coefficients set to zero, the wrapper's forward
output must be bit-identical to the inner layer's forward output.
This pins the additivity contract: turning on Bochner coupling cannot
break existing Walk / Polygon / Triangle behaviour.

The other tests cover:
  * α > 0 changes the output (Hodge term active);
  * β > 0 changes the output (Ricci term active);
  * gradient flow to α, β, hodge_proj, ricci_proj, and inner params;
  * prepare()-based state plumbing;
  * works with Walk and Polygon inner layers (sign-branched);
  * forward signature inherited from HypergraphConv (same 4-arg form);
  * preconditions still rejected.
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma import (
    BochnerHypergraphConv,
    HypergraphConvConfig,
    PolygonConvLayer,
    WalkConvLayer,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_walk_inputs(n_nodes, k_arity, n_walks, d_in, seed=0):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, d_in, generator=rng)
    walks = torch.randint(
        0, n_nodes, (n_walks, k_arity), generator=rng, dtype=torch.long,
    )
    signs = torch.where(
        torch.rand(n_walks, generator=rng) > 0.5,
        torch.ones(n_walks, dtype=torch.int64),
        -torch.ones(n_walks, dtype=torch.int64),
    )
    rows = walks.reshape(-1)
    cols = torch.arange(n_walks).repeat_interleave(k_arity)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.full((rows.shape[0],), 1.0 / k_arity)
    M_v = torch.sparse_coo_tensor(
        indices, values, (n_nodes, n_walks),
    ).coalesce()
    return x, walks, signs, M_v


def _build_synthetic_hodge_laplacian(n_nodes: int, seed: int = 0):
    """Build a small connected sparse Δ_0 with the right shape and
    PSD structure for testing the Hodge term."""
    rng = torch.Generator().manual_seed(seed)
    rows, cols, vals = [], [], []
    # Self-loops (diagonal degree = 2 for simplicity).
    for i in range(n_nodes):
        rows.append(i); cols.append(i); vals.append(2.0)
    # Cycle edges to make it like a path/cycle Laplacian.
    for i in range(n_nodes - 1):
        rows.append(i); cols.append(i + 1); vals.append(-1.0)
        rows.append(i + 1); cols.append(i); vals.append(-1.0)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    values = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(
        indices, values, (n_nodes, n_nodes),
    ).coalesce()


# ---------------------------------------------------------------------
# Headline regression contract
# ---------------------------------------------------------------------


def test_alpha_beta_zero_reproduces_inner_exactly():
    """The critical contract: with alpha = beta = 0, wrapper output
    is bit-identical to the inner layer's output. This guarantees
    Phase 4 is purely additive over Phases 2 / 3-G."""
    torch.manual_seed(0)
    cfg = HypergraphConvConfig(in_features=8, out_features=8, k_arity=3)
    inner = WalkConvLayer(cfg)
    wrapper = BochnerHypergraphConv(inner, alpha=0.0, beta=0.0)
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=10, k_arity=3, n_walks=20, d_in=8, seed=42,
    )
    # Even with non-trivial Hodge / curvature inputs prepared, the
    # alpha=beta=0 gating must zero out their contributions.
    wrapper.prepare(
        hodge_laplacian=_build_synthetic_hodge_laplacian(10),
        primitive_curvatures=torch.randn(20),
    )
    inner.eval()
    wrapper.eval()
    y_inner = inner(x, walks, signs, M_v)
    y_wrapper = wrapper(x, walks, signs, M_v)
    assert torch.equal(y_inner, y_wrapper), (
        f"α=β=0 regression failed; max diff = "
        f"{(y_inner - y_wrapper).abs().max().item():.2e}"
    )


def test_alpha_beta_zero_no_prepare_call():
    """Same regression but without calling prepare() at all. The
    wrapper should default to flat-connection-only behaviour."""
    torch.manual_seed(0)
    cfg = HypergraphConvConfig(in_features=8, out_features=8, k_arity=3)
    inner = WalkConvLayer(cfg)
    wrapper = BochnerHypergraphConv(inner, alpha=0.0, beta=0.0)
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=10, k_arity=3, n_walks=20, d_in=8, seed=1,
    )
    inner.eval()
    wrapper.eval()
    y_inner = inner(x, walks, signs, M_v)
    y_wrapper = wrapper(x, walks, signs, M_v)
    assert torch.equal(y_inner, y_wrapper)


# ---------------------------------------------------------------------
# Active-coupling tests
# ---------------------------------------------------------------------


def test_alpha_nonzero_changes_output():
    """Turning on the Hodge term must move the output (with the
    Hodge Laplacian actually plumbed)."""
    torch.manual_seed(0)
    cfg = HypergraphConvConfig(in_features=6, out_features=6, k_arity=3)
    inner = WalkConvLayer(cfg)
    wrapper = BochnerHypergraphConv(inner, alpha=0.5, beta=0.0)
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=12, k_arity=3, n_walks=20, d_in=6, seed=3,
    )
    wrapper.prepare(
        hodge_laplacian=_build_synthetic_hodge_laplacian(12),
    )
    wrapper.eval()
    y_off = inner(x, walks, signs, M_v)
    y_on = wrapper(x, walks, signs, M_v)
    diff = (y_off - y_on).abs().max().item()
    assert diff > 1e-3, (
        f"Hodge term is dead — α=0.5 changed output by only {diff:.2e}"
    )


def test_beta_nonzero_changes_output():
    """Turning on the Ricci term must move the output (with curvatures
    actually plumbed)."""
    torch.manual_seed(0)
    cfg = HypergraphConvConfig(in_features=6, out_features=6, k_arity=3)
    inner = WalkConvLayer(cfg)
    wrapper = BochnerHypergraphConv(inner, alpha=0.0, beta=0.5)
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=12, k_arity=3, n_walks=20, d_in=6, seed=4,
    )
    wrapper.prepare(
        primitive_curvatures=torch.randn(20),
    )
    wrapper.eval()
    y_off = inner(x, walks, signs, M_v)
    y_on = wrapper(x, walks, signs, M_v)
    diff = (y_off - y_on).abs().max().item()
    assert diff > 1e-3, (
        f"Ricci term is dead — β=0.5 changed output by only {diff:.2e}"
    )


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


def test_gradient_flow_all_components():
    """Backward must populate gradients on: alpha, beta, hodge_proj,
    ricci_proj, AND the inner layer's parameters."""
    torch.manual_seed(0)
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    inner = WalkConvLayer(cfg)
    wrapper = BochnerHypergraphConv(inner, alpha=0.5, beta=0.5)
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=8, k_arity=3, n_walks=12, d_in=4, seed=7,
    )
    wrapper.prepare(
        hodge_laplacian=_build_synthetic_hodge_laplacian(8),
        primitive_curvatures=torch.randn(12),
    )
    y = wrapper(x, walks, signs, M_v)
    loss = y.pow(2).sum()
    loss.backward()
    # Wrapper-specific params
    assert wrapper.alpha.grad is not None and wrapper.alpha.grad.item() != 0.0
    assert wrapper.beta.grad is not None and wrapper.beta.grad.item() != 0.0
    assert wrapper.hodge_proj.weight.grad is not None
    assert wrapper.hodge_proj.weight.grad.abs().sum().item() > 0
    assert wrapper.ricci_proj.weight.grad is not None
    assert wrapper.ricci_proj.weight.grad.abs().sum().item() > 0
    # Inner params still receive gradient.
    for name, p in inner.named_parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0, (
            f"inner param {name!r} has no gradient"
        )


# ---------------------------------------------------------------------
# Composition with different inner layers
# ---------------------------------------------------------------------


def test_works_with_polygon_inner():
    """Wrapper should accept PolygonConvLayer as inner (k_arity ≥ 3,
    cyclic-invariant). With α=β=0, still bit-identical."""
    cfg = HypergraphConvConfig(in_features=6, out_features=6, k_arity=4)
    inner = PolygonConvLayer(cfg).eval()
    wrapper = BochnerHypergraphConv(inner, alpha=0.0, beta=0.0).eval()
    n_nodes, k, n_polys, d_in = 10, 4, 15, 6
    rng = torch.Generator().manual_seed(11)
    x = torch.randn(n_nodes, d_in, generator=rng)
    polys = torch.randint(0, n_nodes, (n_polys, k), generator=rng,
                           dtype=torch.long)
    signs = torch.where(
        torch.rand(n_polys, generator=rng) > 0.5,
        torch.ones(n_polys, dtype=torch.int64),
        -torch.ones(n_polys, dtype=torch.int64),
    )
    rows = polys.reshape(-1)
    cols = torch.arange(n_polys).repeat_interleave(k)
    M_v = torch.sparse_coo_tensor(
        torch.stack([rows, cols], dim=0),
        torch.full((rows.shape[0],), 1.0 / k),
        (n_nodes, n_polys),
    ).coalesce()
    y_inner = inner(x, polys, signs, M_v)
    y_wrap = wrapper(x, polys, signs, M_v)
    assert torch.equal(y_inner, y_wrap)


# ---------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------


def test_forward_signature_same_as_inner():
    """Wrapper.forward(x, primitives, signs, M_v) matches inner's
    signature — drop-in replacement."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    inner = WalkConvLayer(cfg).eval()
    wrapper = BochnerHypergraphConv(inner).eval()
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=8, k_arity=3, n_walks=10, d_in=4,
    )
    # The same call works on both.
    _ = inner(x, walks, signs, M_v)
    _ = wrapper(x, walks, signs, M_v)


def test_preconditions_inherited():
    """Wrapper inherits HypergraphConv's precondition checks. A
    malformed primitives tensor should be rejected."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    inner = WalkConvLayer(cfg)
    wrapper = BochnerHypergraphConv(inner)
    x, _walks, signs, M_v = _build_walk_inputs(
        n_nodes=8, k_arity=3, n_walks=10, d_in=4,
    )
    bad_walks = torch.zeros((10, 5), dtype=torch.long)  # k=5 ≠ config.k_arity=3
    with pytest.raises(ValueError, match="primitives has shape"):
        wrapper(x, bad_walks, signs, M_v)


def test_output_shape():
    cfg = HypergraphConvConfig(in_features=4, out_features=12, k_arity=3)
    inner = WalkConvLayer(cfg).eval()
    wrapper = BochnerHypergraphConv(inner).eval()
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=8, k_arity=3, n_walks=10, d_in=4,
    )
    y = wrapper(x, walks, signs, M_v)
    assert y.shape == (8, 12)


def test_param_count_overhead():
    """Wrapper adds exactly: 1 (α) + 1 (β) + Linear(in, out) × 2."""
    cfg = HypergraphConvConfig(in_features=8, out_features=8, k_arity=3)
    inner = WalkConvLayer(cfg)
    inner_count = sum(p.numel() for p in inner.parameters())
    wrapper = BochnerHypergraphConv(inner)
    wrapper_count = sum(p.numel() for p in wrapper.parameters())
    overhead = wrapper_count - inner_count
    # Each Linear(8, 8): 8*8 + 8 = 72. Two of them: 144. Plus 2 scalars.
    assert overhead == 2 + 2 * (8 * 8 + 8), (
        f"expected overhead {2 + 2 * (8 * 8 + 8)}, got {overhead}"
    )


def test_non_learnable_mixing_uses_buffers():
    """learnable_mixing=False registers alpha/beta as buffers, not
    parameters; param count drops by 2."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3)
    inner = WalkConvLayer(cfg)
    w_learn = BochnerHypergraphConv(inner, learnable_mixing=True)
    w_buf = BochnerHypergraphConv(inner, learnable_mixing=False)
    n_learn = sum(p.numel() for p in w_learn.parameters())
    n_buf = sum(p.numel() for p in w_buf.parameters())
    assert n_learn - n_buf == 2  # alpha and beta moved to buffers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
