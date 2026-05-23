"""Tests for WalkConvLayer (GömbSoma phase 2).

Pins the WalkConv contract:

  * inherits HypergraphConv semantics (permutation equivariance,
    sparse aggregation, gradient flow);
  * is position-aware (walks are directed; reversal generally
    changes the output);
  * is sign-branched (positive and negative walks route through
    independent weight banks);
  * has the documented parameter count;
  * runs on a small SBM smoke graph end-to-end and produces a
    non-degenerate output.

Polygon / triangle / abstraction layers come in phases 3-5.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma import (
    HypergraphConvConfig,
    WalkConvLayer,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_walk_inputs(n_nodes, k_arity, n_walks, d_in, seed=0):
    """Synthetic (x, walks, signs, M_v) tuple."""
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


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_construction_and_param_count():
    """A k_arity=3, in=16, out=16 WalkConv has the documented count."""
    cfg = HypergraphConvConfig(in_features=16, out_features=16, k_arity=3)
    layer = WalkConvLayer(cfg)
    expected_W = 2 * 3 * 16 * 16   # n_branches × k_arity × in × out
    expected_b = 2 * 16            # n_branches × out
    actual = sum(p.numel() for p in layer.parameters())
    assert actual == expected_W + expected_b, (
        f"expected {expected_W + expected_b} params, got {actual}"
    )


def test_forward_shape():
    cfg = HypergraphConvConfig(in_features=8, out_features=12, k_arity=4)
    layer = WalkConvLayer(cfg).eval()
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=15, k_arity=4, n_walks=30, d_in=8,
    )
    y = layer(x, walks, signs, M_v)
    assert y.shape == (15, 12), f"expected (15, 12), got {tuple(y.shape)}"


def test_sign_branching_actually_branches():
    """Flipping every walk's sign in place must change the output
    (because positive and negative walks go through different W banks).
    A layer that ignored the sign would produce the same output."""
    cfg = HypergraphConvConfig(in_features=6, out_features=6, k_arity=3,
                                use_sign_branching=True)
    layer = WalkConvLayer(cfg).eval()
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=10, k_arity=3, n_walks=20, d_in=6, seed=11,
    )
    y_pos = layer(x, walks, signs, M_v)
    y_neg = layer(x, walks, -signs, M_v)
    diff = (y_pos - y_neg).abs().max().item()
    assert diff > 1e-3, (
        f"sign branching is degenerate — flipping signs barely changed "
        f"the output (max diff = {diff:.2e})"
    )


def test_position_awareness():
    """Reversing each walk's vertex order should generally change the
    output (walks are directed). Verified by constructing two input
    tensors that differ only by per-walk reversal."""
    cfg = HypergraphConvConfig(in_features=6, out_features=6, k_arity=4)
    layer = WalkConvLayer(cfg).eval()
    n_nodes, k, n_walks, d_in = 12, 4, 15, 6
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=n_nodes, k_arity=k, n_walks=n_walks, d_in=d_in, seed=3,
    )
    walks_reversed = walks.flip(dims=[1])
    # M_v is unchanged (same vertex-set per walk); only the order
    # within each row of `walks` changes.
    y_fwd = layer(x, walks, signs, M_v)
    y_rev = layer(x, walks_reversed, signs, M_v)
    diff = (y_fwd - y_rev).abs().max().item()
    assert diff > 1e-3, (
        f"layer is position-blind — reversal didn't change output "
        f"(max diff = {diff:.2e}); walks should be directed"
    )


def test_permutation_equivariance():
    """Inherited from HypergraphConv: vertex permutation passes
    through. WalkConv must not break this — verified here for the
    concrete subclass."""
    cfg = HypergraphConvConfig(in_features=6, out_features=8, k_arity=3,
                                use_sign_branching=True)
    layer = WalkConvLayer(cfg).eval()
    n_nodes, k, n_walks, d_in = 11, 3, 22, 6
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=n_nodes, k_arity=k, n_walks=n_walks, d_in=d_in, seed=99,
    )
    perm = torch.randperm(n_nodes)
    inv = torch.argsort(perm)
    x_p = x[inv]
    walks_p = perm[walks]
    M_dense = M_v.to_dense()
    M_v_p = M_dense[inv].to_sparse().coalesce()

    y = layer(x, walks, signs, M_v)
    y_p = layer(x_p, walks_p, signs, M_v_p)
    expected = y[inv]
    assert torch.allclose(y_p, expected, atol=1e-5), (
        f"permutation equivariance failed; max diff = "
        f"{(y_p - expected).abs().max().item():.2e}"
    )


def test_gradient_flow_on_every_position_and_branch():
    """Every position-aware weight in every sign branch must receive
    a gradient. Guards against the layer collapsing to a single bank."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=4,
                                use_sign_branching=True)
    layer = WalkConvLayer(cfg)
    # Construct walks that exercise BOTH sign branches.
    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=8, k_arity=4, n_walks=20, d_in=4, seed=7,
    )
    # Force at least 5 walks of each sign so both branches get rows.
    signs = torch.tensor(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        dtype=torch.int64,
    )
    y = layer(x, walks, signs, M_v)
    loss = y.pow(2).sum()
    loss.backward()

    # W has shape (2, k_arity, in, out). Each branch slice must
    # individually have non-zero gradient.
    grad_W = layer.W.grad
    assert grad_W is not None
    for b in range(2):
        for i in range(layer.k_arity):
            slice_grad = grad_W[b, i].abs().sum().item()
            assert slice_grad > 0, (
                f"branch {b} position {i} has zero gradient — "
                f"dead parameter slice"
            )


def test_no_sign_branching_mode():
    """use_sign_branching=False produces a layer with half the weight
    parameters and identical output regardless of sign input."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=3,
                                use_sign_branching=False)
    layer = WalkConvLayer(cfg).eval()
    # Param count: 1 × 3 × 4 × 4 + 1 × 4 = 48 + 4 = 52
    actual = sum(p.numel() for p in layer.parameters())
    assert actual == 52, f"expected 52 params, got {actual}"

    x, walks, signs, M_v = _build_walk_inputs(
        n_nodes=6, k_arity=3, n_walks=8, d_in=4, seed=1,
    )
    y_a = layer(x, walks, signs, M_v)
    y_b = layer(x, walks, -signs, M_v)
    assert torch.allclose(y_a, y_b, atol=1e-7), (
        "with use_sign_branching=False, output must be sign-invariant"
    )


def test_sbm_smoke():
    """End-to-end smoke on a small synthetic signed SBM: two
    communities, signed edges, walk-conv layer produces a usable
    embedding (non-trivial variance, no NaNs)."""
    cfg = HypergraphConvConfig(in_features=8, out_features=16, k_arity=3)
    layer = WalkConvLayer(cfg).eval()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 40
    # Two-community labelling.
    comm = rng.integers(0, 2, size=n)
    # Vertex features: small noise around the community indicator.
    x = torch.tensor(
        np.stack([comm, 1 - comm], axis=1).repeat(4, axis=1).astype(np.float32)
        + 0.1 * rng.standard_normal((n, 8)).astype(np.float32),
        dtype=torch.float32,
    )
    # Enumerate 50 random 3-walks.
    n_walks = 50
    walks = torch.tensor(rng.integers(0, n, size=(n_walks, 3)),
                          dtype=torch.long)
    # Sign = same-community on both edges → +1 else -1 (simulated σ-product).
    def walk_sign(w):
        e1_same = comm[w[0]] == comm[w[1]]
        e2_same = comm[w[1]] == comm[w[2]]
        return 1 if (e1_same == e2_same) else -1
    signs = torch.tensor(
        [walk_sign(walks[i].tolist()) for i in range(n_walks)],
        dtype=torch.int64,
    )
    # Build M_v.
    rows = walks.reshape(-1)
    cols = torch.arange(n_walks).repeat_interleave(3)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.full((rows.shape[0],), 1.0 / 3)
    M_v = torch.sparse_coo_tensor(
        indices, values, (n, n_walks),
    ).coalesce()

    y = layer(x, walks, signs, M_v)
    assert y.shape == (n, 16)
    assert not torch.isnan(y).any(), "output has NaNs"
    # Non-trivial variance — the layer is actually doing something.
    assert y.std().item() > 0.01, (
        f"output is nearly constant (std = {y.std().item():.3e}); "
        f"layer appears degenerate"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
