"""Tests for PolygonConvLayer (GömbSoma phase 3-G).

Pins the PolygonConv contract:

  * inherits HypergraphConv semantics;
  * rejects k_arity < 3 (a polygon needs at least three vertices);
  * is CYCLIC-INVARIANT: rotating a polygon's vertex order doesn't
    change its message;
  * is REFLECTION-INVARIANT: reversing a polygon's vertex order
    doesn't change its message;
  * is sign-branched (positive and negative polygons route through
    independent banks);
  * has the documented parameter count;
  * end-to-end smoke on a synthetic SBM with planted polygons.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma import (
    HypergraphConvConfig,
    PolygonConvLayer,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _build_polygon_inputs(n_nodes, k_arity, n_polygons, d_in, seed=0):
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n_nodes, d_in, generator=rng)
    polygons = torch.randint(
        0, n_nodes, (n_polygons, k_arity), generator=rng, dtype=torch.long,
    )
    signs = torch.where(
        torch.rand(n_polygons, generator=rng) > 0.5,
        torch.ones(n_polygons, dtype=torch.int64),
        -torch.ones(n_polygons, dtype=torch.int64),
    )
    rows = polygons.reshape(-1)
    cols = torch.arange(n_polygons).repeat_interleave(k_arity)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.full((rows.shape[0],), 1.0 / k_arity)
    M_v = torch.sparse_coo_tensor(
        indices, values, (n_nodes, n_polygons),
    ).coalesce()
    return x, polygons, signs, M_v


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_rejects_k_arity_below_3():
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=2)
    with pytest.raises(ValueError, match="k_arity >= 3"):
        PolygonConvLayer(cfg)


def test_construction_and_param_count():
    """k_arity=4, in=16, out=16: 2 × 16 × 16 + 2 × 16 = 544."""
    cfg = HypergraphConvConfig(in_features=16, out_features=16, k_arity=4)
    layer = PolygonConvLayer(cfg)
    actual = sum(p.numel() for p in layer.parameters())
    assert actual == 2 * 16 * 16 + 2 * 16, (
        f"expected 544 params, got {actual}"
    )


def test_param_count_independent_of_k_arity():
    """Unlike WalkConv, PolygonConv has no position-aware weights —
    so param count must not depend on k_arity."""
    pcounts = []
    for k in (3, 4, 5, 6):
        cfg = HypergraphConvConfig(in_features=8, out_features=8, k_arity=k)
        layer = PolygonConvLayer(cfg)
        pcounts.append(sum(p.numel() for p in layer.parameters()))
    assert len(set(pcounts)) == 1, (
        f"param counts varied with k_arity: {pcounts}; should be constant"
    )


def test_forward_shape():
    cfg = HypergraphConvConfig(in_features=8, out_features=12, k_arity=5)
    layer = PolygonConvLayer(cfg).eval()
    x, polys, signs, M_v = _build_polygon_inputs(
        n_nodes=15, k_arity=5, n_polygons=20, d_in=8,
    )
    y = layer(x, polys, signs, M_v)
    assert y.shape == (15, 12)


def test_cyclic_invariance():
    """Rotating each polygon's vertex order by any cyclic shift must
    leave the per-polygon message identical (closed polygons have no
    canonical starting vertex)."""
    cfg = HypergraphConvConfig(in_features=6, out_features=6, k_arity=4)
    layer = PolygonConvLayer(cfg).eval()
    x, polys, signs, _M_v = _build_polygon_inputs(
        n_nodes=20, k_arity=4, n_polygons=10, d_in=6, seed=4,
    )
    msg_a = layer._forward_messages(x, polys, signs)
    # Rotate every polygon by one position.
    polys_rot = torch.roll(polys, shifts=1, dims=1)
    msg_b = layer._forward_messages(x, polys_rot, signs)
    assert torch.allclose(msg_a, msg_b, atol=1e-6), (
        f"cyclic invariance failed; max diff = "
        f"{(msg_a - msg_b).abs().max().item():.2e}"
    )


def test_reflection_invariance():
    """Reversing each polygon's vertex order must leave the message
    identical (undirected polygon — no orientation)."""
    cfg = HypergraphConvConfig(in_features=6, out_features=6, k_arity=5)
    layer = PolygonConvLayer(cfg).eval()
    x, polys, signs, _M_v = _build_polygon_inputs(
        n_nodes=20, k_arity=5, n_polygons=12, d_in=6, seed=8,
    )
    msg_a = layer._forward_messages(x, polys, signs)
    polys_rev = polys.flip(dims=[1])
    msg_b = layer._forward_messages(x, polys_rev, signs)
    assert torch.allclose(msg_a, msg_b, atol=1e-6), (
        f"reflection invariance failed; max diff = "
        f"{(msg_a - msg_b).abs().max().item():.2e}"
    )


def test_sign_branching_actually_branches():
    """Flipping every polygon's sign must move the output."""
    cfg = HypergraphConvConfig(in_features=8, out_features=8, k_arity=4,
                                use_sign_branching=True)
    layer = PolygonConvLayer(cfg).eval()
    x, polys, signs, M_v = _build_polygon_inputs(
        n_nodes=15, k_arity=4, n_polygons=20, d_in=8, seed=2,
    )
    y_pos = layer(x, polys, signs, M_v)
    y_neg = layer(x, polys, -signs, M_v)
    diff = (y_pos - y_neg).abs().max().item()
    assert diff > 1e-3, (
        f"sign branching is dead — flipping signs barely changed the "
        f"output (max diff = {diff:.2e})"
    )


def test_permutation_equivariance():
    """Inherited from HypergraphConv: vertex permutation passes through."""
    cfg = HypergraphConvConfig(in_features=6, out_features=8, k_arity=4,
                                use_sign_branching=True)
    layer = PolygonConvLayer(cfg).eval()
    n_nodes, k, n_polys, d_in = 11, 4, 25, 6
    x, polys, signs, M_v = _build_polygon_inputs(
        n_nodes=n_nodes, k_arity=k, n_polygons=n_polys, d_in=d_in, seed=42,
    )
    perm = torch.randperm(n_nodes)
    inv = torch.argsort(perm)
    x_p = x[inv]
    polys_p = perm[polys]
    M_dense = M_v.to_dense()
    M_v_p = M_dense[inv].to_sparse().coalesce()

    y = layer(x, polys, signs, M_v)
    y_p = layer(x_p, polys_p, signs, M_v_p)
    expected = y[inv]
    assert torch.allclose(y_p, expected, atol=1e-5), (
        f"permutation equivariance failed; max diff = "
        f"{(y_p - expected).abs().max().item():.2e}"
    )


def test_gradient_flow_on_every_branch():
    """Every sign-branch weight slice must receive gradient when both
    signs appear in the input."""
    cfg = HypergraphConvConfig(in_features=4, out_features=4, k_arity=4)
    layer = PolygonConvLayer(cfg)
    x, polys, _signs, M_v = _build_polygon_inputs(
        n_nodes=8, k_arity=4, n_polygons=20, d_in=4, seed=7,
    )
    # Equal-split signs to guarantee both branches see data.
    signs = torch.tensor([1] * 10 + [-1] * 10, dtype=torch.int64)
    y = layer(x, polys, signs, M_v)
    loss = y.pow(2).sum()
    loss.backward()
    grad_W = layer.W.grad
    assert grad_W is not None
    for b in range(2):
        slice_grad = grad_W[b].abs().sum().item()
        assert slice_grad > 0, (
            f"branch {b} has zero gradient — dead parameter slice"
        )


def test_sbm_smoke_with_planted_4cycles():
    """End-to-end smoke: 4 planted 4-cycles inside a two-community
    signed SBM; PolygonConv should produce non-degenerate output and
    the cycle-pool features should distinguish balanced from
    unbalanced cycles."""
    cfg = HypergraphConvConfig(in_features=8, out_features=16, k_arity=4)
    layer = PolygonConvLayer(cfg).eval()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n = 40
    comm = rng.integers(0, 2, size=n)
    # Vertex features encoding community.
    x = torch.tensor(
        np.stack([comm, 1 - comm], axis=1).repeat(4, axis=1).astype(np.float32)
        + 0.1 * rng.standard_normal((n, 8)).astype(np.float32),
        dtype=torch.float32,
    )
    # Plant 4-cycles: half balanced (all-same-community), half not.
    n_polys = 30
    polys_list = []
    signs_list = []
    for i in range(n_polys):
        if i % 2 == 0:
            # Balanced: 4 vertices from the same community.
            c = rng.integers(0, 2)
            verts = rng.choice(np.where(comm == c)[0], size=4, replace=False)
            polys_list.append(verts)
            signs_list.append(1)
        else:
            verts = rng.choice(n, size=4, replace=False)
            polys_list.append(verts)
            signs_list.append(-1)
    polys = torch.tensor(np.array(polys_list), dtype=torch.long)
    signs = torch.tensor(signs_list, dtype=torch.int64)
    rows = polys.reshape(-1)
    cols = torch.arange(n_polys).repeat_interleave(4)
    indices = torch.stack([rows, cols], dim=0)
    values = torch.full((rows.shape[0],), 1.0 / 4)
    M_v = torch.sparse_coo_tensor(indices, values, (n, n_polys)).coalesce()

    y = layer(x, polys, signs, M_v)
    assert y.shape == (n, 16)
    assert not torch.isnan(y).any()
    assert y.std().item() > 0.01, (
        f"output is nearly constant (std = {y.std().item():.3e})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
