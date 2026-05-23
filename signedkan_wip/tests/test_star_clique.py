"""Tests for the star/clique sparse-tensor primitives.

Verifies the star-clique identity, the binary-signed-graph special
case, and the cycle-as-hyperedge construction.
"""
from __future__ import annotations

import torch

from signedkan_wip.src.hypergraph.star_clique import (
    StarTensor,
    cycle_pool_to_star,
    signed_graph_to_star,
    verify_star_clique_identity,
)


# ─── Signed-graph special case (binary, |e| = 2) ─────────────────────


def test_signed_graph_to_star_basic_shape():
    edges = torch.tensor([[0, 1], [1, 2], [2, 0]])
    signs = torch.tensor([+1.0, -1.0, +1.0])
    star = signed_graph_to_star(edges, signs, n_vertices=3)
    assert star.n_vertices == 3
    assert star.n_hyperedges == 3
    assert star.nnz == 6  # 2 arcs per edge × 3 edges
    # All hyperedge sizes are 2 (binary edges).
    assert torch.equal(star.edge_sizes, torch.full((3,), 2, dtype=torch.long))


def test_signed_graph_to_star_replicates_signs_per_arc():
    edges = torch.tensor([[0, 1]])
    signs = torch.tensor([-1.0])
    star = signed_graph_to_star(edges, signs, n_vertices=2)
    # Both arcs (vertex 0 to edge 0, vertex 1 to edge 0) carry -1.
    assert torch.equal(star.values, torch.tensor([-1.0, -1.0]))


def test_signed_graph_to_star_to_clique_adjacency_recovers_signed_adj():
    """For a signed graph, the product-clique adjacency reduces to the
    standard signed adjacency (with the |e|=2 normalisation: each edge
    contributes σ(e)^2 / 2 = 1/2 to its endpoints' adjacency)."""
    edges = torch.tensor([[0, 1], [1, 2]])
    signs = torch.tensor([+1.0, -1.0])
    star = signed_graph_to_star(edges, signs, n_vertices=3)
    A = star.to_clique_adjacency(include_self_loops=False)
    # Each edge contributes σ²/|e| = σ²/2 = 1/2 to the (u, v) and
    # (v, u) entries. So A[0,1] = A[1,0] = 0.5 (from edge 0), and
    # A[1,2] = A[2,1] = 0.5 (from edge 1). A[0,2] = A[2,0] = 0.
    assert abs(A[0, 1].item() - 0.5) < 1e-6
    assert abs(A[1, 0].item() - 0.5) < 1e-6
    assert abs(A[1, 2].item() - 0.5) < 1e-6
    assert abs(A[0, 2].item() - 0.0) < 1e-6
    # Diagonal is zeroed by default.
    assert torch.allclose(torch.diag(A), torch.zeros(3))


# ─── Cycle pool ──────────────────────────────────────────────────────


def test_cycle_pool_to_star_basic():
    cycles = [
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 3, 4]),
    ]
    signs = torch.tensor([+1.0, -1.0])
    star = cycle_pool_to_star(cycles, signs, n_vertices=5)
    assert star.n_hyperedges == 2
    assert star.nnz == 3 + 4  # k=3 + k=4 = 7
    assert torch.equal(star.edge_sizes, torch.tensor([3, 4], dtype=torch.long))


def test_cycle_pool_to_star_zero_cycles():
    star = cycle_pool_to_star([], torch.zeros(0), n_vertices=5)
    assert star.n_hyperedges == 0
    assert star.nnz == 0


def test_cycle_pool_clique_adjacency_for_triangle():
    """One triangle on {0, 1, 2} with σ = +1. Each pair (u, v) is in
    the triangle's vertex set, so A[u, v] = σ²/k = 1/3 for each of
    the 3 off-diagonal pairs."""
    cycles = [torch.tensor([0, 1, 2])]
    signs = torch.tensor([+1.0])
    star = cycle_pool_to_star(cycles, signs, n_vertices=3)
    A = star.to_clique_adjacency(include_self_loops=False)
    expected = 1.0 / 3.0
    assert abs(A[0, 1].item() - expected) < 1e-6
    assert abs(A[1, 2].item() - expected) < 1e-6
    assert abs(A[0, 2].item() - expected) < 1e-6
    # Symmetry.
    assert torch.allclose(A, A.t())


# ─── Star-clique identity verification ──────────────────────────────


def test_star_clique_identity_on_signed_graph():
    edges = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 3]])
    signs = torch.tensor([+1.0, -1.0, +1.0, -1.0])
    star = signed_graph_to_star(edges, signs, n_vertices=4)
    assert verify_star_clique_identity(star)


def test_star_clique_identity_on_cycle_pool():
    cycles = [
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([0, 2, 4]),
    ]
    signs = torch.tensor([+1.0, -1.0, +1.0])
    star = cycle_pool_to_star(cycles, signs, n_vertices=5)
    assert verify_star_clique_identity(star)


# ─── Degree counts ──────────────────────────────────────────────────


def test_degrees_signed_graph():
    edges = torch.tensor([[0, 1], [0, 2], [1, 2]])  # triangle
    signs = torch.tensor([+1.0, -1.0, +1.0])
    star = signed_graph_to_star(edges, signs, n_vertices=3)
    deg = star.degrees()
    assert torch.equal(deg, torch.tensor([2, 2, 2]))


def test_degrees_with_isolated_vertex():
    edges = torch.tensor([[0, 1]])
    signs = torch.tensor([+1.0])
    star = signed_graph_to_star(edges, signs, n_vertices=3)  # vertex 2 isolated
    deg = star.degrees()
    assert torch.equal(deg, torch.tensor([1, 1, 0]))


# ─── Sparse-COO interop ─────────────────────────────────────────────


def test_to_sparse_coo_shape():
    edges = torch.tensor([[0, 1], [1, 2]])
    signs = torch.tensor([+1.0, -1.0])
    star = signed_graph_to_star(edges, signs, n_vertices=3)
    coo = star.to_sparse_coo()
    assert coo.shape == (3, 2)
    assert coo.is_sparse
