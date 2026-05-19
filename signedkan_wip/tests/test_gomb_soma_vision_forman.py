"""Tests for FormanCurvatureHead (Ricci-Stim phase 1).

Pins Forman-Ricci curvature against known graph-theoretic invariants:

  * triangle K₃: every edge has κ = 0 (degree-3 face-balanced)
  * complete graphs Kₙ: every edge has κ = 0
  * cycle Cₙ (n ≥ 4): every edge has κ = -2 (no triangle shortcuts)
  * path Pₙ: middle edges κ = -2, endpoint-incident edges κ = -1
  * star Sₙ: every edge has κ = 1 - n (severe bottleneck)
  * 4-connected grid: κ = -6 interior, -5 / -4 / -3 on boundaries

Also tests the module's contract (preconditions, vertex aggregation,
empty graph, isolated vertices).
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import (
    FormanCurvature,
    FormanCurvatureHead,
)


# ---------------------------------------------------------------------
# Standard test graphs
# ---------------------------------------------------------------------


def _triangle_edges() -> torch.Tensor:
    return torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long)


def _Kn_edges(n: int) -> torch.Tensor:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return torch.tensor(edges, dtype=torch.long)


def _Cn_edges(n: int) -> torch.Tensor:
    return torch.tensor(
        [(i, (i + 1) % n) for i in range(n)],
        dtype=torch.long,
    )


def _Pn_edges(n: int) -> torch.Tensor:
    return torch.tensor(
        [(i, i + 1) for i in range(n - 1)],
        dtype=torch.long,
    )


def _Sn_edges(n_leaves: int) -> torch.Tensor:
    """Star with one centre (vertex 0) and ``n_leaves`` leaves."""
    return torch.tensor(
        [(0, i + 1) for i in range(n_leaves)],
        dtype=torch.long,
    )


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_triangle_K3_all_zero():
    """K₃: deg = 2 each, every edge incident on exactly 1 triangle.
    κ = 2 − 2 − 2 + 2·1 = 0."""
    head = FormanCurvatureHead()
    out = head(_triangle_edges(), n_nodes=3)
    assert torch.allclose(out.edge_kappa, torch.zeros(3))
    # Vertex κ is mean of incident edge κs — also 0.
    assert torch.allclose(out.vertex_kappa, torch.zeros(3))


def test_complete_graph_K4_all_zero():
    """K₄: deg = 3, each edge in 2 triangles. κ = 2 − 3 − 3 + 4 = 0."""
    head = FormanCurvatureHead()
    out = head(_Kn_edges(4), n_nodes=4)
    assert torch.allclose(out.edge_kappa, torch.zeros(6))


def test_complete_graph_K5_all_zero():
    """K₅: deg = 4, each edge in 3 triangles. κ = 2 − 4 − 4 + 6 = 0."""
    head = FormanCurvatureHead()
    out = head(_Kn_edges(5), n_nodes=5)
    assert torch.allclose(out.edge_kappa, torch.zeros(10))


def test_cycle_C4_all_minus_two():
    """C₄: deg = 2, no triangles. κ = 2 − 2 − 2 + 0 = -2."""
    head = FormanCurvatureHead()
    out = head(_Cn_edges(4), n_nodes=4)
    assert torch.allclose(out.edge_kappa, torch.full((4,), -2.0))
    assert torch.allclose(out.vertex_kappa, torch.full((4,), -2.0))


def test_cycle_C6_all_minus_two():
    head = FormanCurvatureHead()
    out = head(_Cn_edges(6), n_nodes=6)
    assert torch.allclose(out.edge_kappa, torch.full((6,), -2.0))


def test_path_P4_endpoint_vs_interior():
    """P₄ (0–1–2–3): 3 edges, no triangles.
    Edge (0,1): deg 1 + 2 → κ = 2 − 1 − 2 = -1.
    Edge (1,2): deg 2 + 2 → κ = 2 − 2 − 2 = -2.
    Edge (2,3): deg 2 + 1 → κ = -1."""
    head = FormanCurvatureHead()
    out = head(_Pn_edges(4), n_nodes=4)
    expected = torch.tensor([-1.0, -2.0, -1.0])
    assert torch.allclose(out.edge_kappa, expected)


def test_star_S5_all_minus_4():
    """S₅: centre has deg 5, leaves have deg 1.
    Every edge: κ = 2 − 5 − 1 = -4."""
    head = FormanCurvatureHead()
    out = head(_Sn_edges(5), n_nodes=6)  # 1 centre + 5 leaves
    assert torch.allclose(out.edge_kappa, torch.full((5,), -4.0))


def test_isolated_vertices_have_zero_kappa():
    """A graph with disconnected vertices: those vertices have κ_v = 0."""
    edges = _Cn_edges(3)  # K3 on vertices {0,1,2}; vertex 3 isolated
    head = FormanCurvatureHead()
    out = head(edges, n_nodes=4)
    assert out.vertex_kappa[3].item() == 0.0
    assert out.degree[3].item() == 0


def test_self_loops_dropped():
    """A self-loop (u, u) is silently ignored."""
    edges = torch.tensor(
        [[0, 1], [1, 1], [1, 2], [2, 0]], dtype=torch.long
    )
    head = FormanCurvatureHead()
    out = head(edges, n_nodes=3)
    # The self-loop's κ should be 0 (it was dropped).
    assert out.edge_kappa[1].item() == 0.0
    # Remaining edges form a triangle → κ = 0 each.
    assert out.edge_kappa[0].item() == 0.0
    assert out.edge_kappa[2].item() == 0.0
    assert out.edge_kappa[3].item() == 0.0


def test_empty_graph():
    """Zero-edge input returns zero-tensors of the right shape."""
    head = FormanCurvatureHead()
    out = head(torch.zeros((0, 2), dtype=torch.long), n_nodes=5)
    assert out.edge_kappa.shape == (0,)
    assert out.vertex_kappa.shape == (5,)
    assert out.vertex_kappa.abs().sum().item() == 0.0


def test_rejects_wrong_edge_shape():
    head = FormanCurvatureHead()
    with pytest.raises(ValueError, match="edges must have shape"):
        head(torch.zeros((5,), dtype=torch.long), n_nodes=3)


def test_undirected_handling():
    """Passing both (u, v) and (v, u) should not double-count degree."""
    edges = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 0], [0, 2]],
        dtype=torch.long,
    )
    head = FormanCurvatureHead(treat_as_undirected=True)
    out = head(edges, n_nodes=3)
    # Underlying graph is K₃: every edge κ should be 0.
    assert torch.allclose(out.edge_kappa, torch.zeros(6))


def test_4_connected_grid_3x3():
    """3×3 grid of 4-connected patches.
    Vertex (1,1) (centre): deg = 4. Border vertices deg 3. Corners deg 2.
    No triangles (4-conn grid is triangle-free).
    Centre–border edge: κ = 2 − 4 − 3 = -5.
    Border–corner edge:  κ = 2 − 3 − 2 = -3.
    Border–border edge: κ = 2 − 3 − 3 = -4 (only along the middle row/column).

    For 3×3 there's no centre-centre edge (single centre vertex).
    """
    h, w = 3, 3
    n = h * w
    def idx(r, c):
        return r * w + c
    edges = []
    for r in range(h):
        for c in range(w):
            if c + 1 < w:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < h:
                edges.append((idx(r, c), idx(r + 1, c)))
    e = torch.tensor(edges, dtype=torch.long)
    head = FormanCurvatureHead()
    out = head(e, n_nodes=n)

    # Verify expected values for known edges.
    def find_edge_kappa(u, v):
        for i, (a, b) in enumerate(edges):
            if (a, b) == (u, v) or (a, b) == (v, u):
                return out.edge_kappa[i].item()
        raise KeyError(f"({u},{v}) not in edge list")

    # centre is index 4, top-left corner is 0.
    # Edge corner(0) ↔ side(1): deg 2 + deg 3 → κ = -3
    assert find_edge_kappa(0, 1) == -3.0
    # Edge side(1) ↔ centre(4): deg 3 + deg 4 → κ = -5
    assert find_edge_kappa(1, 4) == -5.0


def test_vertex_kappa_is_mean_of_incident():
    """Verified by construction. P₄: edges κ = [-1, -2, -1].
    Vertex 0 has only edge (0,1) → vertex_kappa[0] = -1.
    Vertex 1 has edges (0,1) and (1,2) → mean(-1, -2) = -1.5.
    Vertex 2 has edges (1,2) and (2,3) → mean(-2, -1) = -1.5.
    Vertex 3 has only edge (2,3) → vertex_kappa[3] = -1.
    """
    head = FormanCurvatureHead()
    out = head(_Pn_edges(4), n_nodes=4)
    expected_vk = torch.tensor([-1.0, -1.5, -1.5, -1.0])
    assert torch.allclose(out.vertex_kappa, expected_vk)


def test_dataclass_fields_present():
    head = FormanCurvatureHead()
    out = head(_triangle_edges(), n_nodes=3)
    assert isinstance(out, FormanCurvature)
    assert hasattr(out, "edge_kappa")
    assert hasattr(out, "vertex_kappa")
    assert hasattr(out, "degree")
    assert hasattr(out, "triangle_count")


def test_no_learnable_parameters():
    """The Forman head is a deterministic graph functional — no params."""
    head = FormanCurvatureHead()
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
