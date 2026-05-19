"""Tests for StimulusGraphBuilder (Ricci-Stim phase 5)."""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import (
    AdaptiveQuadtree,
    AnchorTree,
    StimulusGraph,
    StimulusGraphBuilder,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _flat_tree(n_h: int, n_w: int, size: int = 4) -> AnchorTree:
    """A uniform flat anchor grid at scale 0 only."""
    positions = []
    sizes = []
    scales = []
    parents = []
    for r in range(n_h):
        for c in range(n_w):
            positions.append([r * size, c * size])
            sizes.append(size)
            scales.append(0)
            parents.append(-1)
    return AnchorTree(
        positions=torch.tensor(positions, dtype=torch.long),
        sizes=torch.tensor(sizes, dtype=torch.long),
        scales=torch.tensor(scales, dtype=torch.long),
        parent_indices=torch.tensor(parents, dtype=torch.long),
    )


def _multiscale_tree() -> AnchorTree:
    """4 scale-0 anchors (2×2 grid of size 8), one of which has 4 children at scale 1."""
    # scale-0 anchors (0..3) in 2×2 grid at positions (0,0), (0,8), (8,0), (8,8) size 8
    pos = [[0, 0], [0, 8], [8, 0], [8, 8]]
    siz = [8, 8, 8, 8]
    sca = [0, 0, 0, 0]
    par = [-1, -1, -1, -1]
    # Subdivide anchor 0 into 4 scale-1 children at positions (0,0), (0,4), (4,0), (4,4) size 4
    for r in (0, 4):
        for c in (0, 4):
            pos.append([r, c])
            siz.append(4)
            sca.append(1)
            par.append(0)
    return AnchorTree(
        positions=torch.tensor(pos, dtype=torch.long),
        sizes=torch.tensor(siz, dtype=torch.long),
        scales=torch.tensor(sca, dtype=torch.long),
        parent_indices=torch.tensor(par, dtype=torch.long),
    )


# ---------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------


def test_same_scale_edges_on_flat_grid():
    """A 3×3 same-scale grid has 12 undirected 4-conn edges."""
    tree = _flat_tree(3, 3, size=4)
    features = torch.ones(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    # Expected: 6 horizontal + 6 vertical = 12 undirected edges.
    assert sg.edges.shape[0] == 12, (
        f"expected 12 same-scale edges on 3×3 grid, got {sg.edges.shape[0]}"
    )


def test_cross_scale_edges_added():
    """A multiscale tree contributes parent-child edges over and above same-scale."""
    tree = _multiscale_tree()
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    # Same-scale: 4 scale-0 anchors form a 2×2 grid → 4 edges.
    #             4 scale-1 children form a 2×2 grid → 4 edges.
    # Cross-scale: 4 children × 1 parent = 4 edges.
    # Total: 12.
    assert sg.edges.shape[0] == 12, (
        f"expected 12 total edges (4 + 4 + 4), got {sg.edges.shape[0]}"
    )


def test_edge_signs_positive_when_features_aligned():
    """When all features are positive constants, every inner product
    is positive, so every edge sign is +1."""
    tree = _flat_tree(2, 2, size=4)
    features = torch.ones(tree.n_anchors, 3)
    builder = StimulusGraphBuilder(sign_threshold=0.0)
    sg = builder(tree, features)
    assert (sg.edge_signs == +1).all()


def test_edge_signs_negative_when_features_anti_aligned():
    """Features [+v, -v, +v, -v] in a row: adjacent features have
    negative inner product → σ = -1."""
    tree = _flat_tree(1, 4, size=4)  # 4 anchors in a row
    features = torch.tensor([
        [+1.0, 0.0],
        [-1.0, 0.0],
        [+1.0, 0.0],
        [-1.0, 0.0],
    ])
    builder = StimulusGraphBuilder(sign_threshold=0.0)
    sg = builder(tree, features)
    # 3 horizontal edges, all anti-aligned → σ = -1.
    assert (sg.edge_signs == -1).all()


# ---------------------------------------------------------------------
# Primitive enumeration
# ---------------------------------------------------------------------


def test_walks_have_length_3():
    tree = _flat_tree(2, 2, size=4)
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    # Each walk has 3 vertices.
    assert sg.walks.shape[1] == 3


def test_walks_no_backtracking():
    tree = _flat_tree(3, 3, size=4)
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    starts = sg.walks[:, 0]
    ends = sg.walks[:, 2]
    assert (starts != ends).all()


def test_polygons_are_4cycles_in_grid():
    """A 2×2 grid has exactly one 4-cycle plaquette."""
    tree = _flat_tree(2, 2, size=4)
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    assert sg.polygons.shape == (1, 4), (
        f"expected one 4-cycle on a 2×2 grid, got {tuple(sg.polygons.shape)}"
    )


def test_no_triangles_in_pure_flat_grid():
    """A pure 4-connected same-scale grid is triangle-free."""
    tree = _flat_tree(3, 3, size=4)
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    assert sg.triangles.shape[0] == 0


def test_triangles_exist_when_cross_scale_edges():
    """A multiscale tree creates triangles (parent + two adjacent children)."""
    tree = _multiscale_tree()
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    # Each child of anchor 0 has cross-scale edge to anchor 0.
    # Adjacent siblings (children) share a same-scale edge.
    # So (parent=0, child_i, child_j) is a triangle for each adjacent
    # sibling pair (i,j). The 2×2 child grid has 4 adjacent sibling pairs.
    assert sg.triangles.shape[0] == 4, (
        f"expected 4 triangles from quadtree subdivision, got "
        f"{sg.triangles.shape[0]}"
    )


def test_walk_sign_is_sigma_product_of_edges():
    """Walk sign = sign(edge_0) × sign(edge_1)."""
    tree = _flat_tree(2, 2, size=4)
    # Pattern of features that gives mixed-sign edges.
    features = torch.tensor([
        [+1.0, 0.0],
        [-1.0, 0.0],
        [+1.0, 0.0],
        [-1.0, 0.0],
    ])
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    # For each walk (a, b, c), walk_sign should equal edge_sign((a,b)) × edge_sign((b,c)).
    # Reconstruct via the per-edge tensor.
    for w_idx, w in enumerate(sg.walks.tolist()):
        a, b, c = w
        # Find edges (a, b) and (b, c) in sg.edges.
        e1 = None
        e2 = None
        for i, (u, v) in enumerate(sg.edges.tolist()):
            if (u, v) == (a, b) or (u, v) == (b, a):
                e1 = i
            if (u, v) == (b, c) or (u, v) == (c, b):
                e2 = i
        assert e1 is not None and e2 is not None
        expected_sign = sg.edge_signs[e1] * sg.edge_signs[e2]
        assert sg.walk_signs[w_idx] == expected_sign


# ---------------------------------------------------------------------
# Incidence
# ---------------------------------------------------------------------


def test_M_v_walks_shape():
    tree = _flat_tree(2, 2, size=4)
    features = torch.ones(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    assert sg.M_v_walks.shape == (tree.n_anchors, sg.walks.shape[0])


def test_M_v_polygons_uniform_weight():
    tree = _flat_tree(2, 2, size=4)
    features = torch.ones(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    vals = sg.M_v_polygons.coalesce().values()
    if vals.numel() > 0:
        # Each polygon has 4 vertices → weight 1/4.
        assert torch.allclose(vals, torch.full_like(vals, 0.25))


# ---------------------------------------------------------------------
# Curvature and Hodge
# ---------------------------------------------------------------------


def test_edge_curvature_shape():
    tree = _flat_tree(3, 3, size=4)
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    assert sg.edge_curvatures.shape == (sg.edges.shape[0],)


def test_walk_curvature_is_mean_of_edges():
    """walk_curvature = mean of constituent edges' κ."""
    tree = _flat_tree(2, 2, size=4)
    features = torch.ones(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    for w_idx, w in enumerate(sg.walks.tolist()):
        a, b, c = w
        e1 = None
        e2 = None
        for i, (u, v) in enumerate(sg.edges.tolist()):
            if (u, v) == (a, b) or (u, v) == (b, a):
                e1 = i
            if (u, v) == (b, c) or (u, v) == (c, b):
                e2 = i
        expected = 0.5 * (sg.edge_curvatures[e1] + sg.edge_curvatures[e2])
        assert torch.allclose(
            sg.walk_curvatures[w_idx], expected, atol=1e-6,
        )


def test_hodge_laplacian_shape():
    tree = _flat_tree(2, 2, size=4)
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    assert sg.hodge_laplacian_0.shape == (4, 4)
    assert sg.hodge_laplacian_0.is_sparse


# ---------------------------------------------------------------------
# Determinism / budgets / API
# ---------------------------------------------------------------------


def test_determinism():
    tree = _multiscale_tree()
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    a = builder(tree, features)
    b = builder(tree, features)
    assert torch.equal(a.edges, b.edges)
    assert torch.equal(a.edge_signs, b.edge_signs)
    assert torch.equal(a.walks, b.walks)
    assert torch.equal(a.walk_signs, b.walk_signs)
    assert torch.equal(a.polygons, b.polygons)
    assert torch.equal(a.triangles, b.triangles)


def test_walks_respect_budget():
    tree = _flat_tree(4, 4, size=4)
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder(max_walks=20)
    sg = builder(tree, features)
    assert sg.walks.shape[0] <= 20


def test_output_is_stimulus_graph():
    tree = _flat_tree(2, 2, size=4)
    features = torch.randn(tree.n_anchors, 4)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    assert isinstance(sg, StimulusGraph)
    assert sg.n_anchors == 4


def test_rejects_mismatched_features():
    tree = _flat_tree(2, 2, size=4)
    bad_features = torch.randn(99, 4)
    builder = StimulusGraphBuilder()
    with pytest.raises(ValueError, match="anchor_features"):
        builder(tree, bad_features)


def test_integration_with_quadtree():
    """End-to-end smoke: AdaptiveQuadtree → AnchorTree → StimulusGraph."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=4, max_depth=2, max_anchors=64,
        score_threshold=0.05,
    )
    torch.manual_seed(0)
    img = torch.rand(1, 32, 32)
    tree = qt(img)
    n = tree.n_anchors
    features = torch.randn(n, 6)
    builder = StimulusGraphBuilder()
    sg = builder(tree, features)
    assert sg.n_anchors == n
    assert sg.edges.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
