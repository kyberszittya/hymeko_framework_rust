"""Tests for PatchGraphBuilder (GömbSoma vision phase 3-V)."""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import PatchGraphBuilder


def test_grid_dimensions_for_mnist():
    """28×28 image with patch=4 → 7×7 = 49 patches."""
    b = PatchGraphBuilder(image_h=28, image_w=28, patch_size=4)
    assert b.h_patches == 7 and b.w_patches == 7
    assert b.n_patches == 49


def test_edge_count_for_grid():
    """7×7 grid, 4-connected, directed edges: each interior vertex has
    4 neighbours; border has 3; corner has 2. Total directed edges:
    2 × n_undirected. For a 7×7 grid that's 2 × (6×7 + 7×6) = 168."""
    b = PatchGraphBuilder(image_h=28, image_w=28, patch_size=4)
    expected_directed_edges = 2 * (6 * 7 + 7 * 6)  # 168
    assert b.edges.shape == (expected_directed_edges, 2)


def test_walks_are_length_2_no_backtracking():
    """All walks must have shape (n_walks, 3) and no a == c."""
    b = PatchGraphBuilder(image_h=12, image_w=12, patch_size=4)  # 3×3 grid
    assert b.walks.shape[1] == 3
    starts = b.walks[:, 0]
    ends = b.walks[:, 2]
    assert (starts != ends).all(), "found a backtracking walk"


def test_rejects_unaligned_image_size():
    with pytest.raises(ValueError, match="must be divisible"):
        PatchGraphBuilder(image_h=30, image_w=28, patch_size=4)


def test_rejects_bad_patch_size():
    with pytest.raises(ValueError, match="patch_size must be"):
        PatchGraphBuilder(image_h=28, image_w=28, patch_size=0)


def test_patchify_shape():
    """An MNIST-like (1, 28, 28) image patchifies to (49, 16)."""
    b = PatchGraphBuilder(image_h=28, image_w=28, patch_size=4)
    img = torch.randn(1, 28, 28)
    patches = b.patchify(img)
    assert patches.shape == (49, 16), f"got {tuple(patches.shape)}"


def test_patchify_preserves_content():
    """A patch in the upper-left corner contains the top-left pixels."""
    b = PatchGraphBuilder(image_h=8, image_w=8, patch_size=4)
    img = torch.zeros(1, 8, 8)
    # Mark top-left pixel.
    img[0, 0, 0] = 99.0
    patches = b.patchify(img)
    # Top-left patch is index 0; it should contain 99 in its first pixel.
    assert patches[0, 0].item() == 99.0


def test_edge_signs_polarity():
    """A uniform-gradient image has all-same-sign edges in the gradient
    direction. Build an image whose mean brightness strictly increases
    left-to-right: all rightward edges have src brighter? No — src
    brightness < dst, so σ = −1 for rightward, +1 for leftward."""
    b = PatchGraphBuilder(image_h=8, image_w=8, patch_size=4)  # 2×2 grid
    img = torch.zeros(1, 8, 8)
    # Make right column brighter.
    img[0, :, 4:] = 1.0
    patches = b.patchify(img)
    signs = b.edge_signs(patches)
    # Edges in the directed list: enumerate and find a left-to-right edge.
    # Patch 0 (top-left) has mean 0; patch 1 (top-right) has mean 1.
    # Find the (0, 1) edge.
    for i, (s, d) in enumerate(b.edges.tolist()):
        if s == 0 and d == 1:
            assert signs[i].item() == -1, (
                "left→right edge between dark and bright should have σ = -1"
            )
        if s == 1 and d == 0:
            assert signs[i].item() == +1, (
                "right→left edge bright→dark should have σ = +1"
            )


def test_walk_signs_compose_edge_signs():
    """walk sign = product of two constituent edge signs. Construct a
    walk and verify."""
    b = PatchGraphBuilder(image_h=8, image_w=8, patch_size=4)
    img = torch.randn(1, 8, 8)
    patches = b.patchify(img)
    e_signs = b.edge_signs(patches)
    w_signs = b.walk_signs(e_signs)
    # Spot-check: for walk w, w_signs[w] == e_signs[edge1] * e_signs[edge2].
    for w in range(min(5, b.walks.shape[0])):
        e1, e2 = b.walk_edge_idx[w].tolist()
        expected = e_signs[e1].item() * e_signs[e2].item()
        assert w_signs[w].item() == expected, (
            f"walk {w}: expected σ-product {expected}, got {w_signs[w].item()}"
        )


def test_M_v_shape():
    """M_v has shape (n_patches, n_walks) and entries 1/3."""
    b = PatchGraphBuilder(image_h=12, image_w=12, patch_size=4)
    assert b.M_v.shape == (b.n_patches, b.walks.shape[0])
    values = b.M_v.coalesce().values()
    assert torch.allclose(values, torch.full_like(values, 1.0 / 3.0))


def test_encode_one_shot():
    """encode() returns all four components consistently."""
    b = PatchGraphBuilder(image_h=12, image_w=12, patch_size=4)
    img = torch.randn(1, 12, 12)
    patches, walks, walk_signs, M_v = b.encode(img)
    assert patches.shape[0] == b.n_patches
    assert walks.shape == b.walks.shape
    assert walk_signs.shape == (walks.shape[0],)
    assert M_v.shape == (b.n_patches, walks.shape[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
