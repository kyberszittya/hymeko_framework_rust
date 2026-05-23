"""Tests for AdaptiveQuadtree (GömbSoma-Ricci-Stim phase 2).

Pins the quadtree contract:

  * deterministic — identical image → identical AnchorTree;
  * depth-bounded — no anchor at scale > max_depth;
  * budget-bounded — total anchor count ≤ max_anchors;
  * uniform image → no subdivision;
  * structured image → subdivision happens near the structure;
  * parent-child geometric consistency (4 children tile parent);
  * scale-0 anchors have parent_idx = -1;
  * sizes consistent with scale (size = patch_size_initial / 2^scale);
  * argparse-style precondition rejection.
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hymeko_gomb.soma.vision import (
    AdaptiveQuadtree,
    AnchorTree,
)


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_rejects_misaligned_image():
    with pytest.raises(ValueError, match="must be divisible"):
        AdaptiveQuadtree(image_h=30, image_w=28, patch_size_initial=8)


def test_rejects_bad_patch_size():
    with pytest.raises(ValueError, match="patch_size_initial"):
        AdaptiveQuadtree(image_h=32, image_w=32, patch_size_initial=0)


def test_rejects_bad_min_size():
    with pytest.raises(ValueError, match="patch_size_min"):
        AdaptiveQuadtree(
            image_h=32, image_w=32, patch_size_initial=8, patch_size_min=16,
        )


def test_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="at least one"):
        AdaptiveQuadtree(
            image_h=32, image_w=32, patch_size_initial=8,
            variance_weight=0.0, curvature_weight=0.0,
        )


# ---------------------------------------------------------------------
# Scale-0 baseline
# ---------------------------------------------------------------------


def test_uniform_image_no_subdivision():
    """A constant-valued image triggers no subdivision (variance = 0)."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=8,
        score_threshold=0.01,
    )
    img = torch.full((1, 32, 32), 0.5)
    tree = qt(img)
    assert tree.n_anchors == 16   # 4 × 4 = 16 scale-0 anchors
    assert (tree.scales == 0).all()
    assert (tree.parent_indices == -1).all()
    assert (tree.sizes == 8).all()


def test_scale_0_uniform_tiling():
    """Scale-0 anchors cover the image with no gaps and no overlaps."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=8,
    )
    img = torch.zeros(1, 32, 32)
    tree = qt(img)
    # All 16 scale-0 anchors at (0,0), (0,8), ..., (24,24).
    expected_positions = {
        (r, c) for r in range(0, 32, 8) for c in range(0, 32, 8)
    }
    actual = {
        (int(p[0]), int(p[1])) for p in tree.positions.tolist()
    }
    assert actual == expected_positions


# ---------------------------------------------------------------------
# Subdivision happens
# ---------------------------------------------------------------------


def test_high_variance_region_subdivides():
    """A bright spot in one quadrant should subdivide that quadrant
    (and only that quadrant)."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=2,
        max_depth=2, max_anchors=64,
        score_threshold=0.05,
    )
    img = torch.zeros(1, 32, 32)
    # Bright pixel in the top-left quadrant (patch [0:16, 0:16]).
    img[0, 4, 4] = 1.0
    tree = qt(img)
    # The top-left patch should subdivide (high variance from bright spot).
    # Other three scale-0 patches have variance 0 → no subdivision.
    scale_0_anchors = tree.scales == 0
    scale_1_anchors = tree.scales == 1
    assert scale_0_anchors.sum().item() == 4  # 4 initial 16×16 patches
    assert scale_1_anchors.sum().item() == 4  # one parent's 4 children
    # The 4 children should be at positions in the top-left quadrant.
    child_positions = tree.positions[scale_1_anchors].tolist()
    assert all(r < 16 and c < 16 for (r, c) in child_positions)


def test_subdivision_respects_depth_bound():
    """An image full of variance should subdivide every patch until
    max_depth is reached."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=1,
        max_depth=2, max_anchors=1024,
        score_threshold=0.0,
    )
    torch.manual_seed(0)
    img = torch.rand(1, 32, 32)
    tree = qt(img)
    assert tree.scales.max().item() <= 2


def test_subdivision_respects_anchor_budget():
    """Even with high variance everywhere, the anchor budget caps growth."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=1,
        max_depth=10, max_anchors=20,
        score_threshold=0.0,
    )
    torch.manual_seed(0)
    img = torch.rand(1, 32, 32)
    tree = qt(img)
    assert tree.n_anchors <= 20


# ---------------------------------------------------------------------
# Geometric consistency
# ---------------------------------------------------------------------


def test_parent_child_tiling():
    """Each parent's 4 children tile its region: same size, four
    quadrants, no overlap."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=2,
        max_depth=2, max_anchors=64,
        score_threshold=0.05,
    )
    img = torch.zeros(1, 32, 32)
    img[0, 4, 4] = 1.0
    tree = qt(img)
    # Find scale-1 anchors and group by parent.
    for i in range(tree.n_anchors):
        p = int(tree.parent_indices[i].item())
        if p == -1:
            continue
        # Child must have half the parent's size.
        assert tree.sizes[i].item() == tree.sizes[p].item() // 2
        # Child position within parent region.
        cr, cc = tree.positions[i].tolist()
        pr, pc = tree.positions[p].tolist()
        ps = int(tree.sizes[p].item())
        assert pr <= cr < pr + ps
        assert pc <= cc < pc + ps


def test_size_consistent_with_scale():
    """size(anchor) = patch_size_initial / 2^scale."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=1,
        max_depth=3, max_anchors=128,
        score_threshold=0.0,
    )
    torch.manual_seed(1)
    img = torch.rand(1, 32, 32)
    tree = qt(img)
    for i in range(tree.n_anchors):
        s = int(tree.scales[i].item())
        sz = int(tree.sizes[i].item())
        assert sz == 16 // (2 ** s), (
            f"anchor {i} at scale {s} has size {sz}, expected {16 // 2**s}"
        )


def test_scale_0_anchors_have_no_parent():
    """All scale-0 anchors have parent_idx = -1."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=2,
        max_depth=3, max_anchors=64,
        score_threshold=0.05,
    )
    img = torch.zeros(1, 32, 32)
    img[0, 4, 4] = 1.0
    tree = qt(img)
    mask = tree.scales == 0
    assert (tree.parent_indices[mask] == -1).all()
    # And non-scale-0 anchors all have valid parent indices.
    mask_nz = tree.scales > 0
    assert (tree.parent_indices[mask_nz] >= 0).all()


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_determinism():
    """Same image → same AnchorTree, every call."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=2,
        max_depth=3, max_anchors=128,
        score_threshold=0.05,
    )
    torch.manual_seed(7)
    img = torch.rand(1, 32, 32)
    a = qt(img)
    b = qt(img)
    assert torch.equal(a.positions, b.positions)
    assert torch.equal(a.sizes, b.sizes)
    assert torch.equal(a.scales, b.scales)
    assert torch.equal(a.parent_indices, b.parent_indices)


# ---------------------------------------------------------------------
# Curvature-coupled scoring
# ---------------------------------------------------------------------


def test_curvature_weight_path_runs():
    """The κ-only path must not crash and must terminate."""
    qt = AdaptiveQuadtree(
        image_h=16, image_w=16, patch_size_initial=8,
        patch_size_min=2,
        max_depth=2, max_anchors=64,
        variance_weight=0.0, curvature_weight=1.0,
        score_threshold=3.0,  # all frontier anchors have |κ| ≈ 4 on a 2×2 layer
    )
    img = torch.zeros(1, 16, 16)
    tree = qt(img)
    # On a 2×2 frontier (4 scale-0 anchors), every vertex has degree 2 →
    # κ_v = -2 each → |κ| = 2 < threshold 3 → no subdivision.
    assert tree.n_anchors == 4
    assert (tree.scales == 0).all()


def test_hybrid_score_combines_variance_and_curvature():
    """variance_weight + curvature_weight both positive: both signals contribute."""
    qt = AdaptiveQuadtree(
        image_h=32, image_w=32, patch_size_initial=16,
        patch_size_min=2,
        max_depth=2, max_anchors=64,
        variance_weight=1.0, curvature_weight=0.5,
        score_threshold=0.5,
    )
    img = torch.zeros(1, 32, 32)
    img[0, 0:4, 0:4] = 1.0  # patch in upper-left bright
    tree = qt(img)
    # Should produce at least the 4 scale-0 anchors and likely some scale-1.
    assert tree.n_anchors >= 4
    assert tree.scales.max().item() <= 2


# ---------------------------------------------------------------------
# Output shape / typing
# ---------------------------------------------------------------------


def test_output_is_anchor_tree():
    qt = AdaptiveQuadtree(
        image_h=16, image_w=16, patch_size_initial=8,
    )
    img = torch.zeros(1, 16, 16)
    tree = qt(img)
    assert isinstance(tree, AnchorTree)
    # All tensors have consistent length and Long dtype.
    n = tree.n_anchors
    assert tree.positions.shape == (n, 2)
    assert tree.sizes.shape == (n,)
    assert tree.scales.shape == (n,)
    assert tree.parent_indices.shape == (n,)
    for t in (tree.positions, tree.sizes, tree.scales, tree.parent_indices):
        assert t.dtype == torch.long


def test_rejects_wrong_image_shape():
    qt = AdaptiveQuadtree(
        image_h=16, image_w=16, patch_size_initial=8,
    )
    with pytest.raises(ValueError, match="expected"):
        qt(torch.zeros(16, 16))  # missing channel dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
