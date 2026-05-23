"""Correctness + sanity tests for the Rust-backed quadtree.

Compares :class:`AdaptiveQuadtreeRust` against the Python reference
:class:`AdaptiveQuadtree`. The Rust implementation runs the
depth-by-depth subdivision state machine + Forman κ + budget cap;
variance scoring stays on GPU in Python.

The set-equality on returned anchor tuples is the load-bearing pin:
both implementations must produce identical
``(row, col, size, scale, parent)`` tuples for the same inputs,
modulo row ordering.

Plan: ``docs/plans/2026-05-16-gomb-soma-quadtree-triton/``.
"""
from __future__ import annotations

import pytest
import torch

# The Rust binding only works in miniconda (where hymeko is built);
# in .venv the import fails. Gate the whole file accordingly.
try:
    import hymeko as _hymeko
    _BUILD_OK = hasattr(_hymeko, "build_quadtree_rs")
except ImportError:
    _BUILD_OK = False

pytestmark = pytest.mark.skipif(
    not _BUILD_OK,
    reason="hymeko native module missing build_quadtree_rs symbol",
)

from signedkan_wip.src.hymeko_gomb.soma.vision.quadtree import (
    AdaptiveQuadtree,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.quadtree_rust import (
    AdaptiveQuadtreeRust,
)


# ─── helpers ──────────────────────────────────────────────────────────


def _tuples(tree):
    """Set of (row, col, size, scale, parent) tuples — order-invariant
    representation of an AnchorTree."""
    return set(
        (
            int(tree.positions[i, 0]),
            int(tree.positions[i, 1]),
            int(tree.sizes[i]),
            int(tree.scales[i]),
            int(tree.parent_indices[i]),
        )
        for i in range(tree.n_anchors)
    )


def _assert_set_equal(tree_py, tree_rs, label=""):
    py = _tuples(tree_py)
    rs = _tuples(tree_rs)
    only_py = py - rs
    only_rs = rs - py
    if only_py or only_rs:
        msg = (
            f"AnchorTree mismatch ({label}): "
            f"|only_py|={len(only_py)} |only_rs|={len(only_rs)}\n"
            f"first only_py: {list(only_py)[:5]}\n"
            f"first only_rs: {list(only_rs)[:5]}"
        )
        raise AssertionError(msg)


# ─── set-equality across seeds + thresholds ───────────────────────────


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13])
def test_set_equality_default_params(seed: int) -> None:
    torch.manual_seed(seed)
    kwargs = dict(
        image_h=64, image_w=64, patch_size_initial=16,
        patch_size_min=4, max_depth=4, max_anchors=256,
        variance_weight=1.0, curvature_weight=0.5,
        score_threshold=0.05,
    )
    qt_py = AdaptiveQuadtree(**kwargs)
    qt_rs = AdaptiveQuadtreeRust(**kwargs)
    img = torch.randn(3, 64, 64)
    _assert_set_equal(qt_py(img), qt_rs(img), label=f"seed={seed}")


@pytest.mark.parametrize("threshold", [0.0, 0.05, 0.20, 0.5])
def test_set_equality_threshold_sweep(threshold: float) -> None:
    torch.manual_seed(0)
    kwargs = dict(
        image_h=64, image_w=64, patch_size_initial=16,
        patch_size_min=4, max_depth=4, max_anchors=256,
        variance_weight=1.0, curvature_weight=0.0,
        score_threshold=threshold,
    )
    qt_py = AdaptiveQuadtree(**kwargs)
    qt_rs = AdaptiveQuadtreeRust(**kwargs)
    img = torch.randn(3, 64, 64)
    _assert_set_equal(qt_py(img), qt_rs(img), label=f"θ={threshold}")


def test_set_equality_curvature_only() -> None:
    """Drives the Forman-κ branch in isolation (variance_weight=0)."""
    torch.manual_seed(0)
    kwargs = dict(
        image_h=64, image_w=64, patch_size_initial=16,
        patch_size_min=4, max_depth=4, max_anchors=256,
        variance_weight=0.0, curvature_weight=1.0,
        score_threshold=0.01,
    )
    qt_py = AdaptiveQuadtree(**kwargs)
    qt_rs = AdaptiveQuadtreeRust(**kwargs)
    img = torch.randn(3, 64, 64)
    _assert_set_equal(qt_py(img), qt_rs(img), label="curvature-only")


def test_set_equality_at_budget() -> None:
    """Drives the budget-cap branch: max_anchors small enough that the
    budget kicks in mid-tree."""
    torch.manual_seed(0)
    kwargs = dict(
        image_h=64, image_w=64, patch_size_initial=16,
        patch_size_min=4, max_depth=4, max_anchors=40,
        variance_weight=1.0, curvature_weight=0.5,
        score_threshold=0.0,
    )
    qt_py = AdaptiveQuadtree(**kwargs)
    qt_rs = AdaptiveQuadtreeRust(**kwargs)
    img = torch.randn(3, 64, 64)
    _assert_set_equal(qt_py(img), qt_rs(img), label="budget-cap")


def test_set_equality_max_depth_0() -> None:
    """Degenerate case: max_depth=0 yields just the initial tiling."""
    kwargs = dict(
        image_h=64, image_w=64, patch_size_initial=16,
        patch_size_min=4, max_depth=0, max_anchors=256,
        variance_weight=1.0, curvature_weight=0.5,
        score_threshold=0.05,
    )
    qt_py = AdaptiveQuadtree(**kwargs)
    qt_rs = AdaptiveQuadtreeRust(**kwargs)
    img = torch.randn(3, 64, 64)
    _assert_set_equal(qt_py(img), qt_rs(img), label="max_depth=0")


def test_set_equality_128_image() -> None:
    """Larger image to exercise more depths of subdivision."""
    torch.manual_seed(0)
    kwargs = dict(
        image_h=128, image_w=128, patch_size_initial=32,
        patch_size_min=4, max_depth=5, max_anchors=1024,
        variance_weight=1.0, curvature_weight=0.5,
        score_threshold=0.05,
    )
    qt_py = AdaptiveQuadtree(**kwargs)
    qt_rs = AdaptiveQuadtreeRust(**kwargs)
    img = torch.randn(3, 128, 128)
    _assert_set_equal(qt_py(img), qt_rs(img), label="128²")


# ─── invariants on the Rust output alone ─────────────────────────────


def test_rust_output_is_valid_tree() -> None:
    """Every non-root anchor has a valid parent at depth-1."""
    torch.manual_seed(0)
    qt = AdaptiveQuadtreeRust(
        image_h=64, image_w=64, patch_size_initial=16,
        patch_size_min=4, max_depth=4, max_anchors=256,
        variance_weight=1.0, curvature_weight=0.5,
        score_threshold=0.05,
    )
    img = torch.randn(3, 64, 64)
    tree = qt(img)
    for i in range(tree.n_anchors):
        s = int(tree.scales[i])
        p = int(tree.parent_indices[i])
        if s == 0:
            assert p == -1, f"root anchor {i} has parent {p}"
        else:
            assert 0 <= p < i, f"anchor {i} parent {p} out of range"
            assert int(tree.scales[p]) == s - 1, (
                f"anchor {i} at scale {s} has parent at scale "
                f"{int(tree.scales[p])} (expected {s - 1})"
            )


def test_rust_respects_max_anchors() -> None:
    qt = AdaptiveQuadtreeRust(
        image_h=64, image_w=64, patch_size_initial=16,
        patch_size_min=4, max_depth=4, max_anchors=64,
        variance_weight=1.0, curvature_weight=0.5,
        score_threshold=0.0,  # subdivide everything → would exceed budget
    )
    img = torch.randn(3, 64, 64)
    tree = qt(img)
    assert tree.n_anchors <= 64, (
        f"output {tree.n_anchors} anchors exceeds budget 64"
    )


def test_rust_deterministic_under_same_inputs() -> None:
    """Two calls with the same RNG seed + image → identical output."""
    qt = AdaptiveQuadtreeRust(
        image_h=64, image_w=64, patch_size_initial=16,
        patch_size_min=4, max_depth=4, max_anchors=256,
        variance_weight=1.0, curvature_weight=0.5,
        score_threshold=0.05,
    )
    torch.manual_seed(42)
    img = torch.randn(3, 64, 64)
    t1 = qt(img)
    t2 = qt(img)
    assert _tuples(t1) == _tuples(t2)


# ─── constructor sanity ───────────────────────────────────────────────


def test_constructor_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="patch_size_initial"):
        AdaptiveQuadtreeRust(
            image_h=64, image_w=64, patch_size_initial=0,
            patch_size_min=4, max_depth=4, max_anchors=256,
            variance_weight=1.0,
        )
    with pytest.raises(ValueError, match="patch_size_min"):
        AdaptiveQuadtreeRust(
            image_h=64, image_w=64, patch_size_initial=16,
            patch_size_min=32, max_depth=4, max_anchors=256,
            variance_weight=1.0,
        )
    with pytest.raises(ValueError, match="divisible"):
        AdaptiveQuadtreeRust(
            image_h=65, image_w=64, patch_size_initial=16,
            patch_size_min=4, max_depth=4, max_anchors=256,
            variance_weight=1.0,
        )
    with pytest.raises(ValueError, match="weights"):
        AdaptiveQuadtreeRust(
            image_h=64, image_w=64, patch_size_initial=16,
            patch_size_min=4, max_depth=4, max_anchors=256,
            variance_weight=-1.0,
        )
    with pytest.raises(ValueError, match="at least one"):
        AdaptiveQuadtreeRust(
            image_h=64, image_w=64, patch_size_initial=16,
            patch_size_min=4, max_depth=4, max_anchors=256,
            variance_weight=0.0, curvature_weight=0.0,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
