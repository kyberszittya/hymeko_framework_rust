"""Correctness tests for the stim-graph geometry cache.

The cache is only sound if (a) splitting the builder into build_geometry +
apply_signs is bit-exact vs the old forward, (b) a cached backbone forward equals
the uncached one, (c) edge signs are still recomputed from features on a cache
hit (not frozen), and (d) SDRF bypasses the cache. These are the load-bearing
gates from docs/plans/2026-06-16-stim-geometry-cache/.

Run: pytest -p no:randomly signedkan_wip/tests/test_stim_geometry_cache.py
"""
from __future__ import annotations

import torch

from signedkan_wip.src.hymeko_gomb.soma.vision.quadtree import AdaptiveQuadtree
from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_backbone import (
    RicciStimBackbone,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.stim_graph import (
    StimulusGraph,
    StimulusGraphBuilder,
)


def _eq(x: torch.Tensor, y: torch.Tensor) -> bool:
    if x.is_sparse or y.is_sparse:
        return torch.equal(x.to_dense(), y.to_dense())
    return torch.equal(x, y)


def _assert_sg_equal(a: StimulusGraph, b: StimulusGraph) -> None:
    for field in (
        "edges", "edge_signs", "edge_curvatures",
        "walks", "walk_signs", "walk_curvatures",
        "polygons", "polygon_signs", "polygon_curvatures",
        "triangles", "triangle_signs", "triangle_curvatures",
        "M_v_walks", "M_v_polygons", "M_v_triangles", "hodge_laplacian_0",
    ):
        assert _eq(getattr(a, field), getattr(b, field)), f"mismatch in {field}"
    assert a.n_anchors == b.n_anchors


def _tree_and_features(seed: int = 0, d: int = 8):
    torch.manual_seed(seed)
    qt = AdaptiveQuadtree(
        image_h=12, image_w=12, patch_size_initial=4,
        patch_size_min=1, max_depth=1,
    )
    img = torch.rand(1, 12, 12)
    tree = qt(img)
    features = torch.randn(tree.n_anchors, d)
    return tree, features


def test_apply_signs_matches_forward_bit_exactly() -> None:
    builder = StimulusGraphBuilder()
    tree, features = _tree_and_features()
    direct = builder.forward(tree, features)
    split = builder.apply_signs(builder.build_geometry(tree), features)
    _assert_sg_equal(direct, split)


def test_geometry_is_feature_independent_but_signs_are_not() -> None:
    builder = StimulusGraphBuilder()
    tree, features_a = _tree_and_features(seed=1)
    # Independent features (NOT a sign symmetry: negating both endpoints would
    # leave every inner product unchanged, so use genuinely different vectors).
    torch.manual_seed(2)
    features_b = torch.randn_like(features_a)
    geometry = builder.build_geometry(tree)
    sg_a = builder.apply_signs(geometry, features_a)
    sg_b = builder.apply_signs(geometry, features_b)
    # Same geometry object -> identical topology...
    assert _eq(sg_a.edges, sg_b.edges)
    assert _eq(sg_a.walks, sg_b.walks)
    assert _eq(sg_a.M_v_walks, sg_b.M_v_walks)
    # ...but the feature-dependent signs differ (independent features change the
    # inner-product polarity on at least one edge w.p. ~1).
    assert sg_a.edge_signs.numel() > 0
    assert not torch.equal(sg_a.edge_signs, sg_b.edge_signs)


def _backbone(cache: bool, seed: int = 0) -> RicciStimBackbone:
    torch.manual_seed(seed)
    return RicciStimBackbone(
        image_h=12, image_w=12, patch_size_initial=4, patch_size_min=1,
        d_hidden=8, max_depth=1, cache_geometry=cache,
    )


def test_backbone_cached_forward_matches_uncached() -> None:
    bb_off = _backbone(cache=False, seed=7)
    bb_on = _backbone(cache=True, seed=7)  # identical init weights
    torch.manual_seed(99)
    img = torch.rand(1, 12, 12)
    f_off, _ = bb_off(img)
    f_on1, _ = bb_on(img)   # miss: builds + caches
    f_on2, _ = bb_on(img)   # hit: reuses geometry
    assert torch.allclose(f_off, f_on1, atol=0.0)
    assert torch.allclose(f_on1, f_on2, atol=0.0)
    assert bb_on._cache_misses == 1
    assert bb_on._cache_hits == 1


def test_cache_hit_still_recomputes_signs_after_weight_change() -> None:
    bb = _backbone(cache=True, seed=3)
    torch.manual_seed(123)
    img = torch.rand(1, 12, 12)
    f1, _ = bb(img)                       # miss
    with torch.no_grad():
        bb.patch_encoder.weight.mul_(3.0)  # change features -> change signs
    f2, _ = bb(img)                       # hit on geometry, recompute features
    assert bb._cache_misses == 1 and bb._cache_hits == 1
    assert not torch.allclose(f1, f2), "cache hit must still recompute features/signs"


def test_sdrf_bypasses_the_cache() -> None:
    torch.manual_seed(5)
    bb = RicciStimBackbone(
        image_h=12, image_w=12, patch_size_initial=4, patch_size_min=1,
        d_hidden=8, max_depth=1, use_sdrf=True, cache_geometry=True,
    )
    img = torch.rand(1, 12, 12)
    bb(img)
    bb(img)
    # use_sdrf rewires topology from features, so the cache must stay untouched.
    assert bb._cache_hits == 0
    assert bb._cache_misses == 0
    assert bb._geometry_cache == {}
