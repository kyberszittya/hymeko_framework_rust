"""Tests for `ssg_pareto_filter` --- equivalence + edge cases + perf.

The brute O(N² D) path is the SPECIFICATION; the new sort-and-sweep
O(N log N) path (D = 2) must match it bit-for-bit. Plan:
`docs/plans/2026-06-04-klp-skyline/plan.{tex,pdf}`.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from hymeko_neuro.hyperedge.abb_walks import (
    _ssg_pareto_filter_brute,
    _ssg_pareto_filter_sweep_2d,
    ssg_pareto_filter,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _dummy_walks(n: int, walk_len: int = 4) -> tuple[np.ndarray, np.ndarray]:
    v = np.arange(n * (walk_len + 1), dtype=np.int32).reshape(n, walk_len + 1)
    signs = np.ones((n, walk_len), dtype=np.int8)
    return v, signs


def _random_scores(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, d)).astype(np.float64)


# ----------------------------------------------------------------------
# Equivalence: brute is the spec; sweep_2d must match exactly
# ----------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n", [10, 100, 1000])
def test_sweep_matches_brute_random(seed: int, n: int):
    scores = _random_scores(n, 2, seed)
    mask_brute = _ssg_pareto_filter_brute(scores)
    mask_sweep = _ssg_pareto_filter_sweep_2d(scores)
    assert mask_brute.shape == mask_sweep.shape == (n,)
    np.testing.assert_array_equal(mask_brute, mask_sweep)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n", [10, 100])
def test_sweep_matches_brute_integer_scores(seed: int, n: int):
    """Integer scores exercise tie handling more aggressively
    (uniform [0, 5)² generates many duplicate (a_0, a_1) pairs)."""
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 5, size=(n, 2)).astype(np.float64)
    mask_brute = _ssg_pareto_filter_brute(scores)
    mask_sweep = _ssg_pareto_filter_sweep_2d(scores)
    np.testing.assert_array_equal(mask_brute, mask_sweep)


def test_public_dispatch_uses_sweep_at_d2():
    """At D = 2 the public function returns the sweep_2d result;
    the boolean mask must equal the direct sweep call."""
    rng = np.random.default_rng(7)
    n = 500
    v, signs = _dummy_walks(n)
    axes = [rng.random(n), rng.random(n)]
    _, _, mask_public = ssg_pareto_filter(v, signs, axes)
    direct = _ssg_pareto_filter_sweep_2d(np.column_stack(axes))
    np.testing.assert_array_equal(mask_public, direct)


def test_public_dispatch_falls_back_to_brute_at_d3():
    rng = np.random.default_rng(7)
    n = 100
    v, signs = _dummy_walks(n)
    axes = [rng.random(n), rng.random(n), rng.random(n)]
    _, _, mask_public = ssg_pareto_filter(v, signs, axes)
    direct = _ssg_pareto_filter_brute(np.column_stack(axes))
    np.testing.assert_array_equal(mask_public, direct)


# ----------------------------------------------------------------------
# Algebraic edge cases
# ----------------------------------------------------------------------

def test_empty_input_returns_empty_mask():
    v = np.zeros((0, 5), dtype=np.int32)
    signs = np.zeros((0, 4), dtype=np.int8)
    fv, fs, mask = ssg_pareto_filter(v, signs, [np.zeros(0), np.zeros(0)])
    assert fv.shape == (0, 5)
    assert fs.shape == (0, 4)
    assert mask.shape == (0,)


def test_no_axes_returns_all_true():
    n = 7
    v, signs = _dummy_walks(n)
    fv, fs, mask = ssg_pareto_filter(v, signs, [])
    assert mask.all()
    assert fv.shape == (n, 5)
    assert fs.shape == (n, 4)


def test_single_point_is_on_skyline():
    scores = np.array([[0.5, 0.3]])
    assert _ssg_pareto_filter_sweep_2d(scores).tolist() == [True]
    assert _ssg_pareto_filter_brute(scores).tolist() == [True]


def test_all_identical_points_all_pass():
    """If every row is identical, no row strictly dominates any other,
    so the brute reference returns all True. The sweep MUST agree."""
    n = 50
    scores = np.full((n, 2), 0.7)
    mb = _ssg_pareto_filter_brute(scores)
    ms = _ssg_pareto_filter_sweep_2d(scores)
    np.testing.assert_array_equal(mb, ms)
    assert mb.all()


def test_anticorrelated_all_on_skyline():
    """Strictly decreasing a_1 along ascending a_0 means every point
    is undominated (each is best on one axis, worst on the other)."""
    n = 30
    a0 = np.arange(n, dtype=np.float64)
    a1 = -a0
    scores = np.column_stack([a0, a1])
    mb = _ssg_pareto_filter_brute(scores)
    ms = _ssg_pareto_filter_sweep_2d(scores)
    np.testing.assert_array_equal(mb, ms)
    assert mb.all()


def test_strict_chain_only_dominant_survives():
    """A strict chain where each point dominates the next leaves
    only the head of the chain on the skyline."""
    scores = np.array(
        [
            [5.0, 5.0],   # dominates everything below
            [3.0, 3.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ]
    )
    mb = _ssg_pareto_filter_brute(scores)
    ms = _ssg_pareto_filter_sweep_2d(scores)
    np.testing.assert_array_equal(mb, ms)
    assert mb.tolist() == [True, False, False, False]


def test_validates_axis_shapes():
    v = np.zeros((5, 3), dtype=np.int32)
    signs = np.zeros((5, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        ssg_pareto_filter(v, signs, [np.zeros(5), np.zeros(4)])  # wrong shape


# ----------------------------------------------------------------------
# Performance assertion (sweep beats brute at N = 1000)
# ----------------------------------------------------------------------

def _wall_median(fn, scores: np.ndarray, n_iter: int = 5) -> float:
    walls = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(scores)
        walls.append(time.perf_counter() - t0)
    walls.sort()
    return walls[len(walls) // 2]


def test_sweep_is_faster_than_brute_at_n1000():
    """Wall-time assertion: sweep median <= 0.3 × brute median at N=1000.
    Empirically the ratio is ~0.05; 0.3 is a 6× safety margin against
    profile noise on a busy host."""
    rng = np.random.default_rng(13)
    scores = rng.random((1000, 2))
    # Warm-up to trigger numpy lazy init.
    _ssg_pareto_filter_brute(scores)
    _ssg_pareto_filter_sweep_2d(scores)

    brute_wall = _wall_median(_ssg_pareto_filter_brute, scores, n_iter=5)
    sweep_wall = _wall_median(_ssg_pareto_filter_sweep_2d, scores, n_iter=5)

    assert sweep_wall <= 0.3 * brute_wall, (
        f"sweep wall {sweep_wall*1e3:.2f} ms must be <= 0.3 × brute "
        f"{brute_wall*1e3:.2f} ms (ratio {sweep_wall/brute_wall:.3f})"
    )
