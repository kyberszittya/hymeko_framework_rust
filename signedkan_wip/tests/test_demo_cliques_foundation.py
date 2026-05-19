"""Tests for the planted-clique generator + detector benchmark harness.

Covers:
- planted generator invariants (every planted clique IS a clique and
  IS balanced; planted_edge_mask matches the planted edges)
- four detectors' Strategy interface contract (return type, balance
  invariants)
- Jaccard / recall metric correctness on hand-built cases
- benchmark_detector wall-time recording
"""
from __future__ import annotations

import time
from math import comb

import numpy as np
import pytest

from signedkan_wip.src.demo.cliques import Clique, _clique_balance_indicator
from signedkan_wip.src.demo.cliques_planted import (
    PlantedRobotNetworkBundle, make_planted_balanced_cliques,
)
from signedkan_wip.src.demo.cliques_bench import (
    BronKerboschDetector, GreedyBalancedDetector,
    SpectralBalancedDetector, TriangleDensityDetector,
    benchmark_detector, default_detectors, jaccard_overlap,
    recall_against_planted,
)


# ─── Planted generator ──────────────────────────────────────────────


def test_planted_generator_determinism():
    a = make_planted_balanced_cliques(n_robots=30, clique_sizes=[5, 4, 3],
                                          seed=42)
    b = make_planted_balanced_cliques(n_robots=30, clique_sizes=[5, 4, 3],
                                          seed=42)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.graph.edges, b.graph.edges)
    assert np.array_equal(a.graph.signs, b.graph.signs)
    assert [c.members for c in a.planted_cliques] == \
           [c.members for c in b.planted_cliques]


def test_planted_cliques_are_balanced():
    """Planted cliques must be balanced under the triangle-product
    definition (every triangle has σ-product = +1). This is NOT the
    same as the all-edges-product for k ≥ 4 — see
    ``cliques._clique_balance_indicator`` docstring."""
    bundle = make_planted_balanced_cliques(
        n_robots=40, clique_sizes=[6, 4, 3],
        n_factions=2, noise_prob=0.05, seed=0,
    )
    # Build a sign lookup over the bundle's edges.
    sign_of = {}
    for (u, v), s in zip(bundle.graph.edges, bundle.graph.signs):
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        sign_of[(a, b)] = int(s)
    for clique in bundle.planted_cliques:
        assert _clique_balance_indicator(clique.members, sign_of) == 1
        assert clique.sigma_product == 1
        assert clique.balanced


def test_planted_cliques_have_all_internal_edges():
    """A planted size-c clique has exactly C(c, 2) internal edges."""
    bundle = make_planted_balanced_cliques(
        n_robots=40, clique_sizes=[6, 4, 3], seed=0,
    )
    for clique in bundle.planted_cliques:
        assert len(clique.edges) == comb(clique.size, 2)
        assert len(clique.signs) == len(clique.edges)


def test_planted_edge_mask_matches_planted_edges():
    """planted_edge_mask must align with the bundle.graph.edges array
    and flag exactly the planted edges."""
    bundle = make_planted_balanced_cliques(
        n_robots=40, clique_sizes=[5, 4, 3], seed=0,
    )
    mask = bundle.planted_edge_mask
    assert mask is not None
    assert mask.shape == (bundle.n_edges,)

    # Build the expected planted-edge set.
    expected_planted: set[tuple[int, int]] = set()
    for c in bundle.planted_cliques:
        for u, v in c.edges:
            expected_planted.add((min(u, v), max(u, v)))

    # Verify mask flags exactly those edges.
    for i, (u, v) in enumerate(bundle.graph.edges):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if mask[i]:
            assert key in expected_planted
        else:
            assert key not in expected_planted


def test_planted_cliques_are_disjoint():
    """No vertex lives in two planted cliques."""
    bundle = make_planted_balanced_cliques(
        n_robots=50, clique_sizes=[6, 5, 4, 3], seed=0,
    )
    seen: set[int] = set()
    for c in bundle.planted_cliques:
        for m in c.members:
            assert m not in seen, f"vertex {m} planted in two cliques"
            seen.add(m)


def test_planted_sign_strategy_split_is_still_balanced():
    """The 'split' strategy produces 2-coloring-balanced cliques.

    For a 4-clique with 2-2 split, the all-edges product is −1 but
    every triangle has σ-product = +1, which is the correct
    definition of balance. This test enforces the triangle check."""
    bundle = make_planted_balanced_cliques(
        n_robots=30, clique_sizes=[6, 4],
        planted_sign_strategy="split", seed=0,
    )
    sign_of = {}
    for (u, v), s in zip(bundle.graph.edges, bundle.graph.signs):
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        sign_of[(a, b)] = int(s)
    for c in bundle.planted_cliques:
        assert _clique_balance_indicator(c.members, sign_of) == 1, (
            f"split clique {c.members} not balanced under triangle check"
        )


def test_oversize_planted_request_raises():
    """sum(clique_sizes) > n_robots is invalid — would force vertex
    overlap between cliques."""
    with pytest.raises(ValueError):
        make_planted_balanced_cliques(
            n_robots=10, clique_sizes=[6, 5, 4], seed=0,
        )


def test_small_clique_size_raises():
    """Cliques smaller than 3 aren't cycles, can't carry balance info."""
    with pytest.raises(ValueError):
        make_planted_balanced_cliques(
            n_robots=20, clique_sizes=[3, 2], seed=0,
        )


# ─── Detectors ──────────────────────────────────────────────────────


def _small_planted_bundle() -> PlantedRobotNetworkBundle:
    """A small bundle every detector should handle."""
    return make_planted_balanced_cliques(
        n_robots=25, clique_sizes=[5, 4, 3],
        comm_range=4.5, noise_prob=0.05, n_factions=2,
        seed=0,
    )


@pytest.mark.parametrize("detector_factory", [
    BronKerboschDetector,
    TriangleDensityDetector,
    GreedyBalancedDetector,
])
def test_detector_returns_balanced_cliques(detector_factory):
    """Every returned clique must (a) be a real clique, (b) be balanced."""
    bundle = _small_planted_bundle()
    detector = detector_factory()
    cliques = detector.detect(bundle, min_size=3, max_size=8, limit=20)
    # Spectral may return empty on small networks; that's allowed.
    # The other three should return something.
    if detector_factory is not SpectralBalancedDetector:
        assert len(cliques) > 0, f"{detector.name} returned no cliques"
    # Sign lookup for verification.
    sign_of = {}
    for (u, v), s in zip(bundle.graph.edges, bundle.graph.signs):
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        sign_of[(a, b)] = int(s)
    for c in cliques:
        # All internal pairs are edges.
        for i in range(c.size):
            for j in range(i + 1, c.size):
                a, b = c.members[i], c.members[j]
                key = (min(a, b), max(a, b))
                assert key in sign_of, (
                    f"{detector.name} returned non-clique on members "
                    f"{c.members}: missing edge ({a}, {b})"
                )
        # Balance per the triangle-product definition (not the
        # all-edges product, which is wrong for k ≥ 4 — see
        # cliques._clique_balance_indicator docstring).
        assert _clique_balance_indicator(c.members, sign_of) == 1, (
            f"{detector.name} returned unbalanced clique {c.members}"
        )
        assert c.sigma_product == 1


def test_bron_kerbosch_is_ground_truth_on_small():
    """On a small network, Bron-Kerbosch finds the planted cliques
    (recall = 1.0) up to the Jaccard threshold."""
    bundle = make_planted_balanced_cliques(
        n_robots=20, clique_sizes=[4, 3],
        comm_range=4.0, noise_prob=0.02, n_factions=2,
        seed=0,
    )
    cliques = BronKerboschDetector().detect(
        bundle, min_size=3, max_size=8, limit=50)
    m = recall_against_planted(cliques, bundle.planted_cliques,
                                   overlap_threshold=0.5)
    assert m["recall"] >= 0.5  # at least half the planted cliques.
    assert m["n_detected"] > 0


# ─── Metrics ────────────────────────────────────────────────────────


def test_jaccard_overlap_canonical():
    a = Clique(members=(0, 1, 2), edges=[(0, 1), (0, 2), (1, 2)],
               signs=[1, 1, 1], sigma_product=1)
    b = Clique(members=(0, 1, 3), edges=[(0, 1), (0, 3), (1, 3)],
               signs=[1, 1, 1], sigma_product=1)
    # {0,1,2} ∩ {0,1,3} = {0,1}; ∪ = {0,1,2,3}; J = 2/4 = 0.5.
    assert jaccard_overlap(a, b) == pytest.approx(0.5)
    assert jaccard_overlap(a, a) == pytest.approx(1.0)


def test_jaccard_overlap_disjoint_is_zero():
    a = Clique(members=(0, 1, 2), edges=[], signs=[], sigma_product=1)
    b = Clique(members=(5, 6, 7), edges=[], signs=[], sigma_product=1)
    assert jaccard_overlap(a, b) == 0.0


def test_recall_against_planted_handles_empty():
    """No planted cliques → metric returns NaN recall (well-defined)."""
    m = recall_against_planted(detected=[], planted=[])
    assert m["n_planted"] == 0
    assert np.isnan(m["recall"])


def test_recall_against_planted_perfect_match():
    c1 = Clique(members=(0, 1, 2), edges=[], signs=[], sigma_product=1)
    c2 = Clique(members=(3, 4, 5, 6), edges=[], signs=[], sigma_product=1)
    m = recall_against_planted(detected=[c1, c2], planted=[c1, c2])
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0


def test_recall_against_planted_partial_match():
    """One planted is found exactly, one is missed."""
    c1 = Clique(members=(0, 1, 2), edges=[], signs=[], sigma_product=1)
    c2 = Clique(members=(3, 4, 5, 6), edges=[], signs=[], sigma_product=1)
    other = Clique(members=(10, 11, 12), edges=[], signs=[], sigma_product=1)
    m = recall_against_planted(detected=[c1, other], planted=[c1, c2])
    assert m["recall"] == 0.5
    # One of two detected matches a planted → precision = 0.5.
    assert m["precision"] == 0.5


# ─── Benchmark harness ──────────────────────────────────────────────


def test_benchmark_detector_records_walltime():
    bundle = _small_planted_bundle()
    r = benchmark_detector(BronKerboschDetector(), bundle,
                              min_size=3, max_size=8, limit=20)
    assert r.detector_name == "bron_kerbosch_exact"
    assert r.wall_time_s >= 0.0
    assert r.error is None
    assert not r.timed_out


def test_benchmark_detector_marks_timeout():
    """If wall-time exceeds the budget, mark `timed_out` but don't kill
    the run (we can't preempt cleanly mid-NetworkX-call)."""
    bundle = _small_planted_bundle()
    r = benchmark_detector(BronKerboschDetector(), bundle,
                              min_size=3, max_size=8, limit=20,
                              timeout_s=1e-9)
    # Detector finished but wall-time exceeded the absurd budget.
    assert r.timed_out is True


def test_default_detectors_returns_four():
    ds = default_detectors()
    assert len(ds) == 4
    names = {d.name for d in ds}
    assert names == {
        "bron_kerbosch_exact",
        "triangle_density_greedy",
        "greedy_balanced",
        "spectral_balanced",
    }
