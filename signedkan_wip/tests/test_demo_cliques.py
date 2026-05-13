"""Tests for the robot communication cliques demo.

Covers: generator determinism, balance verification on known cases,
clique enumeration semantics + size / limit handling.
"""
from __future__ import annotations

import numpy as np
import pytest

from signedkan_wip.src.datasets import SignedGraph
from signedkan_wip.src.demo.cliques import (
    Clique, RobotNetworkBundle, balance_summary,
    enumerate_balanced_cliques, make_robot_network,
)


def test_generator_determinism():
    """Same seed → identical bundle."""
    a = make_robot_network(n_robots=15, seed=42)
    b = make_robot_network(n_robots=15, seed=42)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.graph.edges, b.graph.edges)
    assert np.array_equal(a.graph.signs, b.graph.signs)
    assert a.n_edges == b.n_edges
    assert a.n_negative_edges == b.n_negative_edges


def test_generator_different_seeds_diverge():
    """Different seeds must produce different networks."""
    a = make_robot_network(n_robots=15, seed=1)
    b = make_robot_network(n_robots=15, seed=2)
    assert not np.array_equal(a.positions, b.positions)


def test_generator_respects_n_robots():
    b = make_robot_network(n_robots=20, seed=0)
    assert b.n_robots == 20
    assert b.positions.shape == (20, 2)


def test_generator_no_self_loops_and_no_duplicates():
    """All edges are u<v ordered, no diagonal entries."""
    b = make_robot_network(n_robots=30, seed=0)
    seen = set()
    for u, v in b.graph.edges:
        assert int(u) != int(v), "self-loop"
        assert int(u) < int(v), "edge ordering must be u<v"
        assert (int(u), int(v)) not in seen, "duplicate edge"
        seen.add((int(u), int(v)))


def test_balance_summary_well_formed():
    b = make_robot_network(n_robots=20, seed=0, noise_prob=0.2)
    s = balance_summary(b)
    assert s["n_robots"] == 20
    assert s["n_edges"] >= 0
    assert s["n_positive"] + s["n_negative"] == s["n_edges"]
    if s["n_edges"]:
        assert 0.0 <= s["negative_fraction"] <= 1.0


def _hand_built_bundle(n: int, edges: list[tuple[int, int, int]]
                          ) -> RobotNetworkBundle:
    """Build a bundle without going through the random generator."""
    es = np.array([(u, v) for u, v, _ in edges], dtype=np.int64)
    ss = np.array([s for _, _, s in edges], dtype=np.int8)
    g = SignedGraph(edges=es, signs=ss, n_nodes=n)
    return RobotNetworkBundle(
        graph=g,
        positions=np.zeros((n, 2), dtype=np.float32),
        seed=0, comm_range=1.0, noise_prob=0.0, area_size=1.0,
        name="hand",
    )


def test_triangle_all_positive_is_balanced():
    """+++  → σ-product = +1 → balanced."""
    b = _hand_built_bundle(3, [(0, 1, +1), (1, 2, +1), (0, 2, +1)])
    cliques = enumerate_balanced_cliques(b, min_size=3, max_size=3)
    assert len(cliques) == 1
    c = cliques[0]
    assert c.size == 3
    assert c.balanced
    assert c.sigma_product == 1


def test_triangle_one_negative_is_unbalanced():
    """++− → σ-product = −1 → NOT in output."""
    b = _hand_built_bundle(3, [(0, 1, +1), (1, 2, +1), (0, 2, -1)])
    cliques = enumerate_balanced_cliques(b, min_size=3, max_size=3)
    assert cliques == []


def test_triangle_two_negatives_is_balanced():
    """+−− → σ-product = +1 → BALANCED ('enemy of my enemy is my friend')."""
    b = _hand_built_bundle(3, [(0, 1, +1), (1, 2, -1), (0, 2, -1)])
    cliques = enumerate_balanced_cliques(b, min_size=3, max_size=3)
    assert len(cliques) == 1
    assert cliques[0].balanced
    assert cliques[0].sigma_product == 1


def test_min_size_filter_excludes_small_cliques():
    b = _hand_built_bundle(3, [(0, 1, +1), (1, 2, +1), (0, 2, +1)])
    assert enumerate_balanced_cliques(b, min_size=4) == []


def test_clique_limit_is_respected(tmp_path):
    """Generate a moderately dense network; verify limit truncates."""
    b = make_robot_network(n_robots=30, comm_range=5.0,
                              noise_prob=0.0, seed=7)
    out = enumerate_balanced_cliques(b, min_size=3, max_size=6, limit=3)
    assert len(out) <= 3
    # Sorted by size descending.
    sizes = [c.size for c in out]
    assert sizes == sorted(sizes, reverse=True)


def test_balanced_clique_edges_match_signs():
    """Internal invariant: len(edges) == C(size, 2) and product
    of signs equals sigma_product."""
    b = make_robot_network(n_robots=25, comm_range=4.5,
                              noise_prob=0.05, seed=11)
    for c in enumerate_balanced_cliques(b, min_size=3, max_size=5):
        from math import comb
        assert len(c.edges) == comb(c.size, 2)
        assert len(c.signs) == len(c.edges)
        prod = 1
        for s in c.signs:
            prod *= s
        assert prod == c.sigma_product == 1
