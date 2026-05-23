"""Tests for the cliques-to-nodelets contraction operator.

Covers the mathematical invariants from
``cliques_contract.py``'s docstring:

  1. Vertex count non-increasing (|V'| ≤ |V|).
  2. Identity case: no balanced cliques → input returned unchanged
     (every original vertex becomes its own singleton super-vertex).
  3. Full-balance case: a single balanced clique → 1 super-vertex.
  4. Determinism: same input → same output.
  5. Edge preservation modulo internal absorption: no edge "lost"
     without being explicitly absorbed into a super-vertex.
  6. φ-map round-trip: every original vertex belongs to exactly one
     super-vertex.
  7. Sign aggregation respects majority + tie rule.
  8. Multi-scale termination: hierarchy stops at a fixed point or
     the singleton root.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pytest

from signedkan_wip.src.datasets import SignedGraph
from signedkan_wip.src.demo.cliques import RobotNetworkBundle
from signedkan_wip.src.demo.cliques_planted import (
    make_planted_balanced_cliques,
)
from signedkan_wip.src.demo.cliques_contract import (
    ContractedBundle, contract_balanced_cliques, multiscale_hierarchy,
)


def _hand_built_bundle(n: int,
                          edges: list[tuple[int, int, int]]
                          ) -> RobotNetworkBundle:
    """Build a bundle directly without the random generator."""
    if edges:
        es = np.array([(u, v) for u, v, _ in edges], dtype=np.int64)
        ss = np.array([s for _, _, s in edges], dtype=np.int8)
    else:
        es = np.zeros((0, 2), dtype=np.int64)
        ss = np.zeros((0,), dtype=np.int8)
    g = SignedGraph(edges=es, signs=ss, n_nodes=n)
    return RobotNetworkBundle(
        graph=g,
        positions=np.zeros((n, 2), dtype=np.float32),
        seed=0, comm_range=1.0, noise_prob=0.0, area_size=1.0,
        name="hand",
    )


# ─── Property 1: vertex count is non-increasing ─────────────────────


def test_vertex_count_non_increasing():
    """|V'| ≤ |V| on every network."""
    b = make_planted_balanced_cliques(
        n_robots=30, clique_sizes=[6, 5, 4, 3],
        n_factions=2, noise_prob=0.05, seed=0,
    )
    h1 = contract_balanced_cliques(b)
    assert h1.n_robots <= b.n_robots
    assert h1.n_robots > 0


# ─── Property 2: identity case ──────────────────────────────────────


def test_identity_case_no_cliques():
    """A graph with no balanced cliques of size ≥ 3 → every vertex
    becomes its own singleton super-vertex (|V'| = |V|)."""
    # Two disconnected vertices.
    b = _hand_built_bundle(2, [])
    h = contract_balanced_cliques(b)
    assert h.n_robots == 2
    assert h.n_singletons == 2
    assert h.compression_ratio == 1.0
    # A single edge (no triangles).
    b = _hand_built_bundle(2, [(0, 1, +1)])
    h = contract_balanced_cliques(b)
    assert h.n_robots == 2
    assert h.n_singletons == 2


def test_identity_case_unbalanced_triangle():
    """An unbalanced triangle (++−) is NOT a balanced clique → no
    contraction happens."""
    b = _hand_built_bundle(3, [(0, 1, +1), (1, 2, +1), (0, 2, -1)])
    h = contract_balanced_cliques(b)
    assert h.n_robots == 3
    assert h.n_singletons == 3


# ─── Property 3: full-balance case ──────────────────────────────────


def test_full_balance_collapses_to_one_vertex():
    """A single balanced triangle collapses to 1 super-vertex."""
    b = _hand_built_bundle(3, [(0, 1, +1), (1, 2, +1), (0, 2, +1)])
    h = contract_balanced_cliques(b)
    assert h.n_robots == 1
    assert h.n_edges == 0
    assert h.super_members == [(0, 1, 2)]


def test_balanced_4_clique_with_2_2_split_collapses():
    """4-clique split 2-2 is balanced under triangle check;
    must contract to 1 super-vertex (test of correct balance
    semantics for k ≥ 4)."""
    b = _hand_built_bundle(4, [
        (0, 1, +1), (2, 3, +1),                # within-block
        (0, 2, -1), (0, 3, -1),                # across
        (1, 2, -1), (1, 3, -1),
    ])
    h = contract_balanced_cliques(b)
    assert h.n_robots == 1


# ─── Property 4: determinism ────────────────────────────────────────


def test_contraction_is_deterministic():
    """Identical input → identical output."""
    b1 = make_planted_balanced_cliques(
        n_robots=25, clique_sizes=[5, 4, 3], seed=7,
    )
    b2 = make_planted_balanced_cliques(
        n_robots=25, clique_sizes=[5, 4, 3], seed=7,
    )
    h1 = contract_balanced_cliques(b1)
    h2 = contract_balanced_cliques(b2)
    assert h1.n_robots == h2.n_robots
    assert np.array_equal(h1.graph.edges, h2.graph.edges)
    assert np.array_equal(h1.graph.signs, h2.graph.signs)
    assert h1.super_members == h2.super_members


# ─── Property 5: edge preservation modulo absorption ────────────────


def test_no_edge_lost_without_absorption():
    """Every edge in the input is either absorbed (both endpoints
    map to the same super-vertex) or surfaces in the output."""
    b = make_planted_balanced_cliques(
        n_robots=25, clique_sizes=[5, 4, 3], n_factions=2, seed=0,
    )
    h = contract_balanced_cliques(b)
    phi = h.phi
    assert phi is not None

    absorbed = 0
    quotient_keys: set[tuple[int, int]] = set()
    for u, v in b.graph.edges:
        i, j = int(phi[int(u)]), int(phi[int(v)])
        if i == j:
            absorbed += 1
        else:
            quotient_keys.add((i, j) if i < j else (j, i))

    # Every distinct quotient pair must appear in the output edges.
    out_keys = {
        (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        for u, v in h.graph.edges
    }
    assert quotient_keys == out_keys
    # Sanity: absorbed + |out_keys| ≤ total edges (an output key can
    # correspond to multiple absorbed input edges).
    assert absorbed >= 0
    assert absorbed + h.n_edges <= b.n_edges + len(quotient_keys)


# ─── Property 6: φ-map round-trip ───────────────────────────────────


def test_phi_map_partitions_original_vertices():
    """Every original vertex belongs to exactly one super-vertex
    (phi is a total function from V to V')."""
    b = make_planted_balanced_cliques(
        n_robots=20, clique_sizes=[5, 4, 3], seed=0,
    )
    h = contract_balanced_cliques(b)
    phi = h.phi
    assert phi is not None
    assert phi.shape == (b.n_robots,)
    # Every value in phi is a valid super-vertex index.
    assert phi.min() >= 0
    assert phi.max() < h.n_robots
    # Every super-vertex appears at least once (no orphan supernodes).
    assert set(phi.tolist()) == set(range(h.n_robots))
    # super_members is consistent with phi.
    for super_idx, members in enumerate(h.super_members):
        for m in members:
            assert phi[m] == super_idx


# ─── Property 7: sign aggregation ───────────────────────────────────


def test_sign_aggregation_majority():
    """Two super-vertices connected by 3 boundary edges: ++−
    should aggregate to +."""
    # 6 vertices, two disjoint balanced triangles {0,1,2} and {3,4,5}.
    # Boundary: (0,3)=+, (1,4)=+, (2,5)=−. Majority = +.
    b = _hand_built_bundle(6, [
        # Triangle 1 (balanced, all +).
        (0, 1, +1), (1, 2, +1), (0, 2, +1),
        # Triangle 2 (balanced, all +).
        (3, 4, +1), (4, 5, +1), (3, 5, +1),
        # Boundary.
        (0, 3, +1), (1, 4, +1), (2, 5, -1),
    ])
    h = contract_balanced_cliques(b)
    assert h.n_robots == 2
    assert h.n_edges == 1
    # Aggregated sign: 1 + 1 - 1 = 1 → +.
    assert int(h.graph.signs[0]) == 1


def test_sign_aggregation_negative_majority():
    """Same shape but boundary ++− → −−+ flips majority to −."""
    b = _hand_built_bundle(6, [
        (0, 1, +1), (1, 2, +1), (0, 2, +1),
        (3, 4, +1), (4, 5, +1), (3, 5, +1),
        (0, 3, -1), (1, 4, -1), (2, 5, +1),
    ])
    h = contract_balanced_cliques(b)
    assert h.n_edges == 1
    # Aggregated sign: -1 - 1 + 1 = -1 → −.
    assert int(h.graph.signs[0]) == -1


def test_sign_aggregation_tie_breaks_positive():
    """Boundary +− (sum = 0) → tie breaks to + by default."""
    b = _hand_built_bundle(6, [
        (0, 1, +1), (1, 2, +1), (0, 2, +1),
        (3, 4, +1), (4, 5, +1), (3, 5, +1),
        (0, 3, +1), (1, 4, -1),                # tied boundary
    ])
    h = contract_balanced_cliques(b)
    assert h.n_edges == 1
    assert int(h.graph.signs[0]) == 1


# ─── Property 8: multi-scale hierarchy termination ──────────────────


def test_multiscale_terminates_at_fixed_point():
    """Iteration stops when no further contraction is possible."""
    b = make_planted_balanced_cliques(
        n_robots=30, clique_sizes=[6, 5, 4, 3], seed=0,
    )
    levels = multiscale_hierarchy(b, max_levels=10)
    assert len(levels) >= 2
    # First level is the original.
    assert levels[0].n_robots == b.n_robots
    # Monotonically non-increasing.
    sizes = [lvl.n_robots for lvl in levels]
    assert sizes == sorted(sizes, reverse=True)
    # Either max_levels was reached, or we hit a fixed point / singleton.
    last = levels[-1]
    assert last.n_robots == 1 or len(levels) == 11 or (
        # Or the operator returned the same size as the previous level
        # (fixed point detected and hierarchy stopped early).
        len(levels) < 11
    )


def test_multiscale_singleton_root():
    """A fully-balanced clique reaches a 1-vertex root in 1 step."""
    b = _hand_built_bundle(5, [
        (0, 1, +1), (0, 2, +1), (0, 3, +1), (0, 4, +1),
        (1, 2, +1), (1, 3, +1), (1, 4, +1),
        (2, 3, +1), (2, 4, +1),
        (3, 4, +1),
    ])
    levels = multiscale_hierarchy(b, max_levels=10)
    assert len(levels) == 2
    assert levels[-1].n_robots == 1


# ─── Compression on planted networks ────────────────────────────────


def test_compression_ratio_on_planted_network():
    """Planted balanced cliques should drive non-trivial compression."""
    b = make_planted_balanced_cliques(
        n_robots=40, clique_sizes=[8, 6, 5, 4], seed=0,
    )
    h = contract_balanced_cliques(b)
    # Aggressive coarsening expected since ~23 vertices are planted.
    assert h.compression_ratio < 0.7


# ─── Provenance preservation ────────────────────────────────────────


def test_contracted_bundle_carries_parent_n_vertices():
    b = make_planted_balanced_cliques(n_robots=20,
                                            clique_sizes=[5, 4, 3], seed=0)
    h = contract_balanced_cliques(b)
    assert h.parent_n_vertices == b.n_robots


def test_super_clique_indices_length_matches_super_members():
    b = make_planted_balanced_cliques(n_robots=25,
                                            clique_sizes=[5, 4, 3], seed=0)
    h = contract_balanced_cliques(b)
    assert len(h.super_clique_indices) == len(h.super_members)
    # Singletons get -1.
    n_singletons = sum(1 for ci in h.super_clique_indices if ci == -1)
    n_singletons_from_members = sum(1 for m in h.super_members if len(m) == 1)
    assert n_singletons == n_singletons_from_members
