"""Stage 0 tests — the degree/sign-preserving signed-incidence scramble (the H2 causal ablation).

Verifies the preserved invariants (node count, per-sign edge counts, signed degree sequence, simple-graph),
that the scramble actually destroys incidence, and determinism (same seed → same scramble; different → different).
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.agents.hypergraph_state import HypergraphState
from hymeko_rl.experiments.incidence_scramble import (
    _undirected_signed_edges,
    scramble_signed_incidence,
    scramble_stats,
    signed_degree_sequences,
)
from hymeko_rl.experiments.structural_probe import build_toy_graph


def _pos_neg_counts(hg: HypergraphState) -> tuple[int, int]:
    edges = _undirected_signed_edges(hg)
    pos = sum(1 for _u, _v, s in edges if s > 0)
    return pos, len(edges) - pos


def test_edge_counts_preserved_per_sign() -> None:
    hg = build_toy_graph()
    o_pos, o_neg = _pos_neg_counts(hg)
    for seed in range(6):
        s_pos, s_neg = _pos_neg_counts(scramble_signed_incidence(hg, seed=seed))
        assert (s_pos, s_neg) == (o_pos, o_neg)


def test_node_count_preserved() -> None:
    hg = build_toy_graph()
    for seed in range(6):
        scr = scramble_signed_incidence(hg, seed=seed)
        assert scr.n_vertices == hg.n_vertices
        assert scr.vertex_labels == hg.vertex_labels


def test_signed_degree_sequence_preserved_exactly() -> None:
    hg = build_toy_graph()
    o_pos, o_neg = signed_degree_sequences(hg)
    for seed in range(6):
        s_pos, s_neg = signed_degree_sequences(scramble_signed_incidence(hg, seed=seed))
        assert s_pos == o_pos and s_neg == o_neg


def test_scramble_changes_incidence() -> None:
    hg = build_toy_graph()
    orig = set(_undirected_signed_edges(hg))
    # over a spread of seeds the scramble must actually move edges (destroy structure), not return the input.
    changed_any = False
    for seed in range(6):
        scr = set(_undirected_signed_edges(scramble_signed_incidence(hg, seed=seed)))
        if scr != orig:
            changed_any = True
        assert scramble_stats(hg, scramble_signed_incidence(hg, seed=seed)).n_edges_changed >= 0
    assert changed_any, "scramble never changed the incidence across 6 seeds"


def test_default_seed_actually_destroys_structure() -> None:
    # the pilot's canonical scramble (seed 0) must change incidence, else the H2 control is a no-op.
    hg = build_toy_graph()
    st = scramble_stats(hg, scramble_signed_incidence(hg, seed=0))
    assert st.n_edges_changed >= 1
    assert st.frac_incidence_changed > 0.0
    assert st.signed_degree_preserved


def test_same_seed_is_deterministic() -> None:
    hg = build_toy_graph()
    a = _undirected_signed_edges(scramble_signed_incidence(hg, seed=7))
    b = _undirected_signed_edges(scramble_signed_incidence(hg, seed=7))
    assert a == b


def test_different_seed_differs() -> None:
    hg = build_toy_graph()
    a = _undirected_signed_edges(scramble_signed_incidence(hg, seed=3))
    b = _undirected_signed_edges(scramble_signed_incidence(hg, seed=4))
    assert a != b


def test_result_is_a_simple_signed_graph() -> None:
    # regression: the first implementation swapped sign classes independently and produced a pair carrying
    # both +1 and -1 (a parallel signed edge). _undirected_signed_edges raises on that, so it must not happen.
    hg = build_toy_graph()
    for seed in range(8):
        scr = scramble_signed_incidence(hg, seed=seed)
        edges = _undirected_signed_edges(scr)                 # raises if a pair carries two signs
        pairs = [(u, v) for u, v, _s in edges]
        assert len(pairs) == len(set(pairs)), "a pair carries two signs (parallel signed edge)"


def test_dense_adj_is_finite_and_shaped() -> None:
    hg = build_toy_graph()
    scr = scramble_signed_incidence(hg, seed=0)
    import torch
    a_pos, a_neg = scr.dense_signed_adj()
    assert a_pos.shape == (hg.n_vertices, hg.n_vertices)
    assert bool(torch.isfinite(a_pos).all()) and bool(torch.isfinite(a_neg).all())


def test_antisymmetric_graph_is_rejected() -> None:
    # a kinematic-style graph (mirror arcs carry OPPOSITE signs) is not a symmetric signed graph → rejected.
    edges = np.asarray([(0, 1), (1, 0), (1, 2), (2, 1)], np.int64)
    signs = np.asarray([+1, -1, +1, -1], np.int64)            # down +1 / up -1 (from_mjcf convention)
    hg = HypergraphState(vertex_labels=("a", "b", "c"), edges=edges, signs=signs, topo_hash="anti")
    with pytest.raises(ValueError, match="symmetric"):
        scramble_signed_incidence(hg, seed=0)


def test_negative_swaps_rejected() -> None:
    hg = build_toy_graph()
    with pytest.raises(ValueError, match="swaps_per_edge"):
        scramble_signed_incidence(hg, seed=0, swaps_per_edge=-1)
