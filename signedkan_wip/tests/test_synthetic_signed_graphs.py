"""Tests for the synthetic signed-graph generators.

Each generator produces a fixed-seed graph; these tests pin down the
expected structural properties so a future generator change cannot
silently break the architectural validation harness.
"""
from __future__ import annotations

import numpy as np
import pytest

from signedkan_wip.src.synthetic_signed_graphs import (
    easy_sbm,
    needle_in_haystack,
    feature_conditioned,
    oracle_node_classification_auc,
    GENERATORS,
)


# ─── Determinism ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(GENERATORS.keys()))
def test_generator_deterministic(name):
    """Same seed → identical graph."""
    s1 = GENERATORS[name](seed=42)
    s2 = GENERATORS[name](seed=42)
    assert s1.graph.n_nodes == s2.graph.n_nodes
    np.testing.assert_array_equal(s1.graph.edges, s2.graph.edges)
    np.testing.assert_array_equal(s1.graph.signs, s2.graph.signs)
    np.testing.assert_array_equal(s1.labels, s2.labels)


@pytest.mark.parametrize("name", list(GENERATORS.keys()))
def test_generator_seed_actually_varies(name):
    """Different seeds → different graphs (probabilistically;
    sufficient if any of {edges, signs, labels} differs).
    Catches accidental seed-ignored generators."""
    s1 = GENERATORS[name](seed=1)
    s2 = GENERATORS[name](seed=2)
    if s1.graph.edges.shape == s2.graph.edges.shape:
        differs = (
            not np.array_equal(s1.graph.edges, s2.graph.edges)
            or not np.array_equal(s1.graph.signs, s2.graph.signs)
        )
        assert differs, f"{name}: seed=1 and seed=2 produced identical graphs"


# ─── Structural properties ──────────────────────────────────────────


def test_easy_sbm_basic_shape():
    s = easy_sbm(n_per_block=100, n_blocks=2, seed=0)
    assert s.graph.n_nodes == 200
    assert s.labels.shape == (200,)
    assert set(np.unique(s.labels).tolist()) == {0, 1}
    assert (s.labels[:100] == 0).all()
    assert (s.labels[100:] == 1).all()
    # SBM should have a non-trivial number of edges.
    assert s.graph.edges.shape[0] > 1000


def test_easy_sbm_within_block_edges_are_mostly_positive():
    """Same-block edges should be ~92% +1 (per generator default
    p_within_pos=0.6 / p_within_pos+p_within_neg=0.65)."""
    s = easy_sbm(n_per_block=200, seed=0)
    same = s.labels[s.graph.edges[:, 0]] == s.labels[s.graph.edges[:, 1]]
    pos = s.graph.signs == 1
    same_block_pos_frac = (same & pos).sum() / max(1, same.sum())
    assert same_block_pos_frac > 0.85, (
        f"same-block +1 frac {same_block_pos_frac:.3f}, expected > 0.85"
    )


def test_easy_sbm_cross_block_edges_are_mostly_negative():
    s = easy_sbm(n_per_block=200, seed=0)
    cross = s.labels[s.graph.edges[:, 0]] != s.labels[s.graph.edges[:, 1]]
    neg = s.graph.signs == -1
    cross_block_neg_frac = (cross & neg).sum() / max(1, cross.sum())
    assert cross_block_neg_frac > 0.85, (
        f"cross-block -1 frac {cross_block_neg_frac:.3f}, expected > 0.85"
    )


def test_needle_in_haystack_two_balanced_communities():
    s = needle_in_haystack(n_per_block=200, seed=0)
    assert s.graph.n_nodes == 400
    n0 = (s.labels == 0).sum()
    n1 = (s.labels == 1).sum()
    assert n0 == n1 == 200


def test_needle_in_haystack_signal_is_within_block():
    """Signal cycles are entirely within one block; a graph with
    only signal+noise should be majority cross-block edges (noise
    dominates)."""
    s = needle_in_haystack(n_per_block=200,
                            n_signal_cycles_per_block=10, seed=0)
    same = s.labels[s.graph.edges[:, 0]] == s.labels[s.graph.edges[:, 1]]
    # Background ER edges are independent of label, so cross-block
    # frac should be roughly 0.5 (one half of all label pairs).
    cross_frac = (~same).sum() / s.graph.edges.shape[0]
    assert 0.3 < cross_frac < 0.7


def test_feature_conditioned_features_separate_modes():
    s = feature_conditioned(n_per_mode=200, feat_dim=4, seed=0)
    assert s.features is not None
    assert s.features.shape == (400, 4)
    # First feature axis should separate modes (mean-shifted by ±1).
    feat_axis_0_mode_0 = s.features[s.labels == 0, 0].mean()
    feat_axis_0_mode_1 = s.features[s.labels == 1, 0].mean()
    assert feat_axis_0_mode_0 > 0.5
    assert feat_axis_0_mode_1 < -0.5


def test_feature_conditioned_oracle_is_perfect():
    """Logistic regression on the features should perfectly
    recover the mode labels — confirms the features carry enough
    signal that a learnable M_e using them has a high ceiling."""
    s = feature_conditioned(n_per_mode=250, seed=0)
    auc = oracle_node_classification_auc(s, seed=0)
    assert auc > 0.95, (
        f"feature_conditioned LR oracle AUC = {auc:.3f}; expected > 0.95 "
        "(features should be linearly separable)"
    )


# ─── Edges have valid structure ────────────────────────────────────


@pytest.mark.parametrize("name", list(GENERATORS.keys()))
def test_edges_are_well_formed(name):
    """Edges are int, sorted u<v, no self-loops, signs in {±1},
    indices < n_nodes."""
    s = GENERATORS[name](seed=0)
    edges = s.graph.edges
    signs = s.graph.signs
    n = s.graph.n_nodes
    assert edges.dtype.kind == "i"
    assert signs.dtype.kind == "i"
    assert edges.shape[1] == 2
    # No self-loops.
    assert (edges[:, 0] != edges[:, 1]).all()
    # u < v.
    assert (edges[:, 0] < edges[:, 1]).all()
    # In range.
    assert (edges >= 0).all()
    assert (edges < n).all()
    # Signs are ±1.
    assert set(np.unique(signs).tolist()).issubset({-1, 1})
    # No duplicate edges.
    rows = list(map(tuple, edges))
    assert len(set(rows)) == len(rows)
