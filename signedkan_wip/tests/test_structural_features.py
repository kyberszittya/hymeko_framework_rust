"""Tests for the leakage-free StructuralFeature registry.

Layers (CLAUDE.md §3): unit (each extractor — shape, exactness, bounds,
train-only), and the registry contract (concat dim, default-spec parity, failure
cases). Pure numpy/scipy — no network, no torch. Plan:
docs/plans/2026-06-18-leakage-free-input-enrichment/.

Run: pytest -p no:randomly signedkan_wip/tests/test_structural_features.py
"""
from __future__ import annotations

import numpy as np
import pytest

from signedkan_wip.src.baselines import structural_features as sf

# A small planted signed graph (triangle 0-1-2 + pendant 2-3), reused across tests.
EDGES = np.array([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=np.int64)
SIGNS = np.array([1, -1, 1, 1], dtype=np.int64)
N = 4


# --- registry contract ----------------------------------------------------

def test_default_spec_reproduces_structural_features_exactly():
    a = sf.build_node_features(("degree",), EDGES, SIGNS, N)
    b = sf._structural_features(EDGES, SIGNS, N)
    assert np.array_equal(a, b)


def test_feature_dim_and_concat_width_agree():
    spec = ("degree", "cycle_k3", "walk_k3", "ratios")
    feats = sf.build_node_features(spec, EDGES, SIGNS, N)
    assert feats.shape == (N, sf.feature_dim(spec))
    assert sf.feature_dim(spec) == 6 + 3 + 9 + 2
    assert feats.dtype == np.float32


def test_unknown_feature_raises_keyerror_with_available():
    with pytest.raises(KeyError) as exc:
        sf.build_node_features(("nope",), EDGES, SIGNS, N)
    assert "available" in str(exc.value)


def test_empty_spec_raises_valueerror():
    with pytest.raises(ValueError):
        sf.build_node_features((), EDGES, SIGNS, N)


def test_available_features_lists_the_registered_set():
    assert {"degree", "cycle_k3", "cycle_k34", "walk_k3", "ratios"} \
        <= set(sf.available_features())


# --- signed adjacency helper ----------------------------------------------

def test_signed_adjacency_symmetric_pm1_and_zero_diag():
    a_signed, a_abs = sf._signed_adjacency(EDGES, SIGNS, N)
    As, Aa = a_signed.toarray(), a_abs.toarray()
    assert np.array_equal(As, As.T) and np.array_equal(Aa, Aa.T)
    assert set(np.unique(As)).issubset({-1.0, 0.0, 1.0})
    assert np.all(np.diag(As) == 0) and np.all(np.diag(Aa) == 0)
    assert As[0, 1] == 1 and As[1, 2] == -1  # edge signs land correctly


# --- walk profile: exactness + meaning ------------------------------------

def test_walk_profile_shape_and_k1_is_signed_degree():
    feats = sf._walk_profile_features(EDGES, SIGNS, N, K=3)
    assert feats.shape == (N, 9)
    # k=1 signed reach (col 0, pre-transform sign*log1p) ⇒ recover net signed degree.
    a_signed, a_abs = sf._signed_adjacency(EDGES, SIGNS, N)
    s1 = a_signed @ np.ones(N)                 # = pos_deg - neg_deg per node
    recovered = np.sign(feats[:, 0]) * np.expm1(np.abs(feats[:, 0]))
    assert np.allclose(recovered, s1, atol=1e-5)


def test_walk_ratio_columns_match_exact_Ak_over_absAk():
    """The ratio columns equal (A^k·1)/(|A|^k·1) exactly (dense brute force)."""
    a_signed, a_abs = sf._signed_adjacency(EDGES, SIGNS, N)
    As, Aa = a_signed.toarray(), a_abs.toarray()
    feats = sf._walk_profile_features(EDGES, SIGNS, N, K=3)
    sv = np.ones(N)
    tv = np.ones(N)
    for k in range(3):
        sv = As @ sv
        tv = Aa @ tv
        expected = sv / (tv + 1e-6)
        assert np.allclose(feats[:, 3 * k + 2], expected, atol=1e-5)


def test_walk_all_positive_graph_has_unit_ratio():
    """On an all-positive graph A == |A|, so signed reach == total reach (ratio 1)."""
    feats = sf._walk_profile_features(EDGES, np.ones_like(SIGNS), N, K=3)
    reached = feats[:, 1] > 0                    # nodes with nonzero total reach
    for k in range(3):
        assert np.allclose(feats[reached, 3 * k + 2], 1.0, atol=1e-5)


def test_walk_rejects_nonpositive_K():
    with pytest.raises(ValueError):
        sf._walk_profile_features(EDGES, SIGNS, N, K=0)


# --- ratios: bounds + planted triangle ------------------------------------

def test_ratio_features_bounds_and_triangle():
    feats = sf._ratio_features(EDGES, SIGNS, N)
    assert feats.shape == (N, 2)
    unsigned, signed = feats[:, 0], feats[:, 1]
    assert np.all((unsigned >= -1e-6) & (unsigned <= 1.0 + 1e-6))
    assert np.all((signed >= -1.0 - 1e-6) & (signed <= 1.0 + 1e-6))
    # Triangle 0-1-2. Nodes 0,1 have degree 2 (C(2,2)=1 pair) ⇒ clustering 1.
    # Node 2 also has the pendant edge ⇒ degree 3 (C(3,2)=3 pairs), 1 triangle ⇒ 1/3.
    assert np.allclose(unsigned[[0, 1]], 1.0, atol=1e-6)
    assert np.isclose(unsigned[2], 1.0 / 3.0, atol=1e-6)
    assert np.isclose(unsigned[3], 0.0, atol=1e-6)   # pendant: no triangle


def test_ratio_signed_clustering_flips_with_balance():
    """One negative edge in the triangle ⇒ unbalanced ⇒ signed clustering < 0."""
    # EDGES triangle signs: (0,1)=+, (1,2)=−, (0,2)=+ ⇒ one negative ⇒ unbalanced.
    feats = sf._ratio_features(EDGES, SIGNS, N)
    assert feats[0, 1] < 0  # signed clustering of a triangle node is negative


# --- leakage-free: pure function of the given (train) edges ----------------

def test_features_are_pure_function_of_input_edges():
    """Dropping an edge changes only the affected nodes — no hidden global state."""
    spec = ("degree", "walk_k3", "ratios")
    full = sf.build_node_features(spec, EDGES, SIGNS, N)
    fewer = sf.build_node_features(spec, EDGES[:3], SIGNS[:3], N)  # drop pendant 2-3
    assert not np.array_equal(full[3], fewer[3])   # node 3 lost its only edge
    assert not np.array_equal(full[2], fewer[2])   # node 2 lost an incident edge
