"""Tests for the CausalHypergraph → HSiKAN HypergraphState bridge (step 1 of the mechanism model)."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.eval.causal.hsikan_mechanism import (
    causal_hg_to_structure,
    lingam_to_signed_adjacency,
    node_features_from_frame,
    signed_adjacency_split,
)
from hymeko_rl.eval.causal.hymeko_emit import CausalHypergraph, Mechanism
from hymeko_rl.eval.causal.lingam import LingamResult


def _B() -> np.ndarray:
    # x_b = 0.9 x_a - 0.3 x_c  (B[effect, cause]); a, c exogenous.
    b = np.zeros((3, 3))
    b[1, 0] = 0.9
    b[1, 2] = -0.3
    return b


def _coffee_push() -> CausalHypergraph:
    return CausalHypergraph.from_mechanisms("coffee_push", [
        Mechanism.pairwise("near_fraction", "total_reward", +0.9),
        Mechanism.pairwise("grasp_fraction", "total_reward", +0.4),
        Mechanism.pairwise("action_noise", "total_reward", -0.3),
        Mechanism.pairwise("action_noise", "near_fraction", -0.2),
    ])


def test_bridge_partitions_variables_and_hubs() -> None:
    st = causal_hg_to_structure(_coffee_push(), sink="total_reward")
    # 4 observed variables + 4 mechanism hubs = 8 vertices; the two index maps partition them.
    assert st.n_vertices == 8
    assert set(st.variable_index) == {"near_fraction", "grasp_fraction", "action_noise", "total_reward"}
    assert len(st.hub_index) == 4
    assert set(st.variable_index.values()).isdisjoint(st.hub_index.values())
    assert len(set(st.variable_index.values()) | set(st.hub_index.values())) == 8


def test_bridge_arcs_match_star_projection_with_signs() -> None:
    chg = _coffee_push()
    st = causal_hg_to_structure(chg, sink="total_reward")
    star = chg.star_projection()
    idx = {name: i for i, name in enumerate(list(star.variables) + list(star.hubs))}
    want = {(idx[s], idx[d], int(sg)) for s, d, sg in star.incidence}
    got = {(int(a), int(b), int(s)) for (a, b), s in zip(st.hg.edges, st.hg.signs)}
    assert got == want
    # a negative mechanism (action_noise→total_reward) must contribute a -1 arc (hub→head).
    assert any(s < 0 for s in st.hg.signs)


def test_bridge_dense_adj_is_finite() -> None:
    import torch
    st = causal_hg_to_structure(_coffee_push(), sink="total_reward")
    a_pos, a_neg = st.hg.dense_signed_adj()
    assert a_pos.shape == (8, 8)
    assert bool(torch.isfinite(a_pos).all()) and bool(torch.isfinite(a_neg).all())


def test_unknown_sink_raises() -> None:
    with pytest.raises(ValueError, match="sink"):
        causal_hg_to_structure(_coffee_push(), sink="not_a_variable")


def test_node_features_shape_and_sink_masked() -> None:
    st = causal_hg_to_structure(_coffee_push(), sink="total_reward")
    n = 7
    cols = {v: np.arange(n, dtype=float) + 1.0 for v in st.variable_names}
    x = node_features_from_frame(st, cols, mask_sink=True)
    assert x.shape == (n, st.n_vertices, 1)
    # sink feature zeroed (must be predicted, not read); a non-sink variable is populated.
    assert float(np.abs(x[:, st.variable_index["total_reward"], 0]).sum()) == 0.0
    assert float(np.abs(x[:, st.variable_index["near_fraction"], 0]).sum()) > 0.0


def test_node_features_sink_present_when_not_masked() -> None:
    st = causal_hg_to_structure(_coffee_push(), sink="total_reward")
    cols = {v: np.ones(4, dtype=float) for v in st.variable_names}
    x = node_features_from_frame(st, cols, mask_sink=False)
    assert float(x[:, st.variable_index["total_reward"], 0].sum()) == 4.0


def test_node_features_length_mismatch_raises() -> None:
    st = causal_hg_to_structure(_coffee_push(), sink="total_reward")
    cols = {"near_fraction": np.ones(5), "grasp_fraction": np.ones(4),
            "action_noise": np.ones(5), "total_reward": np.ones(5)}
    with pytest.raises(ValueError, match="length"):
        node_features_from_frame(st, cols)


# ── LiNGAM → signed hypergraph translation (B = A⁺ − A⁻) ─────────────────────────────────────────────────
def test_split_reconstructs_B_exactly() -> None:
    b = _B()
    a_pos, a_neg = signed_adjacency_split(b)
    assert np.allclose(a_pos - a_neg, b)                     # the defining identity


def test_split_signs_and_disjoint_support() -> None:
    a_pos, a_neg = signed_adjacency_split(_B())
    assert (a_pos >= 0).all() and (a_neg >= 0).all()
    assert np.all(a_pos * a_neg == 0.0)                      # excitatory / inhibitory disjoint
    assert a_pos[1, 0] == 0.9 and a_neg[1, 2] == 0.3         # + coefficient in A⁺, |−coefficient| in A⁻


def test_operator_identity_reproduces_lingam_prediction() -> None:
    # the correspondence: the signed operator applied to x reproduces LiNGAM's Bx (HSiKAN's linear restriction).
    b = _B()
    a_pos, a_neg = signed_adjacency_split(b)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 8))                          # (n_vars, batch)
    assert np.allclose((a_pos - a_neg) @ x, b @ x)


def test_split_prunes_below_min_abs() -> None:
    b = _B()
    b[1, 2] = -0.05                                          # a tiny (noise) coefficient
    a_pos, a_neg = signed_adjacency_split(b, min_abs=0.1)
    assert a_neg[1, 2] == 0.0 and a_pos[1, 0] == 0.9         # tiny pruned, real kept


def test_row_normalize_bounds_absolute_degree() -> None:
    a_pos, a_neg = signed_adjacency_split(_B(), row_normalize=True)
    assert np.all((a_pos + a_neg).sum(axis=1) <= 1.0 + 1e-9)


def test_lingam_result_translation() -> None:
    result = LingamResult(order=[0, 2, 1], adjacency=_B(), names=["a", "b", "c"])
    a_pos, a_neg = lingam_to_signed_adjacency(result)
    assert np.allclose(a_pos - a_neg, result.adjacency)


def test_split_non_square_raises() -> None:
    with pytest.raises(ValueError, match="square"):
        signed_adjacency_split(np.zeros((2, 3)))


def test_split_negative_min_abs_raises() -> None:
    with pytest.raises(ValueError, match="min_abs"):
        signed_adjacency_split(_B(), min_abs=-1.0)
