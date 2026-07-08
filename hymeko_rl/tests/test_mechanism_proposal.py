"""Tests for deterministic mechanism proposal (LiNGAM-SH step 2) — no factorization, no search.

Covers the spec §8 list: common-child grouping (and the single-parent guard), sign/strength determinism, the
explicit reward / monitor-contract proposals, conversion to a mechanism-form CausalHypergraph, and that the
result passes acyclicity + cross-view. The existing pairwise/CIP suite (§I) runs separately and stays green.
"""
from __future__ import annotations

import pytest

from hymeko_rl.eval.causal import (
    proposals_to_causal_hypergraph,
    propose_common_child,
    propose_monitor_contract,
    propose_reward_terms,
)
from hymeko_rl.eval.causal.hymeko_emit import cross_view_verify

# a pairwise DAG where total_reward has two parents (near_fraction, progress_score) and c has one (a)
_EDGES = [("near_fraction", "total_reward", 0.96), ("progress_score", "total_reward", 0.80), ("a", "c", 0.5)]


def test_common_child_groups_two_parents() -> None:
    props = propose_common_child(_EDGES)
    by_head = {p.head: p for p in props}
    assert ("total_reward",) in by_head
    m = by_head[("total_reward",)]
    assert set(m.tail) == {"near_fraction", "progress_score"}     # two parents grouped into one mechanism
    assert m.source == "common_child" and m.head == ("total_reward",)


def test_single_parent_not_grouped_by_default() -> None:
    props = propose_common_child(_EDGES)                          # min_parents=2 default
    assert all(p.head != ("c",) for p in props)                  # 'c' has a single parent 'a' → not proposed
    # explicitly allowed with min_parents=1
    props1 = propose_common_child(_EDGES, min_parents=1)
    assert any(p.head == ("c",) and p.tail == ("a",) for p in props1)


def test_signs_and_strengths_deterministic() -> None:
    edges = [("x", "y", -0.6), ("z", "y", -0.4)]                 # both negative → summed sign negative
    p = propose_common_child(edges)[0]
    assert p.tail == ("x", "z")                                  # sorted by parent name
    assert p.sign == -1
    assert p.strength == pytest.approx(0.5)                      # mean(|−0.6|,|−0.4|)
    # deterministic: same input → identical proposal
    assert propose_common_child(edges)[0] == p


def test_reward_term_proposal() -> None:
    p = propose_reward_terms(["near_fraction", "contact_score"])
    assert p.tail == ("near_fraction", "contact_score") and p.head == ("total_reward",)
    assert p.source == "reward_terms" and p.confidence == 1.0 and p.sign == 1
    pw = propose_reward_terms(["a", "b"], weights={"a": -0.3, "b": -0.5})   # weighted → sign from Σw
    assert pw.sign == -1 and pw.strength == pytest.approx(0.4)


def test_monitor_contract_proposal() -> None:
    p = propose_monitor_contract(["delivery_score", "progress_score", "stagnation_duration"])
    assert p.head == ("monitor_pass",) and p.source == "monitor_contract" and p.confidence == 1.0
    assert set(p.tail) == {"delivery_score", "progress_score", "stagnation_duration"}


def test_proposals_convert_to_mechanism_hypergraph() -> None:
    props = propose_common_child(_EDGES) + [propose_monitor_contract(["delivery_score", "progress_score"])]
    cg = proposals_to_causal_hypergraph(["near_fraction", "progress_score", "total_reward"], props, name="Coin")
    assert len(cg.mechanisms) == len(props)
    assert "delivery_score" in cg.variables and "monitor_pass" in cg.variables   # unioned in


def test_proposed_graph_is_acyclic() -> None:
    props = propose_common_child(_EDGES)
    cg = proposals_to_causal_hypergraph([], props, name="Coin")
    assert cg.check_acyclicity().acyclic


def test_proposed_graph_cross_view_passes(tmp_path) -> None:
    props = [propose_reward_terms(["near_fraction", "contact_score"]),
             propose_monitor_contract(["delivery_score", "progress_score", "stagnation_duration"])]
    cg = proposals_to_causal_hypergraph([], props, name="CoinAll")
    report = cross_view_verify(cg, tmp_path / "proposed.hymeko")
    assert report.agree, report.notes


def test_pairwise_projection_recovers_grouped_edges() -> None:
    props = propose_common_child(_EDGES)
    cg = proposals_to_causal_hypergraph([], props, name="Coin")
    proj = {(c, e) for c, e, _w in cg.pairwise_projection()}
    assert {("near_fraction", "total_reward"), ("progress_score", "total_reward")} <= proj


def test_common_child_min_parents_guard() -> None:
    with pytest.raises(ValueError, match="min_parents"):
        propose_common_child(_EDGES, min_parents=0)
