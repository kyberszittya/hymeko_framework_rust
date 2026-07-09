"""Tests for the HyMeKo reward-SoT → LiNGAM-SH mechanism integration (adapter + comparison + cross-view)."""
from __future__ import annotations

from hymeko_rl.eval.cip.metaworld_reward import (
    _TERM_TO_CIP_VARIABLE,
    hymeko_reward_terms,
    reward_mechanism_proposal,
)
from hymeko_rl.eval.cip.reward_mechanism_integration import _pairwise_proposals, compare_reward_mechanisms


def test_hymeko_reward_terms_loaded() -> None:
    kinds = [k for k, _w in hymeko_reward_terms()]
    assert kinds == ["mw_in_place", "mw_grasp", "mw_near", "mw_dist"]


def test_reward_mechanism_maps_terms_to_cip_variables() -> None:
    # all four terms available → tail is the four mapped CIP variables
    avail = ["near_fraction", "grasp_fraction", "progress_score", "obj_to_target_delta", "total_reward"]
    p = reward_mechanism_proposal(available=avail)
    assert set(p.tail) == {_TERM_TO_CIP_VARIABLE[k] for k in _TERM_TO_CIP_VARIABLE}
    assert p.head == ("total_reward",) and p.source == "reward_terms"


def test_reward_mechanism_filters_to_available() -> None:
    # coffee-push monitor frame only has near_fraction + progress_score → mw_grasp / mw_dist filtered out
    p = reward_mechanism_proposal(available=["near_fraction", "progress_score", "total_reward"])
    assert set(p.tail) == {"near_fraction", "progress_score"}


def test_pairwise_proposals_one_per_edge() -> None:
    props = _pairwise_proposals([("a", "b", 0.5), ("c", "b", -0.3)])
    assert len(props) == 2
    assert props[0].tail == ("a",) and props[0].head == ("b",)
    assert props[1].sign == -1


def test_compare_reward_mechanisms_scores_and_cross_view(tmp_path) -> None:
    """All four candidates scored; the HyMeKo reward mechanism explains B and cross-view-verifies."""
    variables = ["near_fraction", "progress_score", "total_reward"]
    edges = [("near_fraction", "total_reward", 0.9), ("progress_score", "total_reward", 0.7)]
    res = compare_reward_mechanisms(variables, edges, out_path=tmp_path / "cmp.json")
    assert set(res["scores"]) == {"none", "raw_pairwise", "common_child", "hymeko_reward", "hymeko_reward_weighted"}
    assert res["scores"]["none"]["explained_energy"] == 0.0
    assert res["scores"]["hymeko_reward"]["explained_energy"] > 0.0     # the reward mechanism explains B
    # weighted (per-tail loadings) reconstructs the two-parent reward at least as well as binary shared-strength
    assert res["scores"]["hymeko_reward_weighted"]["explained_energy"] >= res["scores"]["hymeko_reward"]["explained_energy"]
    assert set(res["reward_mechanism"]["tail"]) == {"near_fraction", "progress_score"}
    assert res["cross_view"]["agree"] and res["cross_view"]["acyclic"]  # star-expanded, engine-verified
    assert (tmp_path / "cmp.json").exists()
