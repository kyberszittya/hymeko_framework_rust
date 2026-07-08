"""Tests for deterministic mechanism factorization B ≈ A_out·Σ·A_inᵀ (LiNGAM-SH step 3A) — no search, no fit.

Covers spec §8: the B[effect,cause] convention, degenerate/multi-input/multi-output reconstruction, the empty
baseline, improvement for a matching proposal vs a wrong one, deterministic overlap summation, conversion to a
mechanism-form CausalHypergraph, and cross-view. The existing pairwise/CIP suite runs separately and stays green.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.eval.causal import (
    MechanismProposal,
    build_pairwise_b,
    cross_view_verify,
    factorize_from_proposals,
    score_mechanism_set,
)


def _prop(name: str, tail: tuple[str, ...], head: tuple[str, ...], strength: float, sign: int) -> MechanismProposal:
    return MechanismProposal(name=name, tail=tail, head=head, strength=strength, sign=sign,
                             confidence=1.0, source="test", evidence={})


def test_build_pairwise_b_convention() -> None:
    """B[effect, cause] = weight (row = effect, col = cause)."""
    variables = ["a", "b", "c"]
    b = build_pairwise_b(variables, [("a", "b", 0.5), ("b", "c", -0.3)])
    assert b[1, 0] == pytest.approx(0.5)     # a→b : row b (1), col a (0)
    assert b[2, 1] == pytest.approx(-0.3)    # b→c : row c (2), col b (1)
    assert b[0, 1] == 0.0                     # no b→a


def test_degenerate_mechanism_reconstructs_single_edge() -> None:
    fac = factorize_from_proposals(["x", "y"], [("x", "y", 0.7)], [_prop("m", ("x",), ("y",), 0.7, 1)])
    i = fac.variables.index
    assert fac.b_hat[i("y"), i("x")] == pytest.approx(0.7)
    assert fac.metrics["fro_error"] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(fac.b_hat, fac.a_out @ fac.sigma @ fac.a_in.T)


def test_multi_input_single_output_block() -> None:
    edges = [("near_fraction", "total_reward", 0.9), ("progress_score", "total_reward", 0.9)]
    fac = factorize_from_proposals(["near_fraction", "progress_score", "total_reward"], edges,
                                   [_prop("m", ("near_fraction", "progress_score"), ("total_reward",), 0.9, 1)])
    i = fac.variables.index
    assert fac.b_hat[i("total_reward"), i("near_fraction")] == pytest.approx(0.9)
    assert fac.b_hat[i("total_reward"), i("progress_score")] == pytest.approx(0.9)
    assert fac.metrics["fro_error"] == pytest.approx(0.0, abs=1e-9)      # exact when both weights match strength


def test_multi_input_multi_output_block() -> None:
    fac = factorize_from_proposals(["a", "b", "c", "d"], [], [_prop("m", ("a", "b"), ("c", "d"), 0.5, -1)])
    i = fac.variables.index
    for cause in ("a", "b"):
        for effect in ("c", "d"):
            assert fac.b_hat[i(effect), i(cause)] == pytest.approx(-0.5)   # sign·strength on every tail×head


def test_empty_proposals_gives_zero_bhat_and_baseline_residual() -> None:
    edges = [("a", "b", 0.4)]
    fac = factorize_from_proposals(["a", "b"], edges, [])
    assert np.allclose(fac.b_hat, 0.0)
    assert np.allclose(fac.residual, build_pairwise_b(["a", "b"], edges))
    assert fac.metrics["fro_error"] == pytest.approx(fac.metrics["fro_error_baseline"])
    assert fac.metrics["relative_error"] == pytest.approx(1.0)


def test_matching_proposal_improves_reconstruction() -> None:
    edges = [("near_fraction", "total_reward", 0.96), ("progress_score", "total_reward", 0.80)]
    variables = ["near_fraction", "progress_score", "total_reward"]
    m = _prop("m", ("near_fraction", "progress_score"), ("total_reward",), 0.88, 1)
    fac = factorize_from_proposals(variables, edges, [m])
    assert fac.metrics["fro_error"] < fac.metrics["fro_error_baseline"]    # reconstruction improved
    assert 0.0 < fac.metrics["explained_energy"] < 1.0


def test_wrong_proposal_worse_than_matching() -> None:
    edges = [("near_fraction", "total_reward", 0.96), ("progress_score", "total_reward", 0.80)]
    variables = ["near_fraction", "progress_score", "total_reward"]
    good = factorize_from_proposals(
        variables, edges, [_prop("m", ("near_fraction", "progress_score"), ("total_reward",), 0.88, 1)])
    wrong = factorize_from_proposals(          # reversed direction: total_reward → near_fraction
        variables, edges, [_prop("m", ("total_reward",), ("near_fraction",), 0.88, 1)])
    assert wrong.metrics["fro_error"] > good.metrics["fro_error"]


def test_overlapping_mechanisms_sum_deterministically() -> None:
    edges = [("a", "c", 1.0)]
    m1 = _prop("m1", ("a",), ("c",), 0.6, 1)
    m2 = _prop("m2", ("a",), ("c",), 0.5, 1)
    fac = factorize_from_proposals(["a", "c"], edges, [m1, m2])
    i = fac.variables.index
    assert fac.b_hat[i("c"), i("a")] == pytest.approx(1.1)                 # 0.6 + 0.5 summed
    # deterministic: same input → identical b_hat
    fac2 = factorize_from_proposals(["a", "c"], edges, [m1, m2])
    assert np.array_equal(fac.b_hat, fac2.b_hat)


def test_normalize_invariant_bhat() -> None:
    edges = [("a", "b", 0.9), ("c", "b", 0.9)]
    m = _prop("m", ("a", "c"), ("b",), 0.9, 1)
    f_norm = factorize_from_proposals(["a", "c", "b"], edges, [m], normalize=True)
    f_plain = factorize_from_proposals(["a", "c", "b"], edges, [m], normalize=False)
    assert np.allclose(f_norm.b_hat, f_plain.b_hat)                        # B_hat invariant to normalize


def test_factorization_converts_to_hypergraph_and_cross_view(tmp_path) -> None:
    edges = [("near_fraction", "total_reward", 0.9), ("progress_score", "total_reward", 0.9)]
    fac = factorize_from_proposals(["near_fraction", "progress_score", "total_reward"], edges,
                                   [_prop("mech_reward", ("near_fraction", "progress_score"), ("total_reward",),
                                          0.9, 1)])
    cg = fac.to_causal_hypergraph("FacGraph")
    assert len(cg.mechanisms) == 1 and cg.check_acyclicity().acyclic
    report = cross_view_verify(cg, tmp_path / "fac.hymeko")
    assert report.agree, report.notes


def test_score_mechanism_set_baseline() -> None:
    b = np.array([[0.0, 0.5], [0.0, 0.0]])
    scores = score_mechanism_set(b, np.zeros_like(b), n_mechanisms=0, n_parameters=0)
    assert scores["fro_error"] == pytest.approx(0.5) and scores["relative_error"] == pytest.approx(1.0)
    assert scores["explained_energy"] == pytest.approx(0.0)
