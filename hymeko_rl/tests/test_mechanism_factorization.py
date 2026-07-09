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


def _reward_prop(tail: tuple[str, ...], loadings: "dict[str, float] | None" = None) -> MechanismProposal:
    ev: dict = {"tail": list(tail), "output": "y"}
    if loadings is not None:
        ev["loadings"] = loadings
    strength = sum(abs(v) for v in (loadings or {}).values()) / max(1, len(loadings or {})) if loadings else 1.0
    return MechanismProposal(name="rm", tail=tail, head=("y",), strength=strength, sign=1,
                             confidence=1.0, source="reward_terms", evidence=ev)


def test_weighted_beats_binary_on_asymmetric_two_parent(tmp_path) -> None:
    """A. per-tail loadings reconstruct an asymmetric {x1,x2}→{y} reward better than the shared-strength binary fit."""
    from hymeko_rl.eval.causal import fit_loadings_least_squares
    edges = [("x1", "y", 0.95), ("x2", "y", 0.20)]                    # asymmetric parents
    prop = _reward_prop(("x1", "x2"))
    binary = factorize_from_proposals(["x1", "x2", "y"], edges, [prop])
    weighted = fit_loadings_least_squares(["x1", "x2", "y"], edges, [prop])
    assert weighted.metrics["explained_energy"] > binary.metrics["explained_energy"]
    assert weighted.metrics["explained_energy"] == pytest.approx(1.0, abs=1e-9)   # exact for single-output


def test_weighted_equals_raw_pairwise_single_output() -> None:
    """B. with all parents in the tail, per-tail loadings equal the pairwise weights (B[y,x_i])."""
    from hymeko_rl.eval.causal import fit_loadings_least_squares
    edges = [("x1", "y", 0.9), ("x2", "y", 0.7)]
    fac = fit_loadings_least_squares(["x1", "x2", "y"], edges, [_reward_prop(("x1", "x2"))])
    i = fac.variables.index
    assert fac.a_in[i("x1"), 0] == pytest.approx(0.9) and fac.a_in[i("x2"), 0] == pytest.approx(0.7)
    assert fac.metrics["fro_error"] == pytest.approx(0.0, abs=1e-9)


def test_binary_mode_unchanged() -> None:
    """C. the binary shared-strength lstsq path is unchanged (one strength = mean over the two entries)."""
    from hymeko_rl.eval.causal import fit_sigma_least_squares
    edges = [("x1", "y", 0.9), ("x2", "y", 0.7)]
    fac = fit_sigma_least_squares(["x1", "x2", "y"], edges, [_reward_prop(("x1", "x2"))])
    assert float(fac.sigma[0, 0]) == pytest.approx(0.8)              # (0.9+0.7)/2, one shared strength


def test_declared_loadings_used_when_requested() -> None:
    """D. declared reward weights (evidence['loadings']) can be used as loadings instead of fitting."""
    from hymeko_rl.eval.causal import fit_loadings_least_squares
    edges = [("x1", "y", 0.9), ("x2", "y", 0.7)]
    prop = _reward_prop(("x1", "x2"), loadings={"x1": 3.0, "x2": 5.0})
    fac = fit_loadings_least_squares(["x1", "x2", "y"], edges, [prop], use_declared=True)
    i = fac.variables.index
    assert fac.a_in[i("x1"), 0] == pytest.approx(3.0) and fac.a_in[i("x2"), 0] == pytest.approx(5.0)


def test_weighted_sign_mismatch_reported() -> None:
    """E. a fitted loading whose sign disagrees with the declared loading is counted."""
    from hymeko_rl.eval.causal import fit_loadings_least_squares
    edges = [("x1", "y", -0.8)]                                       # true edge negative
    prop = _reward_prop(("x1",), loadings={"x1": 1.0})               # declared positive
    fac = fit_loadings_least_squares(["x1", "y"], edges, [prop])
    assert fac.metrics["n_sign_mismatch"] == pytest.approx(1.0)


def test_weighted_converts_and_cross_view(tmp_path) -> None:
    """F+G. the weighted mechanism (same structure) converts to a CausalHypergraph and cross-view-verifies."""
    from hymeko_rl.eval.causal import cross_view_verify, fit_loadings_least_squares
    edges = [("x1", "y", 0.9), ("x2", "y", 0.7)]
    fac = fit_loadings_least_squares(["x1", "x2", "y"], edges, [_reward_prop(("x1", "x2"))])
    cg = fac.to_causal_hypergraph("Weighted")
    assert cg.check_acyclicity().acyclic
    assert cross_view_verify(cg, tmp_path / "w.hymeko").agree


def test_score_mechanism_set_baseline() -> None:
    b = np.array([[0.0, 0.5], [0.0, 0.0]])
    scores = score_mechanism_set(b, np.zeros_like(b), n_mechanisms=0, n_parameters=0)
    assert scores["fro_error"] == pytest.approx(0.5) and scores["relative_error"] == pytest.approx(1.0)
    assert scores["explained_energy"] == pytest.approx(0.0)
