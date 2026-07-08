"""Tests for deterministic mechanism selection + least-squares Σ (LiNGAM-SH step 3B) — spec §8, no search."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.eval.causal import (
    MechanismProposal,
    cross_view_verify,
    factorize_from_proposals,
    fit_sigma_least_squares,
    select_mechanism_subset,
)


def _prop(name, tail, head, strength=1.0, sign=1):
    return MechanismProposal(name=name, tail=tail, head=head, strength=strength, sign=sign,
                             confidence=1.0, source="test", evidence={})


def test_lstsq_sigma_reconstructs_single_edge() -> None:
    fac = fit_sigma_least_squares(["x", "y"], [("x", "y", 0.7)], [_prop("m", ("x",), ("y",), 0.3, 1)])
    assert float(fac.sigma[0, 0]) == pytest.approx(0.7)         # fitted, not the 0.3 default
    assert fac.metrics["fro_error"] == pytest.approx(0.0, abs=1e-9)


def test_lstsq_improves_over_default_strength() -> None:
    edges = [("x", "y", 0.9)]
    prop = _prop("m", ("x",), ("y",), 0.3, 1)                   # default strength 0.3 is wrong
    default = factorize_from_proposals(["x", "y"], edges, [prop])
    lstsq = fit_sigma_least_squares(["x", "y"], edges, [prop])
    assert lstsq.metrics["fro_error"] < default.metrics["fro_error"]


def test_exhaustive_selects_correct_over_wrong() -> None:
    edges = [("x", "y", 0.8)]
    correct = _prop("correct", ("x",), ("y",))
    wrong = _prop("wrong", ("y",), ("x",))
    res = select_mechanism_subset(["x", "y"], edges, [correct, wrong], method="exhaustive")
    assert res.selected == (correct,) and wrong in res.rejected


def test_exhaustive_selects_two_when_both_needed() -> None:
    edges = [("a", "c", 1.0), ("b", "d", 1.0)]
    m1, m2 = _prop("m1", ("a",), ("c",)), _prop("m2", ("b",), ("d",))
    res = select_mechanism_subset(["a", "b", "c", "d"], edges, [m1, m2], method="exhaustive")
    assert len(res.selected) == 2 and m1 in res.selected and m2 in res.selected


def test_greedy_is_deterministic() -> None:
    edges = [("a", "c", 1.0), ("b", "d", 1.0)]
    props = [_prop("m1", ("a",), ("c",)), _prop("m2", ("b",), ("d",))]
    r1 = select_mechanism_subset(["a", "b", "c", "d"], edges, props, method="greedy")
    r2 = select_mechanism_subset(["a", "b", "c", "d"], edges, props, method="greedy")
    assert r1.selected == r2.selected


def test_empty_proposals_returns_baseline() -> None:
    res = select_mechanism_subset(["a", "b"], [("a", "b", 0.5)], [], method="exhaustive")
    assert res.selected == () and res.rejected == ()
    assert np.allclose(res.factorization.b_hat, 0.0)


def test_complexity_penalty_prefers_grouped_over_redundant_pairwise() -> None:
    # equal weights → grouped (1 mech) and two pairwise (2 mech) both reconstruct exactly; penalty picks grouped
    edges = [("near", "reward", 0.9), ("progress", "reward", 0.9)]
    grouped = _prop("grouped", ("near", "progress"), ("reward",), 0.9, 1)
    p1, p2 = _prop("p1", ("near",), ("reward",), 0.9, 1), _prop("p2", ("progress",), ("reward",), 0.9, 1)
    res = select_mechanism_subset(["near", "progress", "reward"], edges, [grouped, p1, p2], method="exhaustive")
    assert res.selected == (grouped,)                          # one grouped mechanism beats the two pairwise


def test_overlapping_mechanisms_deterministic() -> None:
    edges = [("a", "c", 1.0)]
    m1, m2 = _prop("m1", ("a",), ("c",)), _prop("m2", ("a",), ("c",))
    f1 = fit_sigma_least_squares(["a", "c"], edges, [m1, m2])
    f2 = fit_sigma_least_squares(["a", "c"], edges, [m1, m2])
    assert np.array_equal(f1.b_hat, f2.b_hat)
    i = f1.variables.index
    assert f1.b_hat[i("c"), i("a")] == pytest.approx(1.0)       # split across the two identical bases, sums to 1.0


def test_sign_mismatch_reported() -> None:
    # proposal says sign +1 but the true edge is negative → fitted σ negative → mismatch
    fac = fit_sigma_least_squares(["x", "y"], [("x", "y", -0.8)], [_prop("m", ("x",), ("y",), 0.8, 1)])
    assert fac.metrics["n_sign_mismatch"] == pytest.approx(1.0)
    res = select_mechanism_subset(["x", "y"], [("x", "y", -0.8)], [_prop("m", ("x",), ("y",), 0.8, 1)])
    assert res.candidate_scores["m"]["sign_match"] == 0.0


def test_selected_converts_to_hypergraph_and_cross_view(tmp_path) -> None:
    edges = [("near", "reward", 0.9), ("progress", "reward", 0.9)]
    grouped = _prop("grouped_mech", ("near", "progress"), ("reward",), 0.9, 1)
    res = select_mechanism_subset(["near", "progress", "reward"], edges, [grouped], method="exhaustive")
    cg = res.factorization.to_causal_hypergraph("Selected")
    assert cg.check_acyclicity().acyclic
    assert cross_view_verify(cg, tmp_path / "sel.hymeko").agree


def test_selection_reduces_reconstruction_error() -> None:
    edges = [("near", "reward", 0.9), ("progress", "reward", 0.9)]
    grouped = _prop("g", ("near", "progress"), ("reward",), 0.9, 1)
    res = select_mechanism_subset(["near", "progress", "reward"], edges, [grouped], method="exhaustive")
    assert res.factorization.metrics["fro_error"] < res.factorization.metrics["fro_error_baseline"]
