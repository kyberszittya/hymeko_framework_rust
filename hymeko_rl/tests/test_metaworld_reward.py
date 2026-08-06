"""Tests for the HyMeKo-declared MetaWorld reward (declare + reconstruct + ablate) — pure + tiny real-env."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hymeko_rl.env.reward import read_reward_terms
from hymeko_rl.eval.cip.metaworld_reward import (
    _TERM_TO_COMPONENT,
    ablate_reward,
    evaluate_reward_fidelity,
    fit_reward_weights,
    hymeko_reward,
)

_HAS_METAWORLD = importlib.util.find_spec("metaworld") is not None
_SPEC = "data/robotics/metaworld_reward.hymeko"


def test_reward_hymeko_parses_and_maps() -> None:
    """The .hymeko reward spec parses to (term, weight) and every term has a MetaWorld component mapping."""
    terms = read_reward_terms(_SPEC)
    kinds = [k for k, _w in terms]
    assert kinds == ["mw_in_place", "mw_grasp", "mw_near", "mw_dist"]
    assert all(k in _TERM_TO_COMPONENT for k in kinds)


def test_hymeko_reward_is_weighted_sum() -> None:
    comp = np.array([[1.0, 2.0], [0.0, 3.0]])
    assert list(hymeko_reward(comp, [10.0, 1.0])) == [12.0, 3.0]      # 10*1+1*2 ; 10*0+1*3


def test_fit_recovers_known_weights() -> None:
    rng = np.random.default_rng(0)
    comp = rng.normal(size=(200, 3))
    true_w = np.array([2.0, -1.0, 0.5])
    y = comp @ true_w
    w, r2 = fit_reward_weights(comp, y)
    assert np.allclose(w, true_w, atol=1e-6) and r2 == pytest.approx(1.0)


def test_ablate_drops_and_scales_terms() -> None:
    comp = np.array([[1.0, 1.0, 1.0]])
    terms, weights = ["a", "b", "c"], [2.0, 3.0, 4.0]
    assert list(ablate_reward(comp, terms, weights)) == [9.0]                       # 2+3+4
    assert list(ablate_reward(comp, terms, weights, drop=["b"])) == [6.0]           # 2+0+4
    assert list(ablate_reward(comp, terms, weights, scale={"a": 0.5})) == [8.0]     # 1+3+4


@pytest.mark.skipif(not _HAS_METAWORLD, reason="metaworld package not installed")
def test_reward_fidelity_reconstructs_metaworld(tmp_path) -> None:
    """The HyMeKo Σ weight·term reward reconstructs MetaWorld's dense reward with high R² (fitted)."""
    fid = evaluate_reward_fidelity("push", "SawyerPushV3Policy", n_ep=4)
    assert fid.terms == ("mw_in_place", "mw_grasp", "mw_near", "mw_dist")
    assert fid.r2_fitted > 0.7                       # a faithful decomposition (measured ~0.93 for push)
    assert fid.n_steps > 100
