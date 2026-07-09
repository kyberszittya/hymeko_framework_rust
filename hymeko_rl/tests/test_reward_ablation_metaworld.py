"""Tests for MetaWorld reward-ablation Stage A — spec ablation + offline recompute (pure) + a real-env run."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hymeko_rl.env.reward import read_reward_terms
from hymeko_rl.eval.cip.metaworld_reward import ablate_reward, hymeko_reward
from hymeko_rl.eval.cip.reward_ablation_metaworld import ablate_reward_spec

_HAS_METAWORLD = importlib.util.find_spec("metaworld") is not None
_SPEC = "data/robotics/metaworld_reward.hymeko"


def test_ablation_zeros_term_without_mutating_source() -> None:
    """A. dropping mw_grasp zeros its weight; the original .hymeko file is untouched (re-read is unchanged)."""
    before = read_reward_terms(_SPEC)
    spec = ablate_reward_spec(_SPEC, drop=["mw_grasp"])
    ablated = dict(spec.ablated)
    assert ablated["mw_grasp"] == 0.0 and ablated["mw_in_place"] == dict(spec.original)["mw_in_place"]
    assert read_reward_terms(_SPEC) == before          # file not mutated
    assert ("mw_grasp", 0.0) not in spec.active_terms()  # dropped term absent from active terms


def test_unknown_dropped_term_fails() -> None:
    """B. an unknown term to drop raises clearly."""
    with pytest.raises(ValueError, match="unknown reward term"):
        ablate_reward_spec(_SPEC, drop=["mw_nonexistent"])


def test_downweight_is_deterministic() -> None:
    """C. downweighting scales the term deterministically."""
    s1 = ablate_reward_spec(_SPEC, downweight={"mw_grasp": 0.25})
    s2 = ablate_reward_spec(_SPEC, downweight={"mw_grasp": 0.25})
    assert dict(s1.ablated)["mw_grasp"] == pytest.approx(0.25 * dict(s1.original)["mw_grasp"])
    assert s1.ablated == s2.ablated


def test_recompute_differs_only_through_dropped_term() -> None:
    """D. original − ablated reward equals exactly the dropped term's contribution."""
    kinds = ["mw_in_place", "mw_grasp", "mw_near", "mw_dist"]
    weights = [8.0, 1.2, 1.0, 10.0]
    comp = np.array([[0.5, 0.3, 0.2, -0.1], [0.9, 0.0, 0.4, -0.2]])   # per-rollout component totals
    orig = hymeko_reward(comp, weights)
    ablated = ablate_reward(comp, kinds, weights, drop=["mw_grasp"])
    # difference is exactly weight_grasp · component_grasp
    assert np.allclose(orig - ablated, 1.2 * comp[:, 1])


def test_dropped_term_absent_from_active_terms() -> None:
    """E. the dropped term is absent from the ablated spec's active (non-zero) terms."""
    spec = ablate_reward_spec(_SPEC, drop=["mw_grasp"])
    assert "mw_grasp" not in {k for k, _w in spec.active_terms()}
    assert {"mw_in_place", "mw_near", "mw_dist"} <= {k for k, _w in spec.active_terms()}


@pytest.mark.skipif(not _HAS_METAWORLD, reason="metaworld package not installed")
def test_stage_a_runs_recomputes_and_cross_view(tmp_path) -> None:
    """F+G. the full Stage-A run recomputes offline, the mechanism graphs cross-view-verify, and the verdict is set."""
    from hymeko_rl.eval.cip.reward_ablation_metaworld import run_reward_ablation_stage_a
    s = run_reward_ablation_stage_a("pick-place", n=16, seed=0, out_dir=tmp_path)
    assert s["dropped"] == ["mw_grasp"]
    assert s["original"]["cross_view_agree"] and s["grasp_off"]["cross_view_agree"]      # G
    assert "grasp_fraction" not in s["reparented"]["tail"]                               # dropped term re-parented out
    assert s["verdict"] in ("SUPPORTED_at_reward_computation_level", "NOT_SUPPORTED")
    assert (tmp_path / "reward_mechanism_grasp_off.hymeko").exists()                     # F: ablated mechanism emitted
