"""Tests for the generic MetaWorld info-signal CIP runner (any task) — pure helpers + a tiny real-env run."""
from __future__ import annotations

import importlib.util

import pytest

from hymeko_rl.eval.cip.metaworld_generic_cip import GENERIC_TASKS, _disagreement, _reward_parents, run_generic_cip

_HAS_METAWORLD = importlib.util.find_spec("metaworld") is not None


def test_reward_parents_filters_by_effect() -> None:
    summary = {"strongest_edges": [["near_fraction", "total_reward", 0.9],
                                   ["action_noise", "near_fraction", -0.3],
                                   ["grasp_fraction", "total_reward", 0.4]]}
    parents = _reward_parents(summary)
    assert {e[0] for e in parents} == {"near_fraction", "grasp_fraction"}   # only edges INTO total_reward


def test_disagreement_bounds() -> None:
    aligned = _disagreement([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "t")     # reward tracks task → low
    inverted = _disagreement([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], "t")   # reward anti-tracks → high
    assert aligned < inverted


def test_generic_tasks_registry() -> None:
    assert "push" in GENERIC_TASKS and GENERIC_TASKS["push"] == "SawyerPushV3Policy"
    assert set(GENERIC_TASKS) == {"push", "pick-place", "door-open", "button-press", "reach"}


@pytest.mark.skipif(not _HAS_METAWORLD, reason="metaworld package not installed")
def test_generic_cip_runs_and_cross_view_agrees(tmp_path) -> None:
    summary = run_generic_cip("push", n=14, seed=0, out_dir=tmp_path)
    assert summary["source"].startswith("generic MetaWorld")
    assert (tmp_path / "causal_push_generic.hymeko").exists()
    assert summary["cross_view"]["agree"], summary["cross_view"]
    assert "total_reward" in summary["causal_order"]
