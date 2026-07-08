"""Tests for CIP over the MetaWorld task templates (coffee-push, dial-turn) — synthetic, no metaworld dep.

Pins: the synthetic rollouts produce a monitor-scored spread (not all-pass / all-fail), the pipeline runs
end-to-end and cross-view-verifies the declared DAG, and the discovered chain is sane (approach/engage precedes
the outcome in the causal order). The real monitors are exercised unchanged (read-only).
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from hymeko_rl.eval.cip.metaworld_cip import TEMPLATES, _coffee_rollout, _dial_rollout, run_metaworld_cip


def test_no_metaworld_dependency() -> None:
    """The template CIP path imports and runs without the real metaworld package."""
    assert "metaworld" not in sys.modules


@pytest.mark.parametrize("gen", [_coffee_rollout, _dial_rollout])
def test_rollout_produces_verdict_and_continuous(gen) -> None:
    rng = np.random.default_rng(0)
    r = gen(rng)
    assert hasattr(r.verdict, "monitor_pass") and hasattr(r.verdict, "progress_score")
    assert len(r.continuous) == 3 and all(np.isfinite(v) for v in r.continuous.values())


@pytest.mark.parametrize("task", ["coffee_push", "dial_turn"])
def test_synthetic_batch_has_a_success_spread(task) -> None:
    """A batch is neither all-pass nor all-fail — the skill draw induces a real outcome spread (else LiNGAM is moot)."""
    rng = np.random.default_rng(1)
    passes = [TEMPLATES[task].rollout(rng).verdict.monitor_pass for _ in range(80)]
    assert 0 < sum(passes) < len(passes)


@pytest.mark.parametrize("task", ["coffee_push", "dial_turn"])
def test_pipeline_runs_and_cross_view_agrees(task, tmp_path) -> None:
    """End-to-end: monitor → frame → DirectLiNGAM → .hymeko cross-view, with a sane causal order."""
    summary = run_metaworld_cip(task, n=120, seed=0, out_dir=tmp_path)
    assert (tmp_path / f"{task}_summary.json").exists()
    assert (tmp_path / f"causal_{task}.hymeko").exists()
    assert summary["cross_view"]["agree"], summary["cross_view"]
    order = summary["causal_order"]
    root_input = "approach_error" if task == "coffee_push" else "engage_error"
    # the skill-driven input should precede the manipulation outcome (progress_score) in the recovered order
    if root_input in order and "progress_score" in order:
        assert order.index(root_input) < order.index("progress_score"), order


def test_unknown_task_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown task"):
        run_metaworld_cip("pick_place", n=10, seed=0, out_dir=tmp_path)
