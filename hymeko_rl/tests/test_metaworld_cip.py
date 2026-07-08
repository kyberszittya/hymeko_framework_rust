"""Tests for CIP over the MetaWorld task templates (coffee-push, dial-turn) — synthetic, no metaworld dep.

Pins: the synthetic rollouts produce a monitor-scored spread (not all-pass / all-fail), the pipeline runs
end-to-end and cross-view-verifies the declared DAG, and the discovered chain is sane (approach/engage precedes
the outcome in the causal order). The real monitors are exercised unchanged (read-only).
"""
from __future__ import annotations

import importlib.util
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


_HAS_METAWORLD = importlib.util.find_spec("metaworld") is not None


@pytest.mark.skipif(not _HAS_METAWORLD, reason="metaworld package not installed")
def test_real_env_pipeline_runs_and_cross_view_agrees(tmp_path) -> None:
    """Real MetaWorld coffee-push rollouts → monitor → DirectLiNGAM → .hymeko cross-view (Phase-2 analog)."""
    from hymeko_rl.eval.cip.metaworld_cip import run_metaworld_cip_real
    summary = run_metaworld_cip_real("coffee_push", n=16, seed=0, out_dir=tmp_path)
    assert summary["source"].startswith("real-env")
    assert 0.0 < summary["monitor_pass_rate"] < 1.0          # action noise induces a real spread
    assert (tmp_path / "causal_coffee_push_real.hymeko").exists()
    assert summary["cross_view"]["agree"], summary["cross_view"]
    # total_reward and near_fraction are among the discovered variables (the reward's exact rank is N-dependent —
    # a measured finding at N=80, not a small-N invariant, so it is not asserted here).
    assert {"total_reward", "near_fraction"} <= set(summary["causal_order"])


def test_real_env_unknown_task_raises(tmp_path) -> None:
    from hymeko_rl.eval.cip.metaworld_cip import run_metaworld_cip_real
    with pytest.raises(ValueError, match="no real-env mapping"):
        run_metaworld_cip_real("dial_turn", n=4, seed=0, out_dir=tmp_path)


# --- multi-seed aggregation (pure logic — no metaworld needed) --------------------------------------------------
def _fake_batch(near_w: float, prog_w: float, pass_rate: float, dis: float, agree: bool = True) -> dict:
    return {"monitor_pass_rate": pass_rate, "reward_monitor_disagreement": dis,
            "cross_view": {"agree": agree},
            "strongest_edges": [["near_fraction", "total_reward", near_w],
                                ["progress_score", "total_reward", prog_w]]}


def test_edge_in_finds_either_direction() -> None:
    from hymeko_rl.eval.cip.metaworld_cip import _edge_in
    edges = [["near_fraction", "total_reward", 0.9], ["total_reward", "progress_score", -0.4]]
    assert _edge_in(edges, "near_fraction", "total_reward") == (0.9, "near_fraction->total_reward")
    assert _edge_in(edges, "progress_score", "total_reward") == (-0.4, "total_reward->progress_score")  # reversed
    assert _edge_in(edges, "action_noise", "total_reward") == (None, None)


def test_aggregate_marks_recurrent_sign_consistent_edge_stable() -> None:
    from hymeko_rl.eval.cip.metaworld_cip import _aggregate_batches
    batches = [_fake_batch(0.95, 0.80, 0.50, 0.4), _fake_batch(0.97, 0.82, 0.42, 0.5),
               _fake_batch(0.96, 0.05, 0.53, 0.45)]        # progress edge weak in batch 3
    agg = _aggregate_batches(batches, min_presence=0.6)
    near = agg["edges"]["near_fraction--total_reward"]
    assert near["presence"] == 1.0 and near["sign_consistent"] and near["stable"]
    assert near["dominant_direction"] == "near_fraction->total_reward"
    assert agg["cross_view_all_pass"]
    assert "near_fraction--total_reward" in agg["stable_edges"]
    assert agg["monitor_pass_rate"]["n"] == 3


def test_aggregate_unstable_when_cross_view_fails_or_edge_absent() -> None:
    from hymeko_rl.eval.cip.metaworld_cip import _aggregate_batches
    # one batch missing the near edge and one cross-view failure
    batches = [_fake_batch(0.9, 0.8, 0.5, 0.4),
               {"monitor_pass_rate": 0.4, "reward_monitor_disagreement": 0.5, "cross_view": {"agree": False},
                "strongest_edges": [["progress_score", "total_reward", 0.7]]}]
    agg = _aggregate_batches(batches, min_presence=0.6)
    assert not agg["cross_view_all_pass"]
    assert agg["edges"]["near_fraction--total_reward"]["presence"] == 0.5   # present in 1 of 2


@pytest.mark.skipif(not _HAS_METAWORLD, reason="metaworld package not installed")
def test_render_coffee_push_gif(tmp_path) -> None:
    from hymeko_rl.eval.cip.metaworld_gifs import render_coffee_push_gif
    path, _success, frames = render_coffee_push_gif(0, 0.0, tmp_path / "g.gif", max_steps=16, stride=2, downsample=4)
    assert path.exists() and path.suffix == ".gif"
    assert len(frames) > 0 and frames[0].ndim == 3 and frames[0].shape[2] == 3


@pytest.mark.skipif(not _HAS_METAWORLD, reason="metaworld package not installed")
def test_multiseed_runs_and_aggregates(tmp_path) -> None:
    from hymeko_rl.eval.cip.metaworld_cip import run_metaworld_multiseed
    out = run_metaworld_multiseed("coffee_push", batches=2, n=16, seed0=0, out_dir=tmp_path)
    assert out["batches"] == 2 and len(out["per_batch"]) == 2
    assert (tmp_path / "coffee_push_multiseed_summary.json").exists()
    assert (tmp_path / "batch_0" / "causal_coffee_push_real.hymeko").exists()
    assert "near_fraction--total_reward" in out["aggregate"]["edges"]
    assert isinstance(out["aggregate"]["cross_view_all_pass"], bool)
