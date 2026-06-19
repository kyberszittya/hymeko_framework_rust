"""The declarative reward spec — terms + weights read from a ``.hymeko`` task profile
drive ``step()``'s reward (no hard-coded ``-dist``).

Covers the pure spec/reader (no MuJoCo) and an equivalence check that the spec-driven
reward reproduces the former ``-dist`` exactly.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hymeko_rl.env.reward import REACH_REWARD, RewardSpec, read_reward_terms

_REPO = Path(__file__).resolve().parents[2]
_TASK = _REPO / "data" / "robotics" / "arm_reach_task.hymeko"


def _env_stub(reach_thresh: float = 0.06) -> SimpleNamespace:
    """A minimal stand-in: the reward terms only read ``reach_thresh`` off the env."""
    return SimpleNamespace(reach_thresh=reach_thresh)


# ── spec (pure) ──────────────────────────────────────────────────────────────
def test_reach_reward_is_negative_distance() -> None:
    env = _env_stub()
    assert REACH_REWARD.terms == (("reach_distance", 1.0),)
    assert REACH_REWARD.evaluate(env, 0.5, np.zeros(4)) == pytest.approx(-0.5)
    assert REACH_REWARD.evaluate(env, 0.0, np.zeros(4)) == pytest.approx(0.0)


def test_weighted_terms_sum() -> None:
    spec = RewardSpec(
        (("reach_distance", 1.0), ("success_bonus", 2.0), ("action_cost", 0.1)))
    env = _env_stub(reach_thresh=0.1)
    # dist 0.05 < thresh → bonus fires; action ‖·‖²=2 → cost −0.2.
    # −0.05 + 2.0·1.0 + 0.1·(−2.0) = 1.75
    assert spec.evaluate(env, 0.05, np.array([1.0, 1.0])) == pytest.approx(1.75)
    # dist 0.5 ≥ thresh → no bonus; zero action → no cost. → −0.5
    assert spec.evaluate(env, 0.5, np.zeros(2)) == pytest.approx(-0.5)


def test_unknown_or_empty_rejected() -> None:
    with pytest.raises(ValueError, match="unknown reward term"):
        RewardSpec((("reach_distance", 1.0), ("nope", 1.0)))
    with pytest.raises(ValueError, match="at least one term"):
        RewardSpec(())


# ── reader (the .hymeko bridge) ──────────────────────────────────────────────
def test_read_reward_terms_from_task_profile() -> None:
    """The reader recovers the reaching profile's term + weight — so the .hymeko reward
    and the default Python spec agree."""
    assert read_reward_terms(_TASK) == (("reach_distance", 1.0),)
    assert RewardSpec.from_hymeko(_TASK).terms == (("reach_distance", 1.0),)


def test_reader_parses_weights_and_order(tmp_path: Path) -> None:
    prof = tmp_path / "r.hymeko"
    prof.write_text(
        "p {\n"
        "  @d: rew.reach_distance { weight 0.5; (+ f, - t); }\n"
        "  @b: rew.success_bonus { weight 3.0; }\n"
        "  @s: r.reward_spec { (+ d, + b); }\n"
        "}\n")
    assert read_reward_terms(prof) == (("reach_distance", 0.5), ("success_bonus", 3.0))


def test_reader_rejects_missing_reward_spec(tmp_path: Path) -> None:
    bad = tmp_path / "no_spec.hymeko"
    bad.write_text("p { @d: rew.reach_distance { weight 1.0; (+ f); } }")
    with pytest.raises(ValueError, match="reward_spec"):
        read_reward_terms(bad)


# ── equivalence: spec-driven reward == former -dist ──────────────────────────
def test_env_reward_matches_minus_dist() -> None:
    """``step()``'s reward reproduces the old ``-dist`` exactly (4-DOF arm_world)."""
    from hymeko_rl.env.arm_reach_env import ArmReachEnv

    env = ArmReachEnv(control_mode="torque")
    assert env.reward_spec is REACH_REWARD
    env.reset(seed=4)
    _, reward, _, _, info = env.step(np.zeros(env.n_actions, dtype=np.float32))
    assert reward == pytest.approx(-info["dist"])
