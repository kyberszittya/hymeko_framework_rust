"""P1: the declarative-substrate wiring for the Aibo standing scenario.

``QuadrupedGoalEnv.from_hymeko`` builds the whole balance MDP from ``data/robotics/quadruped_stand.hymeko``
(scene tuning + ``reward_spec``), and ``TrainingSpec.from_hymeko`` reads the declared strategy from the *same*
file — so one ``.hymeko`` carries plant + tuning + objective + strategy. These tests are the round-trip: what
the ``.hymeko`` declares is exactly what the env/trainer receive, and the declarative default cannot silently
drift from the programmatic ``STAND_REWARD``.
"""
from __future__ import annotations

import pytest

from hymeko_rl.env.quadruped_env import (
    _DEFAULT_STAND_SCENARIO,
    STAND_REWARD,
    QuadrupedGoalEnv,
)
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.eval.tasks import get_task
from hymeko_rl.experiments.training_spec import TrainingSpec

_DECLARED = (("standing", 5.0), ("torso_height", 3.0), ("upright", 1.0),
             ("stand_still", 0.1), ("joint_velocity", 0.001))


def test_from_hymeko_builds_stand_env() -> None:
    """The declarative entry builds a stand-mode env over the 22-DOF Aibo: 12 RL-controlled legs, free base."""
    env = QuadrupedGoalEnv.from_hymeko()
    assert env.task == "stand" and env.base == "free"
    assert env.n_actions == 12
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.hg.n_vertices, 2)


def test_from_hymeko_reward_from_declaration() -> None:
    """The env's reward is exactly the declared ``reward_spec`` (term, weight) pairs — nothing hardcoded."""
    env = QuadrupedGoalEnv.from_hymeko()
    assert env.reward_spec.terms == _DECLARED


def test_declared_reward_equals_code_default() -> None:
    """Single-source consistency: the ``.hymeko`` reward_spec equals the code ``STAND_REWARD``, so the
    declarative and programmatic defaults cannot silently drift (fails if either changes without the other)."""
    assert RewardSpec.from_hymeko(_DEFAULT_STAND_SCENARIO).terms == STAND_REWARD.terms


def test_from_hymeko_scene_tuning() -> None:
    """Tuning scalars come from the ``scene`` bundle, not from env defaults."""
    env = QuadrupedGoalEnv.from_hymeko()
    assert env.max_steps == 250
    assert env.stand_cos == pytest.approx(0.9)
    assert env.stand_height_tol == pytest.approx(0.08)
    assert env.ctrl_range == pytest.approx(50.0)
    assert env.flip_cos == pytest.approx(-0.2)


def test_from_hymeko_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        QuadrupedGoalEnv.from_hymeko("data/robotics/does_not_exist_stand.hymeko")


def test_training_strategy_declared_in_same_file() -> None:
    """The training algorithm (BC warm-start -> TD3+BC) is declared in the SAME file and read by TrainingSpec —
    one ``.hymeko`` = plant + tuning + objective + strategy."""
    t = TrainingSpec.from_hymeko(_DEFAULT_STAND_SCENARIO)
    assert t.algorithm == "td3_bc"
    assert t.budget["total_steps"] == 150_000
    assert t.budget["seeds"] == (0, 1, 2)
    assert t.strategy["critic_huber"] == pytest.approx(1.0)
    assert t.strategy["bc_coef"] == pytest.approx(2.5)


def test_registered_task_uses_from_hymeko() -> None:
    """The ``quadruped_stand`` registry entry builds via ``from_hymeko`` — the registered task is single-source."""
    env = get_task("quadruped_stand").make_env()
    assert env.task == "stand" and env.n_actions == 12
    assert env.reward_spec.terms == STAND_REWARD.terms
