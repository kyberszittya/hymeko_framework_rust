"""Turn-authority diagnostic — the AIBO yaw/stability wall behind the off-axis reach failure.

Certifies the measured wall: `AgileTurnGait` emits valid actions; a STABLE (upright) turn is slow
(≲ a few tens of deg/1000 steps — too slow to face a ±40° goal in a bounded horizon); and a strong
(near-full-spin) turn TIPS the robot over (min uprightness < 0.5). This is the reproducible evidence
that off-axis reaching is a kinematic/dynamic limit, not a controller/learning deficit.
"""

from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
from scenarios.aibo.motion_contract import JointVelocityGovernor
from scenarios.aibo.turn_authority import (
    AgileTurnGait,
    measure_turn,
    stable_turn_ceiling,
)


@pytest.fixture(scope="module")
def env() -> QuadrupedGoalEnv:
    return QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                            max_steps=600)


@pytest.fixture(scope="module")
def gait() -> AgileTurnGait:
    return AgileTurnGait()


@pytest.fixture(scope="module")
def gov() -> JointVelocityGovernor:
    return JointVelocityGovernor(v_max=8.0)


def test_agile_turn_action_is_valid(env, gait) -> None:
    env.reset(seed=0)
    a = gait.action(env, turn=0.5, drive=1.0)
    assert a.shape == (int(env.action_space.shape[0]),)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_stable_turn_is_slow(env, gait, gov) -> None:
    # a gentle, upright turn barely rotates — the stable yaw authority is small (the wall).
    m = measure_turn(env, gait, gov, turn=0.3, drive=1.0, steps=500)
    assert not m.tipped
    assert abs(m.yaw_deg_per_1000) < 60.0        # far below what a 40° face-in needs in one episode


def test_strong_turn_tips_the_robot(env, gait, gov) -> None:
    # the strongest skid-steer turn (near full reverse-inner spin) is dynamically unstable — it tips.
    m = measure_turn(env, gait, gov, turn=1.0, drive=1.0, steps=500)
    assert m.tipped                               # min uprightness dropped below 0.5 (flipped)


def test_turn_stability_wall(env, gait, gov) -> None:
    # the diagnostic finds a small stable rate AND a tipping onset: turning fast means falling over.
    res = stable_turn_ceiling(env, gait, gov, drive=1.0, turns=(0.3, 0.5, 0.9, 1.0), steps=500)
    assert res["fastest_stable_deg_per_1000"] < 60.0
    assert res["tipping_onset_turn"] is not None  # some turn magnitude tips within the sweep
