"""Locks the bidirectional-yaw fix: the reduce-inner steered trot must turn BOTH
directions while staying upright (the old amplify-outer primitive tipped one way).
SIMULATION. Requires mujoco + hymeko CLI.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
from scenarios.aibo.locomotion_gait import SteeredTrotGait, body_yaw


def _net_yaw_and_upright(yaw_cmd: float, steps: int = 600) -> tuple[float, float]:
    env = QuadrupedGoalEnv(base="free", task="goal", max_steps=steps + 5)
    env.reset(seed=0)
    gait = SteeredTrotGait()
    y0 = body_yaw(env)
    up = 1.0
    for _ in range(steps):
        env.step(gait.action(env, yaw_cmd=yaw_cmd, drive=1.0))
        up = min(up, env._torso_uprightness())
    return float(np.degrees(body_yaw(env) - y0)), up


def test_left_and_right_turns_are_both_stable() -> None:
    left, up_l = _net_yaw_and_upright(+0.6)
    right, up_r = _net_yaw_and_upright(-0.6)
    # correct signs: +yaw -> turn left (CCW, +deg); -yaw -> turn right (-deg)
    assert left > 3.0, f"left turn too weak: {left:.1f} deg"
    assert right < -3.0, f"right turn too weak: {right:.1f} deg"
    # both stay upright (the old primitive flipped one direction: uprightness < 0)
    assert up_l > 0.5 and up_r > 0.5, f"tipped: up_l={up_l:.2f} up_r={up_r:.2f}"


def test_zero_yaw_bounded_drift() -> None:
    # the open-loop trot drifts ~7 deg / 600 steps (a known scripted-gait
    # imperfection the closed-loop pursuit corrects); assert it is bounded + upright,
    # not a runaway spin.
    net, up = _net_yaw_and_upright(0.0)
    assert abs(net) < 12.0 and up > 0.9
