"""Realistic-motion contract: the governor caps the AIBO's exploit joint speeds."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv  # noqa: E402

from scenarios.aibo.capture_step import CapturePointWidening  # noqa: E402
from scenarios.aibo.motion_contract import JointVelocityGovernor  # noqa: E402


def _env():
    return QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12, max_steps=200)


def _peak_joint_speed(governed: bool, v_max: float = 8.0) -> float:
    import mujoco
    env = _env()
    w = CapturePointWidening()
    gov = JointVelocityGovernor(v_max=v_max)
    env.reset(seed=0)
    env.data.qvel[1] = 1.0
    mujoco.mj_forward(env.model, env.data)
    peak = 0.0
    for _ in range(200):
        a = w.action(env)
        if governed:
            a = gov.govern(env, a)
        env.step(a)
        peak = max(peak, gov.max_joint_speed(env))
    return peak


def test_governor_cuts_the_exploit_joint_speed() -> None:
    # ungoverned the capture-widening flings the legs ~27 rad/s (unphysical); governed it is far lower
    ungoverned = _peak_joint_speed(False)
    governed = _peak_joint_speed(True, v_max=8.0)
    assert ungoverned > 18.0                         # the exploit exists (real Aibo ~3-8 rad/s)
    assert governed < ungoverned * 0.6               # the contract substantially cuts it


def test_governor_preserves_braking_and_shape() -> None:
    env = _env()
    env.reset(seed=0)
    gov = JointVelocityGovernor(v_max=8.0)
    a = np.ones(env.model.nu, np.float32)            # at spawn all joints ~0 speed -> nothing cut
    out = gov.govern(env, a)
    assert out.shape == a.shape and np.allclose(out, a)
