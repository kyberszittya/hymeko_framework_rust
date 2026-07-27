"""AIBO capture-point protective step — the step certifies a lateral push the stand can't."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv  # noqa: E402

from scenarios.aibo.capture_step import (  # noqa: E402
    CapturePointWidening,
    PushRecoveryLyapunov,
    capture_point_y,
    recover_v_series,
)
from scenarios.aibo.lyapunov import evaluate_lyapunov  # noqa: E402


def _env():
    return QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12, max_steps=300)


def test_capture_point_moves_with_lateral_velocity() -> None:
    import mujoco
    env = _env()
    env.reset(seed=0)
    xi0 = capture_point_y(env)
    env.data.qvel[1] = 1.0
    mujoco.mj_forward(env.model, env.data)
    assert capture_point_y(env) > xi0 + 0.05          # +lateral velocity pushes the capture point +y


def test_push_recovery_lyapunov_zero_at_equilibrium() -> None:
    env = _env()
    env.reset(seed=0)
    V = PushRecoveryLyapunov()
    assert V(env) < 0.15                              # spawn is near upright, centred, at rest


def test_capture_widening_certifies_a_push_the_stand_cannot() -> None:
    env = _env()
    V = PushRecoveryLyapunov()
    stepper = CapturePointWidening()
    stand = evaluate_lyapunov(recover_v_series(env, None, 1.0, steps=300, V=V))
    step = evaluate_lyapunov(recover_v_series(env, stepper, 1.0, steps=300, V=V))
    assert not stand["passes"]                        # passive stand does NOT recover from a 1.0 m/s push
    assert step["passes"]                             # the capture-point step DOES (V -> 0)
    assert step["Vfinal"] < stand["Vfinal"]


def test_stepper_action_shape_and_bounds() -> None:
    env = _env()
    env.reset(seed=0)
    a = CapturePointWidening().action(env)
    assert a.shape == (env.model.nu,) and np.all(np.abs(a) <= 1.0 + 1e-6)
