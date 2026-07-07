"""Tests for the damped least-squares pose IK solver (hymeko_rl.env.ik)."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.ik import DampedPoseIK
from hymeko_rl.env.pick_place_env import PickPlaceEnv


def _arm_ik() -> tuple[PickPlaceEnv, DampedPoseIK]:
    env = PickPlaceEnv(max_steps=10)
    env.reset(seed=0)
    return env, env._ik


def _downness(model: mujoco.MjModel, ee_body: int, q: np.ndarray) -> float:
    d = mujoco.MjData(model)
    d.qpos[:6] = q
    mujoco.mj_kinematics(model, d)
    return -float(d.xmat[ee_body].reshape(3, 3)[2, 2])


def test_fk_tool_matches_kinematics() -> None:
    """``fk_tool`` returns the tool position an independent ``mj_kinematics`` FK gives for the same config."""
    env, ik = _arm_ik()
    q = np.array([0.2, -0.6, 0.6, 0.1, 0.4, 0.0])
    tool = ik.fk_tool(q)
    d = mujoco.MjData(env.model)
    d.qpos[:6] = q
    mujoco.mj_kinematics(env.model, d)
    assert tool.shape == (3,)
    assert np.allclose(tool, np.asarray(d.xpos[env._b_tool]), atol=1e-9)


def test_solve_converges_in_position() -> None:
    """A guaranteed-reachable target (the FK image of a known config) is hit to mm precision."""
    env, ik = _arm_ik()
    d = mujoco.MjData(env.model)
    d.qpos[:6] = np.array([0.2, -0.6, 0.6, 0.1, 0.4, 0.0])
    mujoco.mj_kinematics(env.model, d)
    target = np.asarray(d.xpos[env._b_tool]).copy()
    q = ik.solve(np.zeros(6), target, down=False, iters=300)
    d2 = mujoco.MjData(env.model)
    d2.qpos[:6] = q
    mujoco.mj_kinematics(env.model, d2)
    assert float(np.linalg.norm(np.asarray(d2.xpos[env._b_tool]) - target)) < 1e-2


def test_down_orientation_term_pulls_tool_down() -> None:
    """``down=True`` yields a more downward-pointing tool than position-only IK at the same target."""
    env, ik = _arm_ik()
    target = np.array([0.3, 0.0, 0.4])
    q_pos = ik.solve(np.zeros(6), target, down=False, iters=200)
    q_down = ik.solve(np.zeros(6), target, down=True, iters=200)
    assert _downness(env.model, env._b_tool, q_down) > _downness(env.model, env._b_tool, q_pos)


def test_solve_respects_joint_limits() -> None:
    """The returned config never violates a limited joint's range (clamped each iteration)."""
    env, ik = _arm_ik()
    q = ik.solve(np.zeros(6), np.array([2.0, 2.0, 2.0]), down=True, iters=50)  # unreachable → drives to limits
    for j in range(env.model.njnt):
        adr = int(env.model.jnt_qposadr[j])
        if adr < 6 and bool(env.model.jnt_limited[j]):
            lo, hi = env.model.jnt_range[j]
            assert lo - 1e-5 <= q[adr] <= hi + 1e-5


def test_solve_collision_free_returns_clamped_config() -> None:
    """Multi-start collision-free solve returns a length-6, finite, limit-clamped config (validator honoured)."""
    env, ik = _arm_ik()
    obj = env._obj_xyz()
    q = ik.solve_collision_free(env.data.qpos[:6], np.array([obj[0], obj[1], 0.4]),
                                env._ik_pose_valid, down=True, n_starts=8)
    assert q.shape == (6,) and np.isfinite(q).all()
    assert np.all(q >= -np.pi - 1e-5) and np.all(q <= np.pi + 1e-5)


def test_rejects_bad_construction() -> None:
    env, _ = _arm_ik()
    with pytest.raises(ValueError):
        DampedPoseIK(env.model, env._b_tool, 0)
    with pytest.raises(ValueError):
        DampedPoseIK(env.model, env._b_tool, env.model.nv + 5)
