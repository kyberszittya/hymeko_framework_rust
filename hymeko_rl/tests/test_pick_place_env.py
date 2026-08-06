"""Tests for the Phase-0 pick-and-place env (HyMeKo 6-DOF arm + gripper + liftable box + contact)."""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.pick_place_env import PickPlaceEnv


def test_builds_from_hymeko_arm_and_shapes() -> None:
    """The SAME anthropomorphic_arm.hymeko is emitted + a gripper attached; the hypergraph spans both."""
    env = PickPlaceEnv(max_steps=20)
    # base_link, link_0..4, tool, finger_l, finger_r
    assert env.hg.n_vertices == 9
    assert len(env.hg.edges) >= 16
    obs, info = env.reset(seed=0)
    assert obs.shape == (9, 10) and np.isfinite(obs).all()
    assert env.action_space.shape == (7,)            # 6 arm joints + grip
    assert "obj_to_target" in info


def test_step_contract() -> None:
    env = PickPlaceEnv(max_steps=10)
    env.reset(seed=1)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (9, 10) and np.isfinite(r)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert {"lifted", "both_contact", "reached", "death"} <= set(info)


def test_env_is_graspable_contact_at_closed_pose() -> None:
    """Phase-0 acceptance: the gripper is graspable — with the box between the closed fingers, both contact it.
    Tests the grasp geometry + contact detection; the arm posing the gripper at the box is Phase-1 control."""
    env = PickPlaceEnv(max_steps=50, box_half=0.02)
    env.reset(seed=2)

    def qadr(jn: str) -> int:
        return int(env.model.jnt_qposadr[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jn)])

    env.data.qpos[qadr("grip_l")] = 0.012        # close fingers toward centre
    env.data.qpos[qadr("grip_r")] = -0.012
    mujoco.mj_forward(env.model, env.data)
    fl = np.asarray(env.data.xpos[env._b_fl])
    fr = np.asarray(env.data.xpos[env._b_fr])
    mid = (fl + fr) / 2.0                          # place the box between the (closed) fingers
    q = env._obj_qadr
    env.data.qpos[q:q + 3] = mid
    env.data.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    left, right = env._both_contact()
    assert left and right, f"box between closed fingers did not contact both (L={left}, R={right})"


def test_rejects_bad_args() -> None:
    with pytest.raises(ValueError):
        PickPlaceEnv(frame_skip=0)
