"""6D-0 SE(3) pose reach — env-layer tests (position + orientation, reachable-by-construction targets, pose certificate).

Covers the SE(3) additions to the reach env: the 18-dim POSE observation, the quaternion-geodesic orientation error, the
conjunctive pose certificate (position AND orientation), FK-reachability of the sampled target, and the 6-DoF expert
ceiling on the basic (closable) difficulty. The base position-reach env is unchanged (its suite still passes).
"""
import numpy as np
import mujoco
import pytest

from hymeko_rl.env.observation import POSE_OBSERVATION
from hymeko_rl.env.se3_reach_env import SE3ReachEnv


def _env(**kw):
    d = dict(control_mode="position", max_steps=200, reach_thresh=0.06, ang_thresh=0.35,
             start_perturb=0.2, expert_gain=0.5)
    d.update(kw)
    return SE3ReachEnv(**d)


def test_pose_observation_is_18_dim():
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape[1] == POSE_OBSERVATION.obs_dim == 18       # qpos+qvel+target3+eeErr3+quat4+orientErr3+angVel3
    assert "target" in info and "target_quat" in info
    assert env.target_quat().shape == (4,)


def test_orientation_error_geodesic():
    env = _env()
    env.reset(seed=0)
    q = env._ee_quat()
    env._target_quat = q.copy()
    assert env.ang_err() < 1e-5                                  # same orientation → 0
    env._target_quat = (-q).astype(np.float32)
    assert env.ang_err() < 1e-5                                  # double-cover: −q is the same rotation → 0
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, np.array([0.0, 0.0, 1.0]), np.pi)
    t = np.zeros(4)
    mujoco.mju_mulQuat(t, q.astype(np.float64), dq)
    env._target_quat = t.astype(np.float32)
    assert abs(env.ang_err() - np.pi) < 1e-3                     # 180° → π


def test_pose_certificate_needs_both_position_and_orientation():
    env = _env()
    env.reset(seed=1)
    # position satisfied but orientation wrong ⇒ NOT reached
    env._target = env._ee_pos().copy()
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, np.array([1.0, 0.0, 0.0]), 1.2)   # 1.2 rad > ang_thresh
    t = np.zeros(4)
    mujoco.mju_mulQuat(t, env._ee_quat().astype(np.float64), dq)
    env._target_quat = t.astype(np.float32)
    assert not env._reached(dist=0.0)                            # pos ok, ang bad → gate closed
    # both satisfied ⇒ reached
    env._target_quat = env._ee_quat().copy()
    assert env._reached(dist=0.0)


def test_sampled_target_is_fk_reachable():
    env = _env()
    env.reset(seed=2)
    p, quat = env._fk_pose(env._target_q)                       # FK of the stored config reproduces the target pose
    assert np.linalg.norm(p - env._target) < 1e-4
    r = np.zeros(3)
    mujoco.mju_subQuat(r, env._target_quat.astype(np.float64), quat.astype(np.float64))
    assert np.linalg.norm(r) < 1e-4


def test_start_is_bounded_perturbation_of_target_config():
    env = _env(start_perturb=0.2)
    env.reset(seed=3)
    # the start qpos is within start_perturb of the target config (clipped to joint range)
    n = env.n_actions
    assert np.all(np.abs(env.data.qpos[:n] - env._target_q[:n]) <= 0.2 + 1e-6)


def test_ang_thresh_must_be_positive():
    with pytest.raises(ValueError):
        _env(ang_thresh=0.0)


def test_expert_reaches_pose_certificate_on_basic_difficulty():
    env = _env(start_perturb=0.2)
    n = 0
    for s in range(16):
        env.reset(seed=s)
        for _ in range(200):
            _o, _r, term, trunc, info = env.step(env.expert_action)
            if term and not info["death"]:
                n += 1
                break
            if term or trunc:
                break
    assert n >= 12                                              # the 6-DoF DLS-IK expert reaches most basic poses (≈19/20)


def test_extra_step_info_reports_ang_err():
    env = _env()
    env.reset(seed=0)
    _o, _r, _t, _tr, info = env.step(env.expert_action)
    assert "ang_err" in info and info["ang_err"] >= 0.0
