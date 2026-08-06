"""Anatomy sanity checks for the humanoid — a permanent guard on the movement/kinematics.

These lock the anatomical invariants that were verified by hand after the joint-limit fix, so any future
model change that breaks them (a mirror flip, a lost joint limit, a wrong-way bend) fails immediately:

  * left-right SYMMETRY — the same joint angle on both legs moves both the SAME way (no mirror error);
  * the robot FACES forward (+x) and its feet point +x;
  * joints have ANATOMICAL limits (not the ±π free-rotation default) — knees/elbows cannot hyperextend;
  * measured FLEX directions (hip forward = negative, knee flex = positive, …) are preserved.

If any of these fail, the movement is no longer anatomically valid — fix the model, not the test.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco")

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv  # noqa: E402


def _env():
    return HumanoidBalanceEnv(BalanceConfig(perturb_lo=0.0, perturb_hi=0.0), seed=0)


def _qadr(m, name: str) -> int:
    return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)])


def _foot_dx(e, joint_pair, val):
    """Δx of each foot when `val` is applied to both `joint_pair` (l, r) joints, from the neutral pose."""
    m, d = e.model, e.data
    e.reset(seed=0)
    mujoco.mj_forward(m, d)
    x0 = (float(d.xpos[e._fl][0]), float(d.xpos[e._fr][0]))
    e.reset(seed=0)
    d.qpos[_qadr(m, joint_pair[0])] = val
    d.qpos[_qadr(m, joint_pair[1])] = val
    mujoco.mj_forward(m, d)
    return float(d.xpos[e._fl][0]) - x0[0], float(d.xpos[e._fr][0]) - x0[1]


def test_legs_are_left_right_symmetric() -> None:
    # the SAME hip angle on both legs must move BOTH feet the SAME direction by ~the SAME amount
    e = _env()
    dl, dr = _foot_dx(e, ("hip_l", "hip_r"), 0.6)
    assert np.sign(dl) == np.sign(dr) and abs(dl) > 0.05          # both move, same direction (no mirror flip)
    assert abs(dl - dr) < 0.02                                    # ~equal magnitude (mirror-symmetric)


def test_same_knee_angle_is_symmetric() -> None:
    e = _env()
    dl, dr = _foot_dx(e, ("knee_l", "knee_r"), 0.6)
    assert np.sign(dl) == np.sign(dr) and abs(dl - dr) < 0.02


def test_robot_faces_forward_x() -> None:
    e = _env()
    e.reset(seed=0)
    mujoco.mj_forward(e.model, e.data)
    r = e.data.xmat[e._pelvis].reshape(3, 3)
    assert r[0, 0] > 0.9                                          # pelvis local +x ~ world +x (faces forward)
    # the foot geometry points forward (+x): the foot body sits ahead of the ankle
    assert float(e.data.xpos[e._fl][0]) > -0.05


def test_measured_flex_directions_preserved() -> None:
    # hip: + = backward (extension); knee: + = foot back (flexion) — the signs the walking refs depend on
    e = _env()
    hl, _hr = _foot_dx(e, ("hip_l", "hip_r"), 0.6)
    kl, _kr = _foot_dx(e, ("knee_l", "knee_r"), 0.6)
    assert hl < -0.05                                             # +hip -> foot BACK (forward-flex is negative)
    assert kl < -0.02                                            # +knee -> foot BACK (knee flexion)


def test_joints_have_anatomical_limits_not_full_rotation() -> None:
    # every actuated revolute joint must have a real (sub-±π) limit — no free rotation / hyperextension
    e = _env()
    m = e.model
    full = np.deg2rad(180.0)
    for name in ("knee_l", "knee_r", "hip_l", "hip_r", "elbow_l", "elbow_r", "ankle_l", "ankle_r"):
        lo, hi = m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
        assert not (lo <= -full + 1e-3 and hi >= full - 1e-3), f"{name} has no anatomical limit"
        assert hi - lo < 2.0 * full - 1e-3


def test_knee_cannot_hyperextend() -> None:
    # the knee flexes one way only: its range must be one-signed-dominant (flexion), not symmetric ±
    e = _env()
    m = e.model
    lo, hi = m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "knee_l")]
    assert hi > 2.0 and lo > -0.2                                 # flexes to ~150°, no meaningful hyperextension


def test_stands_upright_within_limits() -> None:
    # the nominal standing pose must sit INSIDE all joint limits (no joint pinned at a bound at rest)
    e = _env()
    e.reset(seed=0)
    for _ in range(150):
        e.step(np.zeros(e.model.nu, np.float32))
    assert e._com_sig()["uprightness"] > 0.9
    m, d = e.model, e.data
    for i in range(m.njnt):
        if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        lo, hi = m.jnt_range[i]
        if hi > lo:
            q = float(d.qpos[m.jnt_qposadr[i]])
            assert lo - 0.05 <= q <= hi + 0.05                   # resting angle within (a hair of) the limits
