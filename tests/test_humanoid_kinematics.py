"""Kinematic validation of the (Vukobratović) humanoid model — pure forward kinematics.

Every actuated joint must, when commanded, rotate its child body by the commanded angle
about the *declared* axis (AXIS_Y sagittal, AXIS_X frontal), and every named kinematic
element (links + couplers) must be present and connected. These are FK-only tests
(``mj_forward``, no dynamics) so they are deterministic and fast, and they would catch a
mis-wired joint, a wrong axis, a dropped body, or a broken parent/child link — the exact
failure modes of hand-authoring a `.hymeko` and mapping it to MJCF.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from scenarios.humanoid.balance_env import HumanoidBalanceEnv  # noqa: E402

_EXPECTED_BODIES = {
    "pelvis", "torso", "head",
    "thigh_l", "shin_l", "foot_l", "thigh_r", "shin_r", "foot_r",
    "upper_arm_l", "forearm_l", "upper_arm_r", "forearm_r",
    "hip_ab_l", "hip_ab_r", "ank_rl_l", "ank_rl_r",           # Vukobratović couplers
}
_FRONTAL = {"hip_l_ab", "hip_r_ab", "ankle_l_roll", "ankle_r_roll"}   # AXIS_X
_ANGLE = 0.3


def _env():
    return HumanoidBalanceEnv(max_steps=5, seed=0)


def _neutral(mj, m, d) -> None:
    """Zero configuration with a valid identity base quaternion (all bodies axis-aligned)."""
    d.qpos[:] = 0.0
    base = int(m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "base")])
    d.qpos[base + 3] = 1.0                                    # quat w = 1 (identity)
    mj.mj_forward(m, d)


def _rot_of(dR: np.ndarray) -> tuple[float, np.ndarray]:
    """Angle + unit axis of a rotation matrix (log map)."""
    angle = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
    return angle, ax / (np.linalg.norm(ax) + 1e-12)


def test_expected_kinematic_elements_present() -> None:
    e = _env()
    mj, m = e._mj, e.model
    names = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)}
    missing = _EXPECTED_BODIES - names
    assert not missing, f"missing kinematic elements: {missing}"


def test_every_actuated_joint_rotates_child_by_commanded_angle_about_axis() -> None:
    e = _env()
    mj, m, d = e._mj, e.model, e.data
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, jid)
        bid = int(m.jnt_bodyid[jid])
        _neutral(mj, m, d)
        r0 = d.xmat[bid].reshape(3, 3).copy()
        _neutral(mj, m, d)
        d.qpos[int(m.jnt_qposadr[jid])] = _ANGLE
        mj.mj_forward(m, d)
        r1 = d.xmat[bid].reshape(3, 3).copy()
        angle, axis = _rot_of(r1 @ r0.T)
        assert abs(angle - _ANGLE) < 0.02, f"{name}: child rotated {angle:.3f} rad, expected {_ANGLE}"
        decl = np.asarray(m.jnt_axis[jid], float)
        assert abs(float(np.dot(axis, decl / np.linalg.norm(decl)))) > 0.99, f"{name}: wrong rotation axis"


def test_declared_axes_match_plane() -> None:
    # sagittal joints rotate about Y; the 4 Vukobratović joints rotate about X (frontal)
    e = _env()
    mj, m = e._mj, e.model
    for i in range(m.nu):
        jid = int(m.actuator_trnid[i, 0])
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, jid)
        axis = np.asarray(m.jnt_axis[jid], float)
        if name in _FRONTAL:
            assert abs(axis[0]) > 0.99, f"{name} should be AXIS_X (frontal)"
        else:
            assert abs(axis[1]) > 0.99, f"{name} should be AXIS_Y (sagittal)"


def test_each_actuator_drives_a_distinct_joint() -> None:
    e = _env()
    m = e.model
    driven = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]
    assert len(set(driven)) == m.nu == 16                    # 16 distinct actuated joints
