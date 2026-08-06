"""The HyMeKo humanoid robot model (``data/robotics/humanoid.hymeko``) — a compile/topology guard.

The RL humanoid (pelvis+torso+head + 2 legs + 2 arms, 13 sagittal revolute joints + a floating ``@base``)
must emit to a valid MuJoCo model with the declared actuator count before any ``HumanoidEnv`` is wired
(plan: docs/plans/2026-07-04-quadruped-humanoid-locomotion/, S3 precondition). Distinct from the box-approx
scaling fixture under ``scripts/scaling/fixtures/humanoid/`` — this is the RL model.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from hymeko_rl.env.arm_world import emit_arm_mjcf

_HUMANOID = Path(__file__).resolve().parents[2] / "data" / "robotics" / "humanoid.hymeko"


def _model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(emit_arm_mjcf(str(_HUMANOID), name="humanoid"))


def test_humanoid_emits_and_compiles_with_declared_topology() -> None:
    m = _model()
    # 13 links (pelvis, torso, head, 2×{thigh,shin,foot}, 2×{upper_arm,forearm}) + the world body.
    assert m.nbody == 14, f"expected 14 bodies, got {m.nbody}"
    assert m.njnt == 13, f"expected 13 joints, got {m.njnt}"
    assert m.nu == 13, f"expected 13 actuators (one per revolute joint), got {m.nu}"


def test_humanoid_has_the_expected_named_joints() -> None:
    m = _model()
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)}
    expected = {"abdomen", "neck", "hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r",
                "shoulder_l", "elbow_l", "shoulder_r", "elbow_r"}
    assert expected <= names, f"missing joints: {expected - names}"


def test_humanoid_stands_above_the_floor_at_rest() -> None:
    """A sanity guard on the geometry: at the zero pose the pelvis is near its declared 0.95 m base height,
    i.e. the robot is authored standing, not intersecting the ground (a mis-signed origin would fail this)."""
    m = _model()
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    zs = data.xpos[1:, 2]   # skip the world body
    assert zs.max() > 0.5, "no body is above 0.5 m — humanoid is not authored upright"
    assert zs.min() > -0.05, "a body sits below the floor plane at rest"


def test_humanoid_model_is_finite() -> None:
    m = _model()
    data = mujoco.MjData(m)
    mujoco.mj_forward(m, data)
    import numpy as np
    assert np.all(np.isfinite(data.xpos)), "non-finite body positions in the rest pose"
