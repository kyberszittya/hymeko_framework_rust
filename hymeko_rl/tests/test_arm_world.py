"""The arm-loads-and-steps smoke (the MuJoCo stack works post-install).

Run: pytest -p no:randomly hymeko_rl/tests/test_arm_world.py
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from hymeko_rl.env.arm_world import CONTROL_MODES, N_JOINTS, load_arm, make_arm_mjcf


@pytest.mark.parametrize("mode", list(CONTROL_MODES))
def test_make_arm_mjcf_builds_each_control_mode(mode: str) -> None:
    """torque / position / velocity all compile to a 4-DOF arm (same kinematics, different
    actuators)."""
    m = mujoco.MjModel.from_xml_string(make_arm_mjcf(mode))
    assert m.nu == N_JOINTS and m.nq == N_JOINTS


def test_make_arm_mjcf_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        make_arm_mjcf("impedance")


def test_arm_loads_with_four_joints_and_actuators() -> None:
    model, data = load_arm()
    assert model.nq == N_JOINTS and model.nu == N_JOINTS
    assert np.all(np.isfinite(data.qpos))


def test_arm_steps_and_responds_to_a_setpoint() -> None:
    import mujoco
    model, data = load_arm()
    data.ctrl[:] = np.array([0.5, 0.4, -0.3, 0.6])   # a non-zero joint setpoint
    for _ in range(500):                              # ~1 s at the 2 ms timestep
        mujoco.mj_step(model, data)
    q = data.qpos.copy()
    assert np.all(np.isfinite(q))
    # the position actuators must drive the joints off the zero configuration.
    assert np.linalg.norm(q) > 0.1
