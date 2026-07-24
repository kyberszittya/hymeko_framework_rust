"""The SHARED V3 low-level execution stack — IDENTICAL in calibration and in the coin controller (the user's contract:
the same armature/damping/governor/torque-rate/PD/control_dt/substeps must run in both, or a new contract mismatch
appears). The governor runs PER SUB-STEP via MuJoCo's ``mjcb_control`` (as the real coin does), NOT once per control
step — otherwise the arm accelerates unbounded within a control step and the calibration under-governs.

Layers: PD position command (control rate) → torque-rate limit → directional velocity governor (sub-step rate) → actuator
saturation → mj_step. armature/damping/friction live on the model. No direct qvel state-clamp.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hymeko_rl.env.motion_contract import TorqueGovernorConfig, govern_torque


@dataclass(frozen=True)
class V3Stack:
    """The versioned low-level stack config. Frozen together so calibration == execution."""

    qdot_soft: float
    qdot_hard: float
    armature: float
    damping: float
    friction: float
    kp: float
    kv: float
    tau_rate: float | None
    control_dt: float = 0.01
    substeps: int = 20

    @property
    def gov(self) -> TorqueGovernorConfig:
        return TorqueGovernorConfig(self.qdot_soft, self.qdot_hard)


class GovernedArm:
    """Drives the first ``n`` dofs of a MuJoCo model with the V3 stack. Set the raw control-step torque (PD or direct);
    the governor fires every sub-step via ``mjcb_control`` on the CURRENT velocity, so velocity is capped within the
    control step. Use as a context manager so the global callback is always reset."""

    def __init__(self, model, data, stack: V3Stack, n: int = 4):
        self.m, self.d, self.s, self.n = model, data, stack, n
        model.dof_armature[:n] = stack.armature
        model.dof_damping[:n] = stack.damping
        model.dof_frictionloss[:n] = stack.friction
        self._raw = np.zeros(n)
        self._prev = None

    def __enter__(self):
        gov, n = self.s.gov, self.n
        raw = self._raw

        def cb(_model, data):
            data.ctrl[:n] = govern_torque(raw, data.qvel[:n], gov)     # per sub-step, on the live velocity
        mujoco.set_mjcb_control(cb)
        return self

    def __exit__(self, *exc):
        mujoco.set_mjcb_control(None)
        return False

    def _commit(self, tau):
        n = self.n
        if self.s.tau_rate is not None and self._prev is not None:
            tau = self._prev + np.clip(tau - self._prev, -self.s.tau_rate * self.s.control_dt, self.s.tau_rate * self.s.control_dt)
        tau = np.clip(tau, self.m.actuator_ctrlrange[:n, 0], self.m.actuator_ctrlrange[:n, 1])
        self._raw[:] = tau                                            # the governor reads this every sub-step
        self._prev = tau.copy()
        for _ in range(self.s.substeps):
            mujoco.mj_step(self.m, self.d)
        return tau

    def pd_step(self, q_des):
        """One control step under PD position control (governed per sub-step)."""
        q, qd = self.d.qpos[:self.n], self.d.qvel[:self.n]
        return self._commit(self.s.kp * (np.asarray(q_des, np.float64) - q) - self.s.kv * qd)

    def torque_step(self, tau_cmd):
        """One control step under a direct torque command (governed per sub-step)."""
        return self._commit(np.asarray(tau_cmd, np.float64))
