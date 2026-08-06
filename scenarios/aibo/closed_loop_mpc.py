"""Closed-loop (receding-horizon) capturability MPC for running — rejects disturbances.

The open-loop running plan is a fixed periodic force profile; a **push mid-run** would drift it. A
closed-loop MPC re-solves the stance force *from the actual measured state* each stride, driving the
centroid back onto the nominal running orbit (target forward speed + periodic bounce) while keeping
the capturability Lyapunov bounded. This demonstrates the receding-horizon feedback: inject a
velocity push, and the controller replans the next stance to recover, staying capturable — where the
open-loop plan would diverge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hop_mpc import HopParams
from .running_mpc import RunningGaitMPC


@dataclass
class ClosedLoopRunningMPC:
    """Receding-horizon MPC: each stance, re-solve the force to return to the nominal running orbit."""

    p: HopParams = field(default_factory=HopParams)
    dt: float = 0.02
    n_stance: int = 12
    n_flight: int = 14
    v_forward: float = 0.6
    _target: tuple = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        nom = RunningGaitMPC(p=self.p, dt=self.dt, n_stance=self.n_stance,
                             n_flight=self.n_flight, v_forward=self.v_forward)
        _f, z_td, vz_td = nom.plan_stride()
        self._target = (z_td, vz_td)                          # nominal periodic touchdown [z, vz]

    def _roll(self, state: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Integrate stance (forces) + ballistic flight from ``state``."""
        m, g = self.p.mass, self.p.g
        x = state.copy()
        prof = np.vstack([forces.reshape(self.n_stance, 2), np.zeros((self.n_flight, 2))])
        traj = [x.copy()]
        for fx, fz in prof:
            x = x + self.dt * np.array([x[1], fx / m, x[3], fz / m - g])
            traj.append(x.copy())
        return np.array(traj)

    def plan_stance_from(self, state: np.ndarray) -> np.ndarray:
        """Re-solve the stance force to reach the nominal orbit's touchdown from the CURRENT state."""
        from scipy.optimize import minimize
        z_td, vz_td = self._target

        def obj(z):
            end = self._roll(state, z)[-1]
            track = (end[1] - self.v_forward) ** 2 + (end[2] - z_td) ** 2 + (end[3] - vz_td) ** 2
            return 40.0 * track + self.dt * float(np.sum(z ** 2)) * 1e-3

        cons = [{"type": "ineq", "fun": lambda z: self._roll(state, z)[:, 2].min() - 0.02}]
        for j in range(self.n_stance):
            cons.append({"type": "ineq", "fun": (lambda z, j=j: self.p.mu * z[2 * j + 1] - abs(z[2 * j]))})
        bnds = [(-self.p.f_max, self.p.f_max), (0.0, self.p.f_max)] * self.n_stance
        z0 = np.tile([0.0, self.p.mass * self.p.g], self.n_stance).astype(float)
        return minimize(obj, z0, method="SLSQP", bounds=bnds, constraints=cons,
                        options={"maxiter": 120, "ftol": 1e-6}).x.reshape(self.n_stance, 2)

    def simulate(self, n_strides: int = 6, push_stride: int = 3, push_dvx: float = 0.4):
        """Closed loop: replan each stance from the measured state; inject a forward velocity push once."""
        m, g = self.p.mass, self.p.g
        z_td, vz_td = self._target
        x = np.array([0.0, self.v_forward, z_td, vz_td])
        traj, sched, vx_err = [x.copy()], [], []
        for s in range(n_strides):
            if s == push_stride:
                x[1] += push_dvx                              # disturbance: a forward velocity kick
            forces = self.plan_stance_from(x)                 # RE-SOLVE from the current (disturbed) state
            prof = np.vstack([forces, np.zeros((self.n_flight, 2))])
            for i, (fx, fz) in enumerate(prof):
                x = x + self.dt * np.array([x[1], fx / m, x[3], fz / m - g])
                traj.append(x.copy())
                sched.append(i < self.n_stance)
                vx_err.append(abs(x[1] - self.v_forward))
        return np.array(traj), np.array(sched), np.array(vx_err)

    def capture_lyapunov(self, traj: np.ndarray) -> np.ndarray:
        px, vx, pz = traj[:, 0], traj[:, 1], np.maximum(traj[:, 2], 0.02)
        xi = px + vx * np.sqrt(pz / self.p.g)
        stride = self.v_forward * (self.n_stance + self.n_flight) * self.dt
        foot = np.round(px / max(stride, 1e-6)) * stride
        return 0.5 * (xi - foot) ** 2
