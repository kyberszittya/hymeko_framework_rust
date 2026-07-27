"""Running as a periodic hopping gait — a centroidal limit cycle with a ballistic flight per stride.

Running = a cyclic [stance push → ballistic FLIGHT → stance …] that advances forward at steady
speed. This plans ONE periodic stride (single-shooting trajectory optimisation with a periodicity
constraint on [vx, z, vz]) and repeats it to simulate a continuous run — the vertical state bounces
periodically, the horizontal advances, and the capturability Lyapunov stays bounded every stride
(the orbital stability of running). Same centroidal model / motion-contract-realistic forces as the
hop MPC; embodiment-agnostic (AIBO vs human differ only by mass / height / force limits).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hop_mpc import HopParams


@dataclass
class RunningGaitMPC:
    """Plan a periodic running stride (stance+flight) at a target forward speed; simulate N strides."""

    p: HopParams = field(default_factory=HopParams)
    dt: float = 0.02
    n_stance: int = 12
    n_flight: int = 14               # ballistic flight per stride (F = 0)
    v_forward: float = 0.6           # target steady forward speed (m/s)

    def _stride_states(self, forces: np.ndarray, z_td: float, vz_td: float):
        """Integrate one stride (stance forces then ballistic flight) from touchdown."""
        m, g = self.p.mass, self.p.g
        x = np.array([0.0, self.v_forward, z_td, vz_td])
        traj = [x.copy()]
        prof = np.vstack([forces.reshape(self.n_stance, 2), np.zeros((self.n_flight, 2))])
        for fx, fz in prof:
            x = x + self.dt * np.array([x[1], fx / m, x[3], fz / m - g])
            traj.append(x.copy())
        return np.array(traj)

    def plan_stride(self):
        from scipy.optimize import minimize
        n_f = self.n_stance * 2

        def split(z):
            return z[:n_f], z[n_f], z[n_f + 1]               # stance forces, z_td, vz_td

        def obj(z):
            f, z_td, vz_td = split(z)
            return self.dt * float(np.sum(f ** 2)) * 1e-3

        def periodicity(z):                                  # end [vx,z,vz] == start [vx,z,vz] (periodic stride)
            f, z_td, vz_td = split(z)
            end = self._stride_states(f, z_td, vz_td)[-1]
            return np.array([end[1] - self.v_forward, end[2] - z_td, end[3] - vz_td])

        cons = [{"type": "eq", "fun": periodicity},
                {"type": "ineq", "fun": lambda z: self._stride_states(*split(z))[:, 2].min() - 0.02}]  # pz>0
        for j in range(self.n_stance):
            cons.append({"type": "ineq", "fun": (lambda z, j=j: self.p.mu * z[2 * j + 1] - abs(z[2 * j]))})
        bnds = [(-self.p.f_max, self.p.f_max), (0.0, self.p.f_max)] * self.n_stance + [(0.1, self.p.z0), (-3.0, 0.0)]
        z0 = np.concatenate([np.tile([0.0, self.p.mass * self.p.g], self.n_stance), [self.p.z0 * 0.9, -0.5]])
        res = minimize(obj, z0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-7})
        f, z_td, vz_td = split(res.x)
        return f.reshape(self.n_stance, 2), z_td, vz_td

    def simulate(self, n_strides: int = 5):
        """Repeat the periodic stride to produce a continuous running trajectory + per-stride telemetry."""
        forces, z_td, vz_td = self.plan_stride()
        m, g = self.p.mass, self.p.g
        prof = np.vstack([forces, np.zeros((self.n_flight, 2))])
        x = np.array([0.0, self.v_forward, z_td, vz_td])
        traj, sched = [x.copy()], []
        for _ in range(n_strides):
            for i, (fx, fz) in enumerate(prof):
                x = x + self.dt * np.array([x[1], fx / m, x[3], fz / m - g])
                traj.append(x.copy())
                sched.append(i < self.n_stance)
        return np.array(traj), np.array(sched), forces

    def capture_lyapunov(self, traj: np.ndarray) -> np.ndarray:
        """V_cap over the run: distance² of the capture point from the current stance foot (bounded => stable)."""
        px, vx, pz = traj[:, 0], traj[:, 1], np.maximum(traj[:, 2], 0.02)
        xi = px + vx * np.sqrt(pz / self.p.g)
        stride_len = self.v_forward * (self.n_stance + self.n_flight) * self.dt
        foot = np.round(px / max(stride_len, 1e-6)) * stride_len          # nearest planned footfall
        return 0.5 * (xi - foot) ** 2
