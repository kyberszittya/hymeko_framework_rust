"""Contact-scheduled centroidal hop MPC — a PLANNED flight phase inside the Lyapunov region.

The retracted capture-widening went airborne as an *exploit* (uncontrolled, 27 rad/s, the
certificate passed by accident of unphysical dynamics). A **planned** flight phase is the
opposite: contact-scheduled MPC deliberately leaves the ground for a ballistic flight, yet keeps
the centroidal state within a **recoverable (capturability) Lyapunov region** — exactly how real
jumping/running robots (and people) move. The momentary loss of *static* stability is bounded and
recovered by a controlled landing.

This is the reduced **centroidal** model (COM point mass + ground reaction force), embodiment-
agnostic — the AIBO and the human differ only by mass / stand height / force limits. State
x = [px, vx, pz, vz]; control = ground reaction force F = [Fx, Fz] during STANCE (F = 0 during
FLIGHT). Dynamics: v̇x = Fx/m, v̇z = Fz/m − g. We plan a forward hop (crouch→launch→FLIGHT→land)
by single-shooting trajectory optimisation:

    min  Σ dt·‖F‖²  +  w·‖x_N − x_goal‖²
    s.t. F = 0 during flight;  Fz ∈ [0, F_max];  |Fx| ≤ μ·Fz (friction cone);  pz ≥ 0

and certify the CAPTURABILITY Lyapunov V_cap = ½‖ξ − p_foot‖² (ξ = capture point) stays bounded —
the flight is a controlled excursion, not a fall. Full-body realisation (joint torques via the
contact-Jacobian) is the next layer; here we plan and certify the centroidal trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HopParams:
    """Embodiment parameters (AIBO vs human differ only here)."""

    mass: float = 2.0                # kg (AIBO ~2; human model ~15)
    z0: float = 0.23                 # standing COM height (AIBO 0.23; human 0.645)
    mu: float = 0.9                  # friction coefficient
    f_max: float = 60.0              # ground-force cap (realistic actuation)
    g: float = 9.81


@dataclass
class CentroidalHopMPC:
    """Plan a forward hop with a scheduled flight phase; certify capturability stays bounded."""

    p: HopParams = field(default_factory=HopParams)
    dt: float = 0.02
    n_stance1: int = 18              # crouch + launch
    n_flight: int = 22               # ballistic flight (F = 0)
    n_stance2: int = 20              # landing + settle
    x_target: float = 0.25           # forward hop distance (m)

    def _schedule(self) -> np.ndarray:
        """Per-knot contact flag: True = stance (foot on ground), False = flight."""
        return np.array([True] * self.n_stance1 + [False] * self.n_flight + [True] * self.n_stance2)

    def _rollout(self, forces: np.ndarray) -> np.ndarray:
        """Single-shooting integration of the centroidal dynamics under a force profile (N x 2)."""
        m, g = self.p.mass, self.p.g
        x = np.array([0.0, 0.0, self.p.z0, 0.0])                   # [px, vx, pz, vz] at rest
        traj = [x.copy()]
        for fx, fz in forces:
            x = x + self.dt * np.array([x[1], fx / m, x[3], fz / m - g])
            traj.append(x.copy())
        return np.array(traj)

    def capture_lyapunov(self, traj: np.ndarray) -> np.ndarray:
        """V_cap(t) = ½‖ξ_x − p_footx‖²,  ξ_x = px + vx·√(pz/g) — capturability energy (bounded => recoverable)."""
        px, vx, pz = traj[:, 0], traj[:, 1], np.maximum(traj[:, 2], 0.02)
        xi = px + vx * np.sqrt(pz / self.p.g)
        return 0.5 * (xi - self.x_target) ** 2                     # distance of the capture point from the landing foot

    def plan(self):
        from scipy.optimize import minimize
        sched = self._schedule()
        n = len(sched)
        stance_idx = np.where(sched)[0]

        def unpack(z):                                             # z holds Fx,Fz only on stance knots
            f = np.zeros((n, 2))
            f[stance_idx] = z.reshape(-1, 2)
            return f

        def obj(z):
            f = unpack(z)
            traj = self._rollout(f)
            xf = traj[-1]
            goal = np.array([self.x_target, 0.0, self.p.z0, 0.0])
            return self.dt * float(np.sum(f ** 2)) * 1e-3 + 40.0 * float(np.sum((xf - goal) ** 2))

        cons = []
        for j, k in enumerate(stance_idx):                        # friction cone + non-penetration during stance
            cons.append({"type": "ineq", "fun": (lambda z, j=j: self.p.mu * z[2 * j + 1] - abs(z[2 * j]))})
        cons.append({"type": "ineq", "fun": lambda z: self._rollout(unpack(z))[:, 2].min()})   # pz >= 0
        bnds = [(-self.p.f_max, self.p.f_max), (0.0, self.p.f_max)] * len(stance_idx)
        z0 = np.tile([0.0, self.p.mass * self.p.g], len(stance_idx)).astype(float)
        res = minimize(obj, z0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-6})
        f = unpack(res.x)
        traj = self._rollout(f)
        return traj, f, sched
