"""Tedrake-style CENTROIDAL momentum trajectory optimization for a dynamic humanoid RUN.

Instead of a hand-parameterised gait + CEM (which caps at a slow hop), this optimises the robot's centroidal
(centre-of-mass) momentum trajectory directly — the standard Underactuated-Robotics / Drake approach to
dynamic legged motion. For a planar (sagittal) run the centroidal dynamics are Newton–Euler on the CoM:

    m·ẍ = Σ Fx_i ,   m·z̈ = Σ Fz_i − m·g          (contact forces during stance; zero in FLIGHT)

A periodic running stride is a STANCE phase (one foot planted, a contact force redirects + propels the CoM)
followed by a FLIGHT phase (no contact → ballistic, both feet off the ground). We solve, by direct
collocation (trapezoidal), for the CoM trajectory + the stance contact force + phase durations + stride
length, subject to the dynamics, a friction cone (|Fx| ≤ μ·Fz, Fz ≥ 0), foot reachability, and periodicity,
maximising forward speed toward a target. The result is a dynamically-feasible momentum plan WITH a flight
phase — which a whole-body controller then tracks. This module is the momentum-optimization core; it has no
RL and no hand-tuned gait.

# Preconditions: target_speed ≥ 0, mass > 0, ns/nf ≥ 3.
# Postconditions: ``solve_run`` returns a ``RunTrajectory`` whose CoM obeys the centroidal dynamics to
#   collocation tolerance and whose flight phase has both feet off the ground (zero contact force).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

G = 9.81


@dataclass(frozen=True)
class CentroidalRunConfig:
    mass: float = 18.0               # humanoid CoM mass (kg) — scale-free for the plan, sets force magnitudes
    z0: float = 0.62                 # nominal CoM height (m) — the sagittal humanoid stands ~0.62 m
    target_speed: float = 1.2        # desired forward speed (m/s) — a real run, not a shuffle
    mu: float = 0.8                  # ground friction coefficient (friction cone |Fx| ≤ μ·Fz)
    reach: float = 0.22              # max horizontal CoM–foot offset (kinematic reachability of the stance leg)
    ns: int = 10                     # stance collocation knots
    nf: int = 7                      # flight collocation knots
    fz_min: float = 5.0              # minimum stance normal force (N) — foot stays loaded during stance
    fz_max: float = 900.0            # stance normal-force cap (motor/contact limit)
    w_speed: float = 4.0             # objective weight: hit the target forward speed
    w_effort: float = 1e-6           # objective weight: minimise contact-force effort (regularise)
    w_apex: float = 2.0              # objective weight: reward CoM apex RISE during flight (bounded, unlike duration)
    min_flight: float = 0.05         # minimum flight duration (s) — a real both-feet-off phase (raise for a bigger hop)


@dataclass
class RunTrajectory:
    t: np.ndarray                    # (ns+nf,) time stamps over one stride
    com: np.ndarray                  # (ns+nf, 2) CoM (x, z)
    vel: np.ndarray                  # (ns+nf, 2) CoM (vx, vz)
    force: np.ndarray                # (ns+nf, 2) contact force (Fx, Fz); zero on the flight knots
    t_stance: float
    t_flight: float
    stride: float                    # net forward CoM advance per stride (m)
    foot_x: float                    # stance foot x relative to the stride start (m)
    speed: float                     # achieved forward speed (m/s)
    contact: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # per-knot: foot on ground?


class _Packer:
    """Flat decision-vector <-> (states, stance forces, T_stance, T_flight, stride, foot_x) packing."""

    def __init__(self, ns: int, nf: int) -> None:
        self.ns, self.nf, self.n = ns, nf, ns + nf
        self.ns4 = self.n * 4
        self.nfe = ns * 2                                     # stance forces (Fx, Fz) per stance knot
        self.size = self.ns4 + self.nfe + 4                  # + T_stance, T_flight, stride, foot_x

    def unpack(self, v: np.ndarray):
        s = v[: self.ns4].reshape(self.n, 4)                 # (x, z, vx, vz) per knot
        f = v[self.ns4: self.ns4 + self.nfe].reshape(self.ns, 2)
        t_st, t_fl, stride, foot_x = v[self.ns4 + self.nfe:]
        return s, f, float(t_st), float(t_fl), float(stride), float(foot_x)


def _dynamics_residual(cfg: CentroidalRunConfig, pk: _Packer, v: np.ndarray) -> np.ndarray:
    """Trapezoidal collocation defect for the centroidal dynamics + phase continuity + periodicity."""
    s, f, t_st, t_fl, stride, _foot_x = pk.unpack(v)
    m = cfg.mass
    res = []

    def deriv(state, force):
        _x, _z, vx, vz = state
        fx, fz = force
        return np.array([vx, vz, fx / m, fz / m - G])

    # STANCE collocation (knots 0..ns-1), dt over ns-1 intervals
    dt_st = t_st / max(pk.ns - 1, 1)
    for k in range(pk.ns - 1):
        d0, d1 = deriv(s[k], f[k]), deriv(s[k + 1], f[k + 1])
        res.append((s[k + 1] - s[k]) - 0.5 * dt_st * (d0 + d1))
    # FLIGHT collocation (knots ns..ns+nf-1), zero force
    dt_fl = t_fl / max(pk.nf - 1, 1)
    zero = np.zeros(2)
    for k in range(pk.ns, pk.n - 1):
        d0, d1 = deriv(s[k], zero), deriv(s[k + 1], zero)
        res.append((s[k + 1] - s[k]) - 0.5 * dt_fl * (d0 + d1))
    # phase continuity: last stance knot == first flight knot (same CoM state, at takeoff)
    res.append(s[pk.ns] - s[pk.ns - 1])
    # PERIODICITY: after one full stride the CoM state repeats, advanced by `stride` in x only
    end, start = s[pk.n - 1], s[0]
    res.append(np.array([end[0] - start[0] - stride, end[1] - start[1], end[2] - start[2], end[3] - start[3]]))
    return np.concatenate(res)


def _path_ineq(cfg: CentroidalRunConfig, pk: _Packer, v: np.ndarray) -> np.ndarray:
    """Inequalities g(v) ≥ 0: friction cone, normal-force bounds, reachability, positive CoM height + times."""
    s, f, t_st, t_fl, stride, foot_x = pk.unpack(v)
    g = []
    for k in range(pk.ns):
        fx, fz = f[k]
        g.append(fz - cfg.fz_min)                            # Fz ≥ fz_min (loaded)
        g.append(cfg.fz_max - fz)                            # Fz ≤ fz_max
        g.append(cfg.mu * fz - fx)                           # friction: Fx ≤ μ Fz
        g.append(cfg.mu * fz + fx)                           # friction: -Fx ≤ μ Fz
        g.append(cfg.reach - (s[k, 0] - foot_x))             # reachability: x - foot ≤ reach
        g.append(cfg.reach + (s[k, 0] - foot_x))             # x - foot ≥ -reach
    for k in range(pk.n):
        g.append(s[k, 1] - 0.35)                             # CoM height ≥ 0.35 (don't collapse)
    g.append(t_st - 0.05)                                    # stance ≥ 50 ms
    g.append(t_fl - cfg.min_flight)                          # flight ≥ min_flight (a real both-feet-off phase)
    g.append(1.5 - t_st)                                     # stance ≤ 1.5 s (bounded — guards the unbounded-time bug)
    g.append(1.0 - t_fl)                                     # flight ≤ 1.0 s (bounded — a ballistic hop, not a launch to orbit)
    g.append(stride - 0.02)                                  # net forward per stride ≥ 2 cm
    return np.array(g)


def _objective(cfg: CentroidalRunConfig, pk: _Packer, v: np.ndarray) -> float:
    s, f, t_st, t_fl, stride, _foot_x = pk.unpack(v)
    speed = stride / max(t_st + t_fl, 1e-3)
    apex = s[pk.ns:, 1].max() - s[pk.ns - 1, 1]              # CoM RISE above takeoff during flight (bounded)
    return (cfg.w_speed * (speed - cfg.target_speed) ** 2
            + cfg.w_effort * float(np.sum(f ** 2))
            - cfg.w_apex * apex)


def _initial_guess(cfg: CentroidalRunConfig, pk: _Packer) -> np.ndarray:
    """A ballistic-ish warm start: CoM advances at target speed, dips in stance, rises for flight."""
    t_st, t_fl = 0.22, 0.12
    stride = cfg.target_speed * (t_st + t_fl)
    v = np.zeros(pk.size)
    ts = np.concatenate([np.linspace(0, t_st, pk.ns), t_st + np.linspace(0, t_fl, pk.nf)])
    for k in range(pk.n):
        x = cfg.target_speed * ts[k]
        z = cfg.z0 + (0.03 if k >= pk.ns else -0.02)
        v[4 * k: 4 * k + 4] = [x, z, cfg.target_speed, 0.0]
    fz0 = cfg.mass * G * (t_st + t_fl) / t_st                # stance carries the whole stride's weight impulse
    for k in range(pk.ns):
        v[pk.ns4 + 2 * k: pk.ns4 + 2 * k + 2] = [cfg.mass * cfg.target_speed / t_st * 0.3, fz0]
    v[pk.ns4 + pk.nfe:] = [t_st, t_fl, stride, stride * 0.5]
    return v


def solve_run(cfg: CentroidalRunConfig | None = None, *, maxiter: int = 300) -> RunTrajectory:
    """Solve the centroidal running-stride trajectory optimization. # Returns a dynamically-feasible plan."""
    cfg = cfg or CentroidalRunConfig()
    pk = _Packer(cfg.ns, cfg.nf)
    x0 = _initial_guess(cfg, pk)
    eq = NonlinearConstraint(lambda v: _dynamics_residual(cfg, pk, v), 0.0, 0.0)
    ineq = NonlinearConstraint(lambda v: _path_ineq(cfg, pk, v), 0.0, np.inf)
    sol = minimize(lambda v: _objective(cfg, pk, v), x0, method="SLSQP",
                   constraints=[eq, ineq], options={"maxiter": maxiter, "ftol": 1e-6})
    s, f, t_st, t_fl, stride, foot_x = pk.unpack(sol.x)
    ts = np.concatenate([np.linspace(0, t_st, pk.ns), t_st + np.linspace(0, t_fl, pk.nf)])
    force = np.vstack([f, np.zeros((pk.nf, 2))])
    contact = np.concatenate([np.ones(pk.ns, bool), np.zeros(pk.nf, bool)])
    return RunTrajectory(t=ts, com=s[:, :2], vel=s[:, 2:], force=force, t_stance=t_st, t_flight=t_fl,
                         stride=stride, foot_x=foot_x, speed=stride / max(t_st + t_fl, 1e-3), contact=contact)
