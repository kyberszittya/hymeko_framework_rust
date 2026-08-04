r"""Centroidal running closed loop — the L-regulated runner whose capturability basin M2 certifies.

The reduced state is ``x = (z, ż, L, pitch)``: CoM height + rate, centroidal angular momentum, and torso pitch.
The closed loop is the scripted L-regulator visualised in the artifact — a SLIP-like vertical bounce (stance
spring + flight ballistic) plus the angular-momentum ports (foot placement + arm swing + a shoulder/pitch hold)
that damp ``L`` and pull the pitch upright. A run "falls" the moment the pitch crosses ``fall_pitch``.

This mirrors ``viability.closed_loop_step`` one dimension up: one shared ``centroidal_step`` integrator, reused by
the rollout labeller here and by the neural certificate's lookahead, so the certified basin is about exactly the
dynamics that produced the labels.

# Preconditions: ``dt`` pinned; ``fall_pitch > 0``. # Postconditions: rollouts are deterministic (no RNG).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CentroidalConfig:
    """Centroidal closed-loop + sampling parameters (defaults mirror the visualization)."""

    z0: float = 0.92                 # nominal CoM height
    ts: float = 0.20                 # stance duration
    tf: float = 0.10                 # flight duration
    grav: float = 9.81
    inertia: float = 1.6             # torso inertia about the pitch axis
    kz: float = 260.0                # stance vertical spring
    dz: float = 45.0                 # stance vertical damping
    # NOTE: at this (strong) regulation the fall is pitch-dominated (L is weakly coupled), so the basin is close
    # to {|pitch| < fall_pitch} and the (L,pitch) certificate is a valid but only mildly-nonlinear demonstration.
    # Softening the regulation makes L genuinely matter, but then the attractor is a LIMIT CYCLE (L_ss, pitch_ss)
    # and a point-Lyapunov V collapses — limit-cycle-aware certification is the honest follow-up (see the report).
    torque_bias: float = 2.4         # (r_foot−r_com)×F bias on L during stance
    l_damp: float = 9.5              # L-port regulation gain (foot placement + arm swing)
    pitch_gain: float = 7.0          # torso-pitch attitude regulation
    fall_pitch: float = 1.25         # |pitch| beyond this = fell
    dt: float = 0.004
    horizon: float = 1.5
    settle_pitch: float = 0.15
    settle_l: float = 0.5
    # sampling box for the reduced state (z, ż, L, pitch)
    z_span: float = 0.15
    zd_max: float = 1.5
    l_max: float = 6.0
    pitch_max: float = 1.4
    grid_n: int = 9
    max_samples: int = 100_000
    seed: int = 0

    @property
    def cycle(self) -> float:
        return self.ts + self.tf


def centroidal_step(state: np.ndarray, t: float, cfg: CentroidalConfig) -> np.ndarray:
    r"""One semi-implicit step of the L-regulated centroidal closed loop. ``state`` columns = (z, ż, L, pitch).

    The single shared integrator for the rollout labeller and the certificate lookahead. # Post: same shape.
    """
    z, zd, ll, pitch = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    stance = (t % cfg.cycle) < cfg.ts
    push = cfg.kz * (cfg.z0 - z) - cfg.dz * zd + cfg.grav        # stance spring + gravity-comp + thrust
    zdd = np.where(stance, push - cfg.grav, -cfg.grav)           # flight = ballistic
    zd = zd + zdd * cfg.dt
    z = z + zd * cfg.dt
    ldot = np.where(stance, cfg.torque_bias, 0.0) - cfg.l_damp * ll   # contact torque (stance) − port damping
    ll = ll + ldot * cfg.dt
    pitch = pitch + (ll / cfg.inertia) * cfg.dt - cfg.pitch_gain * pitch * cfg.dt   # ∫L/I − attitude hold
    return np.stack([z, zd, ll, pitch], axis=1)


def _sample_grid(cfg: CentroidalConfig, offset: float = 0.0) -> np.ndarray:
    """Deterministic product grid over (z, ż, L, pitch); subsampled deterministically if over ``max_samples``."""
    n = cfg.grid_n
    axes = [
        cfg.z0 + np.linspace(-cfg.z_span, cfg.z_span, n) + offset * (2 * cfg.z_span / n),
        np.linspace(-cfg.zd_max, cfg.zd_max, n) + offset * (2 * cfg.zd_max / n),
        np.linspace(-cfg.l_max, cfg.l_max, n) + offset * (2 * cfg.l_max / n),
        np.linspace(-cfg.pitch_max, cfg.pitch_max, n) + offset * (2 * cfg.pitch_max / n),
    ]
    grid = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)
    if len(grid) > cfg.max_samples:
        idx = np.linspace(0, len(grid) - 1, cfg.max_samples).astype(int)
        logger.warning("centroidal grid %d capped to %d samples", len(grid), cfg.max_samples)
        grid = grid[idx]
    return grid


def centroidal_rollout(cfg: CentroidalConfig, x0: np.ndarray | None = None) -> "tuple[np.ndarray, np.ndarray]":
    r"""Vectorised rollouts of the closed loop → (X0, labels). label=1 recovers (never falls, pitch+L settle).

    A trajectory "falls" the instant ``|pitch| > fall_pitch``; otherwise it "recovers" iff the pitch and angular
    momentum settle. # Postconditions: deterministic; ``len(X0) == len(labels)``.
    """
    x0 = _sample_grid(cfg) if x0 is None else x0
    state = x0.copy()
    fell = np.zeros(len(x0), dtype=bool)
    steps = int(round(cfg.horizon / cfg.dt))
    for i in range(steps):
        state = centroidal_step(state, i * cfg.dt, cfg)
        fell |= np.abs(state[:, 3]) > cfg.fall_pitch
    settled = (np.abs(state[:, 3]) < cfg.settle_pitch) & (np.abs(state[:, 2]) < cfg.settle_l)
    return x0, (settled & ~fell).astype(int)
