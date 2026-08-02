r"""Learned viability boundary of the energy-shaping closed loop — self-validating on the pendulum.

"The boundary" is the region-of-attraction / capturability boundary: the separatrix between states the
IDA-PBC controller returns to the target and states that escape (cross the antipodal saddle = "fall"). It is a
level set of the Lyapunov function ``H_d``; on the actuated pendulum it is ANALYTIC — the barrier at the
index-1 saddle, ``c* = ½ k π²`` — so a learned boundary can be checked against ground truth (milestone M0).

Pipeline (see ``docs/plans/2026-08-02-learned-viability-boundary/``):
    sample_viability   seeded, vectorised closed-loop rollouts → recover/fall labels (the deployed control law)
    LearnedBoundary    numpy-only logistic on curvature-aware features (u², v²) — no new dependency
    validate_boundary  IoU + per-class error of the learned set vs the analytic ROA on a HELD-OUT grid

# Preconditions: ``ViabilityConfig.k > 0`` (a genuine well); pinned ``dt``; horizon long enough to disambiguate.
# Postconditions: ``in_roa ≡ H_d < c*``; ``sample_viability`` is deterministic for a fixed config.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def _wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class ViabilityConfig:
    """Closed-loop + sampling parameters. The control law matches ``symbolic_ph`` / the visualization."""

    k: float = 24.0                  # IDA-PBC shaping gain (well stiffness) — sets c* = ½kπ²
    kd: float = 6.0                  # damping injection
    b: float = 0.12                  # natural joint damping
    m: float = 1.0
    ell: float = 1.0
    grav: float = 9.81
    target: float = math.pi          # θ* (inverted equilibrium)
    dt: float = 0.004
    horizon: float = 4.0
    grid_n: int = 41
    thetadot_max: float = 12.0
    settled_tol: float = 0.06        # |θ−θ*| to count as converged
    settled_vtol: float = 0.30       # |θ̇| to count as converged
    antipode_tol: float = 0.15       # reaching within this of θ*+π = crossed the saddle = fell
    max_grid: int = 40000
    seed: int = 0

    @property
    def mgl(self) -> float:
        return self.m * self.grav * self.ell

    @property
    def inertia(self) -> float:
        return self.m * self.ell ** 2


def separatrix_level(cfg: ViabilityConfig) -> float:
    r"""The ROA barrier ``c* = ½ k π²`` (``H_d`` at the antipodal index-1 saddle). # Post: ``> 0``."""
    if cfg.k <= 0:
        raise ValueError("k must be positive for a viable well")
    return 0.5 * cfg.k * math.pi ** 2


def hamiltonian_d(theta, thetadot, cfg: ViabilityConfig):
    r"""Closed-loop energy ``H_d = ½ I θ̇² + ½ k·wrap(θ−θ*)²`` (the Lyapunov function)."""
    return 0.5 * cfg.inertia * np.asarray(thetadot) ** 2 + 0.5 * cfg.k * _wrap(np.asarray(theta) - cfg.target) ** 2


def in_roa(theta, thetadot, cfg: ViabilityConfig):
    """Exact ROA membership (analytic ground truth): ``H_d < c*``."""
    return hamiltonian_d(theta, thetadot, cfg) < separatrix_level(cfg)


def control_torque(theta, thetadot, cfg: ViabilityConfig):
    r"""The shared IDA-PBC control ``τ = mgl·sinθ − k·wrap(θ−θ*) − kd·θ̇``.

    The potential-shaping part reproduces ``symbolic_ph.ida_pbc_potential_shaping`` (the deployed law); the
    ``−kd·θ̇`` term is damping assignment. Reused by rollouts so the learned boundary matches deployment.
    """
    theta = np.asarray(theta)
    thetadot = np.asarray(thetadot)
    return cfg.mgl * np.sin(theta) - cfg.k * _wrap(theta - cfg.target) - cfg.kd * thetadot


def _grid(cfg: ViabilityConfig, offset: float = 0.0) -> np.ndarray:
    """(N,2) grid of (θ, θ̇) over θ∈θ*±π, θ̇∈±θ̇max. ``offset`` (in cells) shifts a disjoint held-out grid."""
    n = cfg.grid_n
    if n * n > cfg.max_grid:
        n = int(math.isqrt(cfg.max_grid))
        logger.warning("grid capped to %dx%d (grid_n²=%d > max_grid=%d)", n, n, cfg.grid_n ** 2, cfg.max_grid)
    step_u, step_v = 2 * math.pi / n, 2 * cfg.thetadot_max / n
    us = -math.pi + (np.arange(n) + 0.5 + offset) * step_u
    vs = -cfg.thetadot_max + (np.arange(n) + 0.5 + offset) * step_v
    uu, vv = np.meshgrid(us, vs)
    return np.stack([cfg.target + uu.ravel(), vv.ravel()], axis=1)


def sample_viability(cfg: ViabilityConfig, offset: float = 0.0) -> "tuple[np.ndarray, np.ndarray]":
    r"""Vectorised closed-loop rollouts over the grid → (X, labels). label=1 recovers, 0 falls/does-not-settle.

    All grid points are integrated simultaneously (struct-of-arrays). A trajectory "falls" the moment it comes
    within ``antipode_tol`` of θ*+π (crossed the saddle); otherwise it "recovers" iff it settles near θ*.
    # Postconditions: deterministic (no RNG); ``len(X) == len(labels)``.
    """
    x = _grid(cfg, offset)
    theta = x[:, 0].copy()
    thetadot = x[:, 1].copy()
    fell = np.zeros(len(x), dtype=bool)
    anti = cfg.target + math.pi
    for _ in range(int(round(cfg.horizon / cfg.dt))):
        tau = control_torque(theta, thetadot, cfg)
        thetaddot = (-cfg.mgl * np.sin(theta) - cfg.b * thetadot + tau) / cfg.inertia
        thetadot = thetadot + thetaddot * cfg.dt
        theta = theta + thetadot * cfg.dt
        fell |= np.abs(_wrap(theta - anti)) < cfg.antipode_tol
    settled = (np.abs(_wrap(theta - cfg.target)) < cfg.settled_tol) & (np.abs(thetadot) < cfg.settled_vtol)
    return x, (settled & ~fell).astype(int)


def analytic_labels(x: np.ndarray, cfg: ViabilityConfig) -> np.ndarray:
    """Ground-truth ROA labels for a grid, from the analytic separatrix."""
    return in_roa(x[:, 0], x[:, 1], cfg).astype(int)


class LearnedBoundary:
    r"""Numpy-only logistic boundary on curvature-aware features ``(u², v²)`` with ``u = wrap(θ−θ*), v = θ̇``.

    The analytic separatrix ``½Iθ̇² + ½k u² = c*`` is linear in ``(u², v²)``, so this feature map can represent
    it exactly — the learned decision curve is a conic that should recover the ellipse. No new dependency.
    """

    def __init__(self, cfg: ViabilityConfig, iters: int = 6000, lr: float = 0.5) -> None:
        self.cfg, self.iters, self.lr = cfg, iters, lr
        self._w: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None

    def _features(self, x: np.ndarray) -> np.ndarray:
        u = _wrap(x[:, 0] - self.cfg.target)
        return np.stack([u * u, x[:, 1] ** 2], axis=1)

    def _design(self, x: np.ndarray) -> np.ndarray:
        f = (self._features(x) - self._mu) / self._sd
        return np.hstack([np.ones((len(f), 1)), f])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LearnedBoundary":
        """# Preconditions: ``len(x)==len(y)``; ``y ∈ {0,1}``. Deterministic (seeded init, full-batch GD)."""
        if len(x) != len(y):
            raise ValueError("x and y must have equal length")
        f = self._features(x)
        self._mu, self._sd = f.mean(axis=0), f.std(axis=0) + 1e-9
        design = self._design(x)
        w = np.random.RandomState(self.cfg.seed).normal(0.0, 0.01, design.shape[1])
        yv = y.astype(float)
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-design @ w))
            w -= self.lr * design.T @ (p - yv) / len(yv)
        self._w = w
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("call fit() before predict()")
        return (1.0 / (1.0 + np.exp(-self._design(x) @ self._w)) > 0.5).astype(int)


def validate_boundary(model: LearnedBoundary, cfg: ViabilityConfig) -> dict:
    """IoU + per-class error of the learned recoverable set vs the ANALYTIC ROA on a held-out (shifted) grid."""
    x = _grid(cfg, offset=0.5)                               # half-cell shift → disjoint from the training grid
    y_true, y_pred = analytic_labels(x, cfg), model.predict(x)
    inter = int(np.sum((y_true == 1) & (y_pred == 1)))
    union = int(np.sum((y_true == 1) | (y_pred == 1)))
    return {"iou": float(inter / union) if union else 1.0,
            "err_recover": float(np.mean(y_pred[y_true == 1] != 1)) if np.any(y_true == 1) else 0.0,
            "err_fall": float(np.mean(y_pred[y_true == 0] != 0)) if np.any(y_true == 0) else 0.0,
            "n": int(len(x))}
