"""Basic time-series forecasting datasets — pure-NumPy generators.

Each generator returns ``(series, dt)`` where ``series`` is a 1-D
``np.ndarray`` of length ``n_steps`` and ``dt`` is the sampling
interval used (recorded in metadata; not strictly required by the
trainer).

Datasets
--------
- ``sine`` — pure 1 Hz sine with optional additive Gaussian noise.
- ``noisy_sine`` — sine + harmonic + Gaussian noise (forecasting is
  trickier; tests robustness).
- ``mackey_glass`` — Mackey-Glass delay-differential equation
  (canonical chaotic time series).
- ``lorenz_x`` — x-component of the Lorenz attractor (chaotic 3-state).

All series are mean-zero, unit-stdev normalised at the end so that
forecasting losses are comparable across datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class TSConfig:
    n_steps: int = 4096
    seed: int = 0
    noise_std: float = 0.05


def sine(cfg: TSConfig) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(cfg.seed)
    dt = 0.05
    t = np.arange(cfg.n_steps) * dt
    y = np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, cfg.noise_std, cfg.n_steps)
    return _normalize(y), dt


def noisy_sine(cfg: TSConfig) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(cfg.seed)
    dt = 0.05
    t = np.arange(cfg.n_steps) * dt
    y = (
        np.sin(2 * np.pi * 1.0 * t)
        + 0.4 * np.sin(2 * np.pi * 2.7 * t)
        + 0.2 * np.sin(2 * np.pi * 5.3 * t)
        + rng.normal(0, cfg.noise_std * 2.0, cfg.n_steps)
    )
    return _normalize(y), dt


def mackey_glass(cfg: TSConfig, beta: float = 0.2, gamma: float = 0.1,
                  tau: int = 17, n: float = 10.0) -> Tuple[np.ndarray, float]:
    """Mackey-Glass DDE — discretised Euler with delay ``tau``.

    Canonical chaotic dynamics at ``tau >= 17`` (Lyapunov > 0).
    """
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0
    burn = max(tau * 8, 256)
    total = cfg.n_steps + burn
    y = np.empty(total + tau, dtype=np.float64)
    y[:tau] = 1.2 + rng.normal(0, 0.01, tau)
    for k in range(tau, total + tau - 1):
        y[k + 1] = y[k] + dt * (beta * y[k - tau] / (1.0 + y[k - tau] ** n) - gamma * y[k])
    out = y[burn + tau:]
    return _normalize(out), dt


def lorenz_x(cfg: TSConfig, sigma: float = 10.0, rho: float = 28.0,
              beta: float = 8.0 / 3.0) -> Tuple[np.ndarray, float]:
    """Lorenz attractor x-component via RK4."""
    rng = np.random.default_rng(cfg.seed)
    dt = 0.01
    burn = 5000
    total = cfg.n_steps + burn
    x = np.empty(total)
    state = np.array([1.0, 1.0, 1.0]) + rng.normal(0, 0.01, 3)

    def rhs(s):
        return np.array([sigma * (s[1] - s[0]),
                          s[0] * (rho - s[2]) - s[1],
                          s[0] * s[1] - beta * s[2]])

    for k in range(total):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        x[k] = state[0]
    return _normalize(x[burn:]), dt


def _normalize(y: np.ndarray) -> np.ndarray:
    y = y.astype(np.float32, copy=False)
    return (y - y.mean()) / (y.std() + 1e-8)


DATASETS = {
    "sine": sine,
    "noisy_sine": noisy_sine,
    "mackey_glass": mackey_glass,
    "lorenz_x": lorenz_x,
}


def load(name: str, cfg: TSConfig | None = None) -> Tuple[np.ndarray, float]:
    cfg = cfg or TSConfig()
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; options: {sorted(DATASETS)}")
    return DATASETS[name](cfg)


def windowed(series: np.ndarray, window: int, horizon: int = 1
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a window of length ``window`` with target ``horizon`` steps
    ahead.  Returns ``(X, y)`` with X shape ``(N, window)`` and y shape
    ``(N,)``.
    """
    if series.ndim != 1:
        raise ValueError(f"series must be 1D, got shape {series.shape}")
    if window < 1 or horizon < 1:
        raise ValueError(f"window={window} horizon={horizon} must be >= 1")
    n = len(series) - window - horizon + 1
    if n <= 0:
        raise ValueError(f"series too short ({len(series)}) for window={window}+horizon={horizon}")
    X = np.lib.stride_tricks.sliding_window_view(series, window_shape=window)[: n]
    y = series[window + horizon - 1 : window + horizon - 1 + n]
    return X.copy(), y.copy()


__all__ = ["TSConfig", "DATASETS", "load", "windowed",
           "sine", "noisy_sine", "mackey_glass", "lorenz_x"]
