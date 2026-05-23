"""Reference paths for the lateral-tracking benchmark.

A ``Track`` is a parametric curve over arc-length ``s`` with:
- ``position(s)``  → (x_ref, y_ref)
- ``heading(s)``   → ψ_ref (tangent angle)
- ``curvature(s)`` → κ_ref (signed)
- ``project(x, y)`` → (s_nearest, lateral_error, heading_error)

The benchmark uses three canonical paths:

- ``straight_track``      — y_ref ≡ 0 (sanity, all controllers should solve)
- ``sinusoid_track``      — y_ref = A sin(2π x / λ); steady-state cornering
- ``s_curve_track``       — composite cubic + linear; lane-change-style transient
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class Track:
    """Discretised reference path."""

    s: np.ndarray       # arc-length samples
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray     # tangent heading at each sample
    kappa: np.ndarray   # signed curvature at each sample

    def project(self, x: float, y: float) -> Tuple[int, float, float, float]:
        """Find nearest path point.  Returns (idx, lateral_err, heading_err, kappa)."""
        d2 = (self.x - x) ** 2 + (self.y - y) ** 2
        idx = int(np.argmin(d2))
        # Lateral error: signed perpendicular distance from path tangent.
        dx, dy = x - self.x[idx], y - self.y[idx]
        psi_r = self.psi[idx]
        # Tangent unit vec (cos ψ, sin ψ); normal unit vec (−sin ψ, cos ψ).
        lat = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
        return idx, float(lat), float(psi_r), float(self.kappa[idx])

    def lookahead(self, idx: int, distance: float) -> Tuple[float, float]:
        """Return (x, y) of the point ``distance`` ahead along the path from idx."""
        target_s = self.s[idx] + distance
        if target_s >= self.s[-1]:
            return float(self.x[-1]), float(self.y[-1])
        j = int(np.searchsorted(self.s, target_s))
        j = min(j, len(self.s) - 1)
        return float(self.x[j]), float(self.y[j])


def _build_from_xy(x: np.ndarray, y: np.ndarray) -> Track:
    """Compute (s, psi, kappa) by finite differences."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.cumsum(ds) - ds[0]
    psi = np.arctan2(dy, dx)
    # Signed curvature κ = (x' y'' − y' x'') / (x'² + y'²)^{3/2}
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-9
    kappa = (dx * ddy - dy * ddx) / denom
    return Track(s=s, x=x, y=y, psi=psi, kappa=kappa)


def straight_track(length: float = 100.0, n: int = 1001) -> Track:
    x = np.linspace(0.0, length, n)
    y = np.zeros_like(x)
    return _build_from_xy(x, y)


def sinusoid_track(length: float = 100.0, amplitude: float = 2.0,
                    wavelength: float = 30.0, n: int = 2001) -> Track:
    x = np.linspace(0.0, length, n)
    y = amplitude * np.sin(2.0 * np.pi * x / wavelength)
    return _build_from_xy(x, y)


def s_curve_track(length: float = 100.0, lane_width: float = 3.5, n: int = 2001
                    ) -> Track:
    """Smooth lane-change: y ramps from 0 → ``lane_width`` via a tanh
    centred at the midpoint of the track."""
    x = np.linspace(0.0, length, n)
    centre = length / 2.0
    y = lane_width * 0.5 * (1.0 + np.tanh((x - centre) / 5.0))
    return _build_from_xy(x, y)


__all__ = ["Track", "straight_track", "sinusoid_track", "s_curve_track"]
