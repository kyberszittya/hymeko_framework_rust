"""Kinematic bicycle model for lateral-tracking experiments.

State (x, y, psi, v) — global position + heading + longitudinal
velocity.  Control (delta) — front-wheel steering angle.

ẋ   = v cos(ψ)
ẏ   = v sin(ψ)
ψ̇   = (v / L) tan(δ)
v̇   = a  (constant velocity assumed when not given)

Integration: RK4 by default.  The model is exact for the rear-axle
midpoint; controllers must respect that lateral error is measured at
the rear-axle, not the front wheel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BicycleParams:
    L: float = 2.5          # wheelbase (m)
    v_target: float = 5.0   # longitudinal velocity setpoint (m/s)
    delta_max: float = 0.6  # ≈ 34° steering limit (rad)
    a_max: float = 1.0      # longitudinal accel cap (m/s²)


@dataclass
class BicycleState:
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0
    v: float = 5.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.v], dtype=np.float64)

    @classmethod
    def from_array(cls, arr) -> "BicycleState":
        x, y, psi, v = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        return cls(x=x, y=y, psi=psi, v=v)


class BicycleVehicle:
    """RK4-integrated kinematic bicycle."""

    def __init__(self, params: BicycleParams | None = None) -> None:
        self.p = params or BicycleParams()

    def derivative(self, state: BicycleState, delta: float, a: float) -> np.ndarray:
        delta = float(np.clip(delta, -self.p.delta_max, self.p.delta_max))
        a = float(np.clip(a, -self.p.a_max, self.p.a_max))
        return np.array([
            state.v * np.cos(state.psi),
            state.v * np.sin(state.psi),
            state.v / self.p.L * np.tan(delta),
            a,
        ], dtype=np.float64)

    def step(self, state: BicycleState, delta: float, dt: float,
              a: float = 0.0) -> BicycleState:
        s0 = state.to_array()
        k1 = self.derivative(BicycleState.from_array(s0), delta, a)
        k2 = self.derivative(BicycleState.from_array(s0 + 0.5 * dt * k1), delta, a)
        k3 = self.derivative(BicycleState.from_array(s0 + 0.5 * dt * k2), delta, a)
        k4 = self.derivative(BicycleState.from_array(s0 + dt * k3), delta, a)
        new = s0 + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        # Keep ψ in (-π, π]
        new[2] = ((new[2] + np.pi) % (2.0 * np.pi)) - np.pi
        return BicycleState.from_array(new)


__all__ = ["BicycleParams", "BicycleState", "BicycleVehicle"]
