r"""Arms as a reaction wheel — the flight-phase balance port the centroidal lump omits.

Our centroidal model tracks only the *total* angular momentum ``L`` (``L̇`` = external contact moment, zero in
flight) and the torso pitch — it does not resolve where the momentum sits. Physically the **arms** are a control:
carrying angular momentum ``I_arm·ω_arm`` they act as a **reaction wheel**, torquing the torso *without ground
contact* — so they stabilise the pitch **even in flight**, where the foot (stance-only) has no authority. This is
the diver/gymnast/cat strategy, and it is what the "arm swing = L-port" in the visualization abstracted.

Model: an inverted torso ``I_t·pitcḧ = mgl·sin(pitch) + τ_foot(stance) + τ_arm`` where ``τ_arm = −I_arm·ω̇_arm``
(accelerating the arm torques the torso oppositely), the arm angle integrates ``ω_arm`` and is **hard-limited** to
``±arm_range`` (a real arm cannot windmill forever — at the stop it gives no further outward torque). Foot torque
is bounded and available only in stance.

Measured: reaction-wheel arms extend the recoverable balance basin ~0.37 → ~0.66 (the arms save the flight-phase
states the foot loses), using their full ±arm_range swing.

# Preconditions: an unstable torso (``mgl > 0``). # Postconditions: the arm angle never exceeds ``arm_range``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmBalanceConfig:
    """Inverted-torso balance with a reaction-wheel arm + intermittent (stance) foot torque."""

    dt: float = 0.004
    inertia_torso: float = 1.6
    mgl: float = 6.0                 # gravitational toppling of the upright torso
    fall_pitch: float = 1.0
    ts: float = 0.2                  # stance / flight (foot torque only in stance)
    tf: float = 0.1
    tau_foot_max: float = 6.0
    inertia_arm: float = 0.25
    arm_range: float = 1.6           # hard mechanical limit on the arm angle (rad)
    arm_accel_max: float = 80.0
    kp: float = 14.0                 # torso-stabilising PD (desired torque)
    kd: float = 5.0
    horizon: float = 1.5

    @property
    def cycle(self) -> float:
        return self.ts + self.tf


def balance_rollout(x0: np.ndarray, cfg: ArmBalanceConfig, use_arm: bool,
                    ) -> "tuple[np.ndarray, np.ndarray]":
    r"""Roll the torso balance from initial ``(pitch, pitchdot)`` (arms start at rest) → (recovered, arm_amplitude).

    The desired stabilising torque ``−kp·pitch − kd·pitchdot`` is realised by the foot (stance, bounded); with
    ``use_arm`` the shortfall (especially in flight) is supplied by the reaction-wheel arm, which saturates at
    ``±arm_range`` (no outward torque past the stop). # Postconditions: ``|arm angle| ≤ arm_range`` throughout.
    """
    pitch, pd = x0[:, 0].copy(), x0[:, 1].copy()
    theta_a, omega_a = np.zeros(len(x0)), np.zeros(len(x0))
    fell, amp = np.zeros(len(x0), dtype=bool), np.zeros(len(x0))
    for i in range(int(round(cfg.horizon / cfg.dt))):
        stance = (i * cfg.dt % cfg.cycle) < cfg.ts
        tau_des = -cfg.kp * pitch - cfg.kd * pd
        tau_foot = np.clip(tau_des, -cfg.tau_foot_max, cfg.tau_foot_max) * (1.0 if stance else 0.0)
        tau_arm = np.zeros(len(x0))
        if use_arm:
            accel = np.clip(-(tau_des - tau_foot) / cfg.inertia_arm, -cfg.arm_accel_max, cfg.arm_accel_max)
            next_theta = theta_a + (omega_a + accel * cfg.dt) * cfg.dt
            saturated = (np.abs(next_theta) > cfg.arm_range) & (np.sign(accel) == np.sign(np.where(
                theta_a == 0.0, accel, theta_a)))
            accel = np.where(saturated, 0.0, accel)                 # at the mechanical stop: no outward torque
            omega_a = np.where(np.abs(theta_a) >= cfg.arm_range, 0.0, omega_a + accel * cfg.dt)
            theta_a = np.clip(theta_a + omega_a * cfg.dt, -cfg.arm_range, cfg.arm_range)
            tau_arm = -cfg.inertia_arm * accel                      # reaction torque on the torso
            amp = np.maximum(amp, np.abs(theta_a))
        pd = pd + (cfg.mgl * np.sin(pitch) + tau_foot + tau_arm) / cfg.inertia_torso * cfg.dt
        pitch = pitch + pd * cfg.dt
        fell |= np.abs(pitch) > cfg.fall_pitch
    return ~fell, amp


def recoverable_basin(cfg: ArmBalanceConfig, use_arm: bool, n: int = 41) -> dict:
    """Fraction of a (pitch, pitchdot) grid the controller recovers, and the peak arm swing used."""
    th = np.linspace(-0.9, 0.9, n)
    pdot = np.linspace(-4.0, 4.0, n)
    tt, pp = np.meshgrid(th, pdot)
    x0 = np.stack([tt.ravel(), pp.ravel()], axis=1)
    recovered, amp = balance_rollout(x0, cfg, use_arm)
    return {"recovered_fraction": float(recovered.mean()),
            "max_arm_swing": float(amp[recovered].max()) if recovered.any() else 0.0,
            "median_arm_swing": float(np.median(amp[recovered])) if recovered.any() else 0.0}
