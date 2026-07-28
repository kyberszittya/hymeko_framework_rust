"""Scenario-side deterministic locomotion controllers for the AIBO quadruped.

Adds YAW (turning) to the straight-only base trot via differential left/right
stride amplitude — a scripted skid-steer, NOT RL. Measured yaw authority:
~168 deg / 1000 steps (stable in one turn direction; the other direction needs
gait tuning — recorded, not hidden). ``drive=0`` degenerates to a PD stand-hold
(the STOP/HOLD controller). One parameterised gait covers WALK / TURN / STOP —
no per-mode function proliferation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# leg order fl(0), fr(1), bl(2), br(3); left legs {0,2}, right legs {1,3}
_DIAG_PHASE = (0.0, float(np.pi), float(np.pi), 0.0)

_PI = float(np.pi)
# Named left-right base-gait phase patterns (leg order fl,fr,bl,br). The distinction that matters for
# the crab-symmetry campaign is INSTANTANEOUS left-right symmetry:
#   diag  = diagonal trot (fl,br) vs (fr,bl) — instantaneously left-right ASYMMETRIC (the measured
#           root of the omni-crab one-sidedness; see reports/2026-07-28-aibo-crab-symmetry-resolved.md)
#   bound = front pair vs back pair — instantaneously left-right SYMMETRIC (fl==fr, bl==br): the
#           symmetric substrate Phase A tests (does a symmetric scaffold let the crab reach both sides?)
#   pace  = left pair vs right pair — the two SIDES are half a period apart (still not instantaneous)
#   pronk = all four in phase — symmetric but ~no net locomotion (degenerate control)
GAIT_PHASES: "dict[str, tuple[float, float, float, float]]" = {
    "diag": (0.0, _PI, _PI, 0.0),
    "bound": (0.0, 0.0, _PI, _PI),
    "pace": (0.0, _PI, 0.0, _PI),
    "pronk": (0.0, 0.0, 0.0, 0.0),
}


def body_yaw(env: object) -> float:
    """World-frame yaw of the torso (radians)."""
    m = np.asarray(env.data.xmat[env.torso]).reshape(3, 3)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def goal_bearing(env: object) -> float:
    dx = float(env.goal[0]) - float(env.data.xpos[env.torso, 0])
    dy = float(env.goal[1]) - float(env.data.xpos[env.torso, 1])
    return float(np.arctan2(dy, dx))


def wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def heading_error(env: object) -> float:
    """Signed bearing to the goal relative to the torso heading (radians)."""
    return wrap(goal_bearing(env) - body_yaw(env))


@dataclass(frozen=True)
class SteeredTrotGait:
    """Diagonal PD-trot with a BIDIRECTIONAL yaw command via reduce-inner stride.

    Yaw is produced by SLOWING the inner (turn-side) legs' stride while the outer
    legs keep nominal amplitude -- a differential-drive skid-steer that never
    amplifies a side above nominal and adds no abduction bias. That symmetry is
    what makes it stable in BOTH turn directions (the earlier amplify-outer +
    abduction primitive over-drove one diagonal and tipped: see
    reports/2026-07-27-aibo-simple-scenarios / the yaw diagnostic).

    Convention: ``yaw_cmd > 0`` turns LEFT (CCW, +yaw); ``yaw_cmd < 0`` turns
    RIGHT (CW). ``drive=0`` -> PD stand-hold. ``yaw_cmd=0`` -> straight trot.

    # Preconditions env exposes ``_leg_qadr``, ``_act_dofs``, ``_q0``, ``pd_kp``,
    ``pd_kd``, ``ctrl_range``, ``frame_skip``, 4 legs x 3 joints (fl,fr,bl,br).
    # Postconditions ``action(env, yaw_cmd, drive)`` returns ``(n_actions,)`` in ``[-1, 1]``.
    """

    hip_amp: float = 0.7
    knee_amp: float = 0.3
    freq: float = 1.2
    steer_gain: float = 0.9
    phase: "tuple[float, float, float, float]" = _DIAG_PHASE   # per-leg gait phase (default = diagonal trot)

    def action(self, env: object, yaw_cmd: float = 0.0, drive: float = 1.0) -> np.ndarray:
        t = int(getattr(env, "_step", 0)) * int(env.frame_skip) * float(env.model.opt.timestep)
        ph = 2.0 * np.pi * self.freq * t
        q = np.asarray(env.data.qpos)[env._leg_qadr]
        qd = np.asarray(env.data.qvel)[env._act_dofs]
        target = np.asarray(env._q0)[env._leg_qadr].copy()
        for leg in range(len(target) // 3):
            base = 3 * leg
            is_left = leg in (0, 2)
            # +yaw turns LEFT -> reduce LEFT (inner) legs; -yaw turns RIGHT -> reduce RIGHT.
            inner = is_left if yaw_cmd >= 0 else (not is_left)
            side = (1.0 - self.steer_gain * abs(yaw_cmd)) if inner else 1.0
            target[base + 1] += drive * self.hip_amp * max(0.1, side) * np.sin(ph + self.phase[leg])
            target[base + 2] += drive * self.knee_amp * np.sin(ph + self.phase[leg] + np.pi)
        tau = -env.pd_kp * (q - target) - env.pd_kd * qd
        return np.clip(tau / env.ctrl_range, -1.0, 1.0).astype(np.float32)
