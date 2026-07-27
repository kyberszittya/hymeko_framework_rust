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
    """Diagonal PD-trot with a yaw command via differential L/R stride amplitude.

    # Preconditions env exposes ``_leg_qadr``, ``_act_dofs``, ``_q0``, ``pd_kp``,
    ``pd_kd``, ``ctrl_range``, ``frame_skip``, 4 legs x 3 joints (abduct/flex/knee).
    # Postconditions ``action(env, yaw_cmd, drive)`` returns ``(n_actions,)`` in
    ``[-1, 1]``. ``drive=0`` -> PD stand-hold; ``yaw_cmd<0`` turns the stable way.
    """

    hip_amp: float = 0.7
    knee_amp: float = 0.3
    freq: float = 1.2
    steer_gain: float = 0.9
    abd_gain: float = 0.35

    def action(self, env: object, yaw_cmd: float = 0.0, drive: float = 1.0) -> np.ndarray:
        t = int(getattr(env, "_step", 0)) * int(env.frame_skip) * float(env.model.opt.timestep)
        ph = 2.0 * np.pi * self.freq * t
        q = np.asarray(env.data.qpos)[env._leg_qadr]
        qd = np.asarray(env.data.qvel)[env._act_dofs]
        target = np.asarray(env._q0)[env._leg_qadr].copy()
        for leg in range(len(target) // 3):
            base = 3 * leg
            side = 1.0 - self.steer_gain * yaw_cmd if leg in (0, 2) \
                else 1.0 + self.steer_gain * yaw_cmd
            target[base + 0] += (-self.abd_gain * yaw_cmd if leg in (0, 2)
                                 else self.abd_gain * yaw_cmd)
            target[base + 1] += drive * self.hip_amp * max(0.05, side) * np.sin(ph + _DIAG_PHASE[leg])
            target[base + 2] += drive * self.knee_amp * np.sin(ph + _DIAG_PHASE[leg] + np.pi)
        tau = -env.pd_kp * (q - target) - env.pd_kd * qd
        return np.clip(tau / env.ctrl_range, -1.0, 1.0).astype(np.float32)
