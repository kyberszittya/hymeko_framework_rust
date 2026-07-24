"""REALISTIC_MOTION_CONTRACT_V1 — physical-plausibility limits + metrics + a hard acceptance gate for every task.

Motivation (2026-07-24 audit): the coin arm reached 27 rad/s (real robots run 1–3) and slid the coin at 1.5 m/s — the
video's "teleport" was genuine unrealistic motion, not a playback artifact (playback is 3.33× slow-mo). Armature/damping
alone did not fix it; the PRIMARY layer must be a velocity/slew-rate LIMIT on the commands, reward penalties second, and a
LOW-TERMINAL-VELOCITY certificate so a policy can't "throw" the object through the target for a momentary success.

This module is task-independent: a config of limits, a per-step metrics accumulator, the slew-rate command limiter
(position control), thresholded (ReLU(x−1)) penalty terms, a velocity-aware terminal predicate, and the hard gate
`assert_realistic_motion`. Every recorder/experiment reports the metrics; new position-controlled tasks (6D-1 reach,
pick-place, AIBO) apply the limiter from the start.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MotionLimits:
    """Hard physical limits + safe bands. Hard limits gate acceptance; safe bands set where reward penalties begin.
    Defaults are conservative articulated-arm values (rad, rad/s, m, s)."""

    # HARD gate = the clean, meaningful, controllable signals: joint velocity, EE speed, terminal velocity. Instantaneous
    # joint ACCELERATION is servo-/contact-transient-dominated (a realistic 1.9 rad/s arm still shows ~1600 rad/s² qacc
    # spikes at target changes and contacts), so acceleration is a REPORTED diagnostic + a SOFT reward penalty on the
    # COMMANDED trajectory, not a hard gate — otherwise the gate false-fails physically-fine motion.
    joint_vel_hard: float = 3.0        # rad/s — above this the motion is unphysical (gate fails)
    ee_speed_hard: float = 1.5         # m/s
    joint_vel_safe: float = 2.0        # rad/s — reward penalty begins above this
    ee_speed_safe: float = 1.0         # m/s
    joint_acc_safe: float = 40.0       # rad/s² — soft-penalty band (commanded accel), NOT a hard gate
    joint_acc_report: float = 3000.0   # rad/s² — informational ceiling for the report only (transient-inclusive)
    terminal_joint_vel: float = 0.25   # rad/s — a certified terminal state must be near-stationary
    terminal_ee_speed: float = 0.05    # m/s
    settle_steps: int = 5              # consecutive low-velocity steps required for a stable certificate


@dataclass
class MotionMetrics:
    """Per-step accumulator → the mandatory per-task motion report. Feed each step's joint velocity/acceleration, EE
    speed, and control vector; `finalize` returns peaks, means, jerk, integrated effort, terminal velocity."""

    _jv: list = field(default_factory=list)
    _ja: list = field(default_factory=list)
    _ee: list = field(default_factory=list)
    _u: list = field(default_factory=list)

    def add(self, joint_vel, joint_acc, ee_speed, ctrl) -> None:
        self._jv.append(float(np.max(np.abs(joint_vel))))
        self._ja.append(float(np.max(np.abs(joint_acc))))
        self._ee.append(float(ee_speed))
        self._u.append(np.asarray(ctrl, np.float64).copy())

    def finalize(self, *, dt: float = 0.01, settle_steps: int = 5) -> dict:
        jv, ja, ee = np.asarray(self._jv), np.asarray(self._ja), np.asarray(self._ee)
        u = np.asarray(self._u) if self._u else np.zeros((max(1, len(jv)), 1))
        du = np.abs(np.diff(u, axis=0)) if len(u) > 1 else np.zeros((1, u.shape[1]))
        jerk = np.abs(np.diff(ja)) / dt if len(ja) > 1 else np.zeros(1)
        # terminal settling: count trailing steps whose EE speed is below its own final median band
        n = len(jv)
        return {
            "n_steps": int(n), "sim_time_s": round(n * dt, 3),
            "peak_joint_vel": round(float(jv.max()) if n else 0.0, 3),
            "mean_joint_vel": round(float(jv.mean()) if n else 0.0, 3),
            "peak_joint_acc": round(float(ja.max()) if n else 0.0, 1),
            "peak_ee_speed": round(float(ee.max()) if n else 0.0, 3),
            "mean_ee_speed": round(float(ee.mean()) if n else 0.0, 3),
            "peak_jerk": round(float(jerk.max()), 1),
            "max_ctrl_step_delta": round(float(du.max()), 4),
            "integrated_ctrl_effort": round(float(np.abs(u).sum() * dt), 3),
            "terminal_joint_vel": round(float(jv[-1]) if n else 0.0, 3),
            "terminal_ee_speed": round(float(ee[-1]) if n else 0.0, 3),
        }


def slew_limited_position(q_desired_raw, q_prev, *, joint_vel_limit: float, dt: float) -> np.ndarray:
    """Velocity-limited POSITION command: the commanded joint target cannot move faster than ``joint_vel_limit`` — the
    primary anti-teleport layer (removes single-step target jumps that a servo yanks the arm toward). # Postconditions
    ``‖result − q_prev‖∞ ≤ joint_vel_limit·dt``."""
    q_desired_raw = np.asarray(q_desired_raw, np.float64)
    q_prev = np.asarray(q_prev, np.float64)
    step = joint_vel_limit * dt
    return q_prev + np.clip(q_desired_raw - q_prev, -step, step)


def _relu(x):
    return np.maximum(x, 0.0)


def motion_penalties(joint_vel, joint_acc, ee_speed, ctrl, ctrl_prev, limits: MotionLimits) -> dict:
    """Thresholded (ReLU(x/safe − 1)²) penalties — normal speed is NOT penalised, only the excess above the safe band —
    plus a smoothness (Δu²) term. The reward's SECOND layer (the limiter is the first)."""
    jv = np.abs(np.asarray(joint_vel, np.float64))
    p_speed = float(np.mean(_relu(jv / limits.joint_vel_safe - 1.0) ** 2))
    p_ee = float(_relu(float(ee_speed) / limits.ee_speed_safe - 1.0) ** 2)
    p_acc = float(np.mean(_relu(np.abs(np.asarray(joint_acc, np.float64)) / limits.joint_acc_safe - 1.0) ** 2))
    du = np.asarray(ctrl, np.float64) - np.asarray(ctrl_prev, np.float64)
    p_smooth = float(np.mean(du ** 2))
    return {"speed": p_speed, "ee_speed": p_ee, "accel": p_acc, "smooth": p_smooth}


def terminal_velocity_certified(joint_vel, ee_speed, limits: MotionLimits) -> bool:
    """A task success is only physical if the terminal state is near-stationary (no throw-through). Combine with the
    task's pose/footprint predicate: certified = task_success AND terminal_velocity_certified(...)."""
    return bool(np.max(np.abs(joint_vel)) < limits.terminal_joint_vel and float(ee_speed) < limits.terminal_ee_speed)


def assert_realistic_motion(metrics: dict, limits: MotionLimits, *, label: str = "") -> dict:
    """REALISTIC_MOTION_CONTRACT_V1 hard gate — raises AssertionError when a rollout's peak joint VELOCITY or EE SPEED
    exceeds the hard limits (the clean, controllable signals). Instantaneous acceleration is transient-dominated and is
    reported, not gated. Returns a pass/margin dict on success. Every task's acceptance calls this; a run that trips it
    is not a valid demonstration or measurement."""
    checks = {"joint_vel": (metrics["peak_joint_vel"], limits.joint_vel_hard),
              "ee_speed": (metrics["peak_ee_speed"], limits.ee_speed_hard)}
    violations = {k: (v, lim) for k, (v, lim) in checks.items() if v > lim}
    if violations:
        raise AssertionError(f"{label}: REALISTIC_MOTION_CONTRACT violated — " +
                             "; ".join(f"{k} {v:.2f} > hard {lim}" for k, (v, lim) in violations.items()))
    return {"ok": True, "margins": {k: round(lim - v, 3) for k, (v, lim) in checks.items()}}


def motion_report(metrics: dict, limits: MotionLimits, *, label: str = "") -> dict:
    """The mandatory per-task motion report (success/time/peaks/effort/settling) + the hard-limit verdict (non-raising).
    Hard verdict is on VELOCITY + EE speed only; acceleration/jerk are diagnostics."""
    within = (metrics["peak_joint_vel"] <= limits.joint_vel_hard and metrics["peak_ee_speed"] <= limits.ee_speed_hard)
    return {"label": label, **metrics, "within_hard_limits": bool(within),
            "hard_limits": {"joint_vel": limits.joint_vel_hard, "ee_speed": limits.ee_speed_hard},
            "accel_is_diagnostic_only": True}
