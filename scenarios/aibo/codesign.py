"""AIBO morphology↔controller CO-DESIGN — jointly search stance width × gait params for reach.

The stance-width lever alone is a tradeoff (wide → turns well but walks poorly). Co-design searches the
*joint* space — morphology (stance width) × controller (stride sign, hip amplitude, turn strategy) —
and scores each point by multi-position goal reach. It maps the propulsion/turning/stability Pareto
frontier and returns the best-reaching (model, gait) pair. Scripted controllers (fast, no RL).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .locomotion_gait import _DIAG_PHASE, heading_error


@dataclass(frozen=True)
class CoDesignPoint:
    """One (morphology, controller) candidate: stance width + gait stride sign & amplitude."""

    stance_width: float = 0.062
    stride_sign: float = 1.0
    hip_amp: float = 0.7
    knee_amp: float = 0.3
    steer_gain: float = 0.9
    freq: float = 1.2

    def gait_action(self, env, yaw_cmd: float, drive: float) -> np.ndarray:
        """Parameterised reduce-inner trot: ``stride_sign`` sets walk direction, ``hip_amp`` its power."""
        t = int(getattr(env, "_step", 0)) * int(env.frame_skip) * float(env.model.opt.timestep)
        ph = 2.0 * np.pi * self.freq * t
        q = np.asarray(env.data.qpos)[env._leg_qadr]
        qd = np.asarray(env.data.qvel)[env._act_dofs]
        tgt = np.asarray(env._q0)[env._leg_qadr].copy()
        for leg in range(4):
            base = 3 * leg
            is_left = leg in (0, 2)
            inner = is_left if yaw_cmd >= 0 else (not is_left)
            side = (1.0 - self.steer_gain * abs(yaw_cmd)) if inner else 1.0
            tgt[base + 1] += self.stride_sign * drive * self.hip_amp * max(0.1, side) * np.sin(
                ph + _DIAG_PHASE[leg])
            tgt[base + 2] += drive * self.knee_amp * np.sin(ph + _DIAG_PHASE[leg] + np.pi)
        tau = -env.pd_kp * (q - tgt) - env.pd_kd * qd
        return np.clip(tau / env.ctrl_range, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class AgilityProfile:
    """Cheap agility measures of a co-design point: forward propulsion + stable turn + uprightness."""

    forward: float
    turn_deg_per_1000: float
    turn_upright: float


def profile(env, point: CoDesignPoint, governor, fwd_steps: int = 600, turn_steps: int = 500,
            seed: int = 0) -> AgilityProfile:
    """Measure forward-walk displacement and turn authority (with uprightness) for ``point``."""
    from .locomotion_gait import body_yaw
    env.reset(seed=seed)
    for _ in range(80):
        env.step(governor.govern(env, point.gait_action(env, 0.0, 0.0)))
    x0 = float(env.data.xpos[env.torso, 0])
    for _ in range(fwd_steps):
        env.step(governor.govern(env, point.gait_action(env, 0.0, 1.0)))
    forward = float(env.data.xpos[env.torso, 0]) - x0
    env.reset(seed=seed)
    y0, up = body_yaw(env), 1.0
    for _ in range(turn_steps):
        env.step(governor.govern(env, point.gait_action(env, 1.0, 1.0)))
        up = min(up, float(env.data.xmat[env.torso].reshape(3, 3)[2, 2]))
    dyaw = float(np.arctan2(np.sin(body_yaw(env) - y0), np.cos(body_yaw(env) - y0)))
    return AgilityProfile(forward=round(forward, 3),
                          turn_deg_per_1000=round(np.degrees(dyaw) * 1000.0 / turn_steps, 1),
                          turn_upright=round(up, 3))


def _pursuit(env, point: CoDesignPoint, state: dict) -> np.ndarray:
    """Turn-then-walk pursuit for ``point`` (align to the goal, then walk; wide hysteresis)."""
    d, h = float(env.dist_to_goal()), float(heading_error(env))
    if d <= 0.12:
        return point.gait_action(env, 0.0, 0.0)
    if state["mode"] == "align":
        if abs(h) < 0.15:
            state["mode"] = "walk"
        return point.gait_action(env, float(np.clip(h / 0.5, -0.7, 0.7)), 1.0)
    if abs(h) > 0.5:
        state["mode"] = "align"
    return point.gait_action(env, float(np.clip(h * 1.2, -0.4, 0.4)), 1.0)


def multi_position_reach(env, point: CoDesignPoint, governor, grid, horizon: int = 1600,
                         seed0: int = 400) -> dict:
    """Turn-then-walk to each (dist, bearing) goal; return the UPRIGHT reach rate + mean min-dist."""
    reached, upright_reach, dists = 0, 0, []
    for i, (dist, bdeg) in enumerate(grid):
        env.reset(seed=seed0 + i)
        b = bdeg * np.pi / 180.0
        tx = float(env.data.xpos[env.torso, 0])
        ty = float(env.data.xpos[env.torso, 1])
        env.goal = np.array([tx + dist * np.cos(b), ty + dist * np.sin(b)], np.float32)
        env._prev_dist = env.dist_to_goal()
        state = {"mode": "align"}
        min_d, min_up = float(env.dist_to_goal()), 1.0
        for _ in range(horizon):
            env.step(governor.govern(env, _pursuit(env, point, state)))
            min_d = min(min_d, float(env.dist_to_goal()))
            min_up = min(min_up, float(env.data.xmat[env.torso].reshape(3, 3)[2, 2]))
            if float(env.dist_to_goal()) <= 0.12:
                break
        ok = min_d <= 0.12
        reached += int(ok)
        upright_reach += int(ok and min_up > 0.5)   # a VALID reach stays upright throughout
        dists.append(round(min_d, 3))
    n = len(grid)
    return {"reach_rate": reached / n, "upright_reach_rate": upright_reach / n,
            "mean_min_dist": round(float(np.mean(dists)), 3)}
