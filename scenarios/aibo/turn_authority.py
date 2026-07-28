"""Turn-authority diagnostic — why the AIBO can't reach off-axis goals (the yaw/stability wall).

After the residual-RL negatives (a bounded residual over the trot can't extend multi-position reach),
this characterises the *richer turning primitive* directly. `AgileTurnGait` generalises the trot's
"reduce-inner" skid-steer to a **reverse-inner in-place spin** (inner legs can stride backward), the
strongest skid-steer turn available. The measurement is a **wall**:

  * a **stable** turn (kept upright) yields only ~20°/1000 steps — far too slow to face a ±40° goal in
    a bounded horizon (facing 40° needs ~2000 steps of pure turning);
  * a **fast** turn (full reverse-inner spin) **tips the robot over** (min uprightness → negative) —
    the same instability the campaign found with abduction turning.

So off-axis goal-reaching is bottlenecked by the AIBO's **turn/stability tradeoff** — a kinematic/
dynamic property of the skid-steer on this model, not a controller or learning deficit. This module
is the reproducible diagnostic behind that finding; it is **not** a shipped locomotion controller
(the fast turn is an unstable exploit — used here only to measure the tipping onset).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .locomotion_gait import _DIAG_PHASE, body_yaw


@dataclass(frozen=True)
class AgileTurnGait:
    """Skid-steer trot whose inner legs can REVERSE — a continuum walk → pivot → in-place spin.

    ``turn ∈ [−1, 1]``: sign = direction, magnitude = turn rate; the inner-side stride scale is
    ``1 − 2·|turn|`` (0 at |turn|=0.5 → pivot; −1 at |turn|=1 → full reverse → spin). ``drive`` sets
    the forward/stride amplitude. # Preconditions: env exposes the ``SteeredTrotGait`` interface.
    # Warning: |turn| near 1 is DYNAMICALLY UNSTABLE (tips) — see :func:`measure_turn`.
    """

    hip_amp: float = 0.7
    knee_amp: float = 0.3
    freq: float = 1.2

    def action(self, env: object, turn: float = 0.0, drive: float = 1.0) -> np.ndarray:
        t = int(getattr(env, "_step", 0)) * int(env.frame_skip) * float(env.model.opt.timestep)
        ph = 2.0 * np.pi * self.freq * t
        q = np.asarray(env.data.qpos)[env._leg_qadr]
        qd = np.asarray(env.data.qvel)[env._act_dofs]
        target = np.asarray(env._q0)[env._leg_qadr].copy()
        for leg in range(len(target) // 3):
            base = 3 * leg
            is_left = leg in (0, 2)
            inner = is_left if turn >= 0 else (not is_left)
            side = (1.0 - 2.0 * abs(turn)) if inner else 1.0     # inner reverses at |turn|>0.5
            target[base + 1] += drive * self.hip_amp * side * np.sin(ph + _DIAG_PHASE[leg])
            target[base + 2] += drive * self.knee_amp * np.sin(ph + _DIAG_PHASE[leg] + np.pi)
        tau = -env.pd_kp * (q - target) - env.pd_kd * qd
        return np.clip(tau / env.ctrl_range, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class TurnMeasurement:
    """Yaw authority of one turn command: rate (deg per 1000 steps), stability, whether it tipped."""

    turn: float
    yaw_deg_per_1000: float
    min_upright: float

    @property
    def tipped(self) -> bool:
        return self.min_upright < 0.5


def measure_turn(env, gait: AgileTurnGait, governor, turn: float, drive: float = 1.0,
                 steps: int = 500, seed: int = 0) -> TurnMeasurement:
    """Command a constant ``turn`` for ``steps`` and measure the yaw rate + minimum uprightness.

    Postcondition: ``yaw_deg_per_1000`` is the signed world-yaw change scaled to a 1000-step rate;
    ``min_upright`` is the torso up-axis z over the run (``tipped`` iff it dropped below 0.5).
    """
    env.reset(seed=seed)
    y0 = body_yaw(env)
    min_up = 1.0
    for _ in range(steps):
        env.step(governor.govern(env, gait.action(env, turn=turn, drive=drive)))
        min_up = min(min_up, float(env.data.xmat[env.torso].reshape(3, 3)[2, 2]))
    dyaw = float(np.arctan2(np.sin(body_yaw(env) - y0), np.cos(body_yaw(env) - y0)))
    return TurnMeasurement(turn=turn, yaw_deg_per_1000=float(np.degrees(dyaw)) * 1000.0 / steps,
                           min_upright=round(min_up, 3))


def stable_turn_ceiling(env, gait: AgileTurnGait, governor, drive: float = 1.0,
                        turns=(0.3, 0.5, 0.7, 0.9, 1.0), steps: int = 500) -> dict:
    """Sweep turn magnitude; return the fastest STABLE (upright) yaw rate and the tipping onset.

    The AIBO turn/stability wall: the stable rate is small; beyond a threshold the turn tips.
    """
    rows = [measure_turn(env, gait, governor, t, drive=drive, steps=steps) for t in turns]
    stable = [r for r in rows if not r.tipped]
    fastest_stable = max((abs(r.yaw_deg_per_1000) for r in stable), default=0.0)
    tip_onset = min((r.turn for r in rows if r.tipped), default=None)
    return {
        "fastest_stable_deg_per_1000": round(fastest_stable, 1),
        "tipping_onset_turn": tip_onset,
        "rows": [(r.turn, round(r.yaw_deg_per_1000, 1), r.min_upright, r.tipped) for r in rows],
    }
