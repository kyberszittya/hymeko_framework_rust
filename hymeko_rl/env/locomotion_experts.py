"""Scripted locomotion demonstrators (the ``TeacherRegistry`` side of the transfer spine) for the
fast-dynamics substrates. Two Strategy classes, both mapping a live env to a normalised action in
``[-1, 1]`` — the BC anchor / DAgger teacher the RL learners refine from:

  * :class:`CpgGait`      — a central-pattern-generator (phase-offset sinusoids) open-loop gait for the
                            legged runners (cheetah / humanoid). A *reference* gait, not an optimum; the
                            RL policy improves on it (per the imitation-bounded coin-toss/pick-place lesson).
  * :class:`DiffDrivePursuit` — a closed-loop pure-pursuit velocity controller for the diff-drive vehicle:
                            steer toward the active waypoint, drive at a target speed. A genuine, reliable
                            waypoint follower (the vehicle's ``scripted_controller`` ceiling).

Neither is imported into the RL loop's hot path in bulk; each is queried per-step by DAgger/eval. Kept out
of ``locomotion_env`` (§6.5 #4: env vs expert are separable concerns)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CpgGait:
    """Open-loop central-pattern-generator gait: ``a_i(t) = amp_i · sin(2π·freq·t·dt + phase_i) + bias_i``,
    clipped to ``[-1, 1]``. A phase-offset sinusoid per actuated joint — the classic legged locomotion
    reference. ``amp``/``phase``/``bias`` are per-joint (length ``n_actions``); scalars broadcast.

    # Preconditions ``n_actions >= 1``; ``amp``/``phase``/``bias``, if arrays, have length ``n_actions``.
    # Postconditions ``action(env)`` returns shape ``(n_actions,)`` in ``[-1, 1]``; deterministic in the
    env's step counter (so BC/DAgger see a repeatable teacher)."""

    n_actions: int
    freq: float = 1.5                       # gait cycles per second
    amp: np.ndarray = field(default_factory=lambda: np.array(0.6))
    phase: np.ndarray = field(default_factory=lambda: np.array(0.0))
    bias: np.ndarray = field(default_factory=lambda: np.array(0.0))

    @classmethod
    def alternating(cls, n_actions: int, *, n_front_offset: int | None = None, freq: float = 1.5,
                    amp: float = 0.7) -> "CpgGait":
        """A bounding gait: the front half of the joints runs in anti-phase (``+π``) to the back half, with a
        within-leg knee/ankle lag (``π/2`` per joint down the chain). A sane default forward-driving reference
        for a symmetric two-leg-group runner (cheetah's back/front, the humanoid's legs). ``n_front_offset``
        splits back vs front actuators (defaults to the midpoint)."""
        split = n_actions // 2 if n_front_offset is None else int(n_front_offset)
        phase = np.zeros(n_actions, dtype=np.float64)
        for i in range(n_actions):
            leg_group_phase = np.pi if i >= split else 0.0            # front anti-phase to back
            within = (i % max(1, split)) * (np.pi / 2.0)             # knee/ankle lag down the chain
            phase[i] = leg_group_phase + within
        return cls(n_actions=n_actions, freq=float(freq),
                   amp=np.full(n_actions, float(amp)), phase=phase, bias=np.zeros(n_actions))

    def action(self, env: object) -> np.ndarray:
        """Sinusoidal action at the env's current sim time (``env._step · frame_skip · dt``)."""
        step = int(getattr(env, "_step", 0))
        dt = float(getattr(env, "_dt", 0.01)) * int(getattr(env, "frame_skip", 1))
        t = step * dt
        amp = np.broadcast_to(np.asarray(self.amp, dtype=np.float64), (self.n_actions,))
        phase = np.broadcast_to(np.asarray(self.phase, dtype=np.float64), (self.n_actions,))
        bias = np.broadcast_to(np.asarray(self.bias, dtype=np.float64), (self.n_actions,))
        a = amp * np.sin(2.0 * np.pi * self.freq * t + phase) + bias
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def __call__(self, env: object) -> np.ndarray:
        return self.action(env)


@dataclass(frozen=True)
class AckermannPursuit:
    """Pure-pursuit controller for an Ackermann car (F1TENTH): steer the front wheels toward the active
    waypoint, drive the rear wheels at a throttle. Returns the F1TENTH ``[throttle, steer]`` action in
    ``[-1, 1]²`` (throttle→rear speed, steer→front angle). ``ramp_steps`` softens the standing start.

    # Preconditions the env exposes ``heading_error()``. # Postconditions shape ``(2,)`` in ``[-1, 1]``."""

    throttle: float = 0.85
    steer_gain: float = 1.5                 # heading error (rad) → normalised steer command
    ramp_steps: int = 0

    def action(self, env: object) -> np.ndarray:
        err = float(env.heading_error())                 # +: waypoint CCW (to +y / left)
        ramp = 1.0 if self.ramp_steps <= 0 else min(1.0, (int(getattr(env, "_step", 0)) + 1) / self.ramp_steps)
        throttle = self.throttle * ramp
        steer = float(np.clip(self.steer_gain * err, -1.0, 1.0))
        return np.array([throttle, steer], dtype=np.float32)

    def __call__(self, env: object) -> np.ndarray:
        return self.action(env)


@dataclass(frozen=True)
class QuadrupedTrotGait:
    """Closed-loop **PD trot** for a quadruped (e.g. the Aibo): PD-tracks a diagonal-trot joint trajectory
    (offsets the standing pose ``q0``), producing a slow, stable forward walk. Diagonal pairs move anti-phase
    ({fl,br} vs {fr,bl}); the knee runs anti-phase to the hip. PD tracking (not open-loop torque) gives the
    closed-loop balance a bare CPG lacks — but it is still a *weak* demonstrator (scripted gaits shuffle;
    RL/DAgger is the lever for real speed, exactly as for Aibo standing).

    Reads the env's leg layout (``_leg_qadr`` grouped per leg as abduct/flex/knee), rest pose (``_q0``), PD
    gains (``pd_kp``/``pd_kd``) and ``ctrl_range``. # Preconditions ``n_actions`` divisible by 3 (abduct/flex/
    knee per leg), 4 legs ordered fl,fr,bl,br. # Postconditions returns ``(n_actions,)`` torques in ``[-1, 1]``."""

    hip_amp: float = 0.7
    knee_amp: float = 0.3
    knee_phase: float = float(np.pi)
    freq: float = 1.2
    _diag_phase: tuple = (0.0, float(np.pi), float(np.pi), 0.0)   # fl, fr, bl, br (diagonal trot)

    def action(self, env: object) -> np.ndarray:
        t = int(getattr(env, "_step", 0)) * int(env.frame_skip) * float(env.model.opt.timestep)
        ph = 2.0 * np.pi * self.freq * t
        q = np.asarray(env.data.qpos)[env._leg_qadr]
        qd = np.asarray(env.data.qvel)[env._act_dofs]
        target = np.asarray(env._q0)[env._leg_qadr].copy()
        for leg in range(len(target) // 3):
            base = 3 * leg
            lp = self._diag_phase[leg % len(self._diag_phase)]
            target[base + 1] += self.hip_amp * np.sin(ph + lp)                       # hip_flex swing
            target[base + 2] += self.knee_amp * np.sin(ph + lp + self.knee_phase)    # knee (anti-phase)
        tau = -env.pd_kp * (q - target) - env.pd_kd * qd
        return np.clip(tau / env.ctrl_range, -1.0, 1.0).astype(np.float32)

    def __call__(self, env: object) -> np.ndarray:
        return self.action(env)


@dataclass(frozen=True)
class DiffDrivePursuit:
    """Closed-loop pure-pursuit controller for a 4-wheel diff-drive vehicle. Steers toward the active
    waypoint (heading error → differential wheel command) at a target forward drive, mapped to the four
    wheels via the left/right split. ``left_idx``/``right_idx`` index the 4-vector action
    (``[fr, fl, rr, rl]`` order from the plant: right={fr,rr}, left={fl,rl}).

    # Preconditions the env exposes ``heading_error()`` (signed radians, +left) and ``dist_to_goal()``.
    # Postconditions ``action(env)`` returns shape ``(4,)`` in ``[-1, 1]``; drives forward and turns toward
    the waypoint; eases off near it."""

    drive: float = 0.95                     # nominal forward wheel command
    steer_gain: float = 0.7                 # rad → differential command
    slowdown: float = 1.2                   # m: linearly ease off drive within this range of the waypoint
    ramp_steps: int = 0                     # >0: ramp throttle 0→1 over this many steps (avoids high-speed wheelspin)
    # PHYSICAL (not nominal) wheel split for the robot_4wh geometry. The +heading-error convention is a CCW
    # turn toward a +y (left) waypoint; a CCW turn needs the −y-side wheels {fl,rl}={1,3} FAST and the +y-side
    # wheels {fr,rr}={0,2} SLOW. (The plant labels fr/rr "right", but +y is its right side, so the physical
    # turn-CCW-fast group is the nominally-"left" wheels — indices decided by the geometry, not the names.)
    turn_fast_idx: tuple = (1, 3)           # speed up on +err (−y side) → turns CCW toward a +y waypoint
    turn_slow_idx: tuple = (0, 2)           # slow down on +err (+y side)

    def action(self, env: object) -> np.ndarray:
        err = float(env.heading_error())                 # +: waypoint CCW (to +y) of the chassis heading
        # Forward drive scales with (a) alignment — ZERO past 90° so it turns in place, not circles wide — and
        # (b) proximity — linearly eased within `slowdown` m so it does not overshoot into a hard re-acquire.
        # Differential turn steers toward the waypoint. A textbook pure-pursuit, not per-track tuning.
        near = float(np.clip(env.dist_to_goal() / self.slowdown, 0.25, 1.0))
        ramp = 1.0 if self.ramp_steps <= 0 else min(1.0, (int(getattr(env, "_step", 0)) + 1) / self.ramp_steps)
        drive = self.drive * ramp * near * max(0.0, float(np.cos(np.clip(err, -np.pi, np.pi))))
        turn = float(np.clip(self.steer_gain * err, -1.0, 1.0))
        a = np.zeros(4, dtype=np.float64)
        for i in self.turn_fast_idx:
            a[i] = drive + turn
        for i in self.turn_slow_idx:
            a[i] = drive - turn
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def __call__(self, env: object) -> np.ndarray:
        return self.action(env)
