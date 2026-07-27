"""Spring-hop-to-goal — the spring-legged AIBO hops forward to a designated goal, staying upright.

A goal is placed a forward distance ahead; the AIBO reaches it with repeated spring-powered hops.
Each hop cycle:

  1. **LOAD**    — crouch the knee springs (store ½·K·θ²).
  2. **LAUNCH**  — release (actuator off → the springs fire = a passive vertical lift) while a
     **motor-limited** hip-flex push (≤ ``hip_tau_cap`` N·m) drives the body forward; the push
     magnitude is scaled by the remaining distance, so it decelerates into the goal.
  3. **RESTABILIZE** — PD-hold the full standing posture (all 12 leg joints → q0) so the body
     returns to standing height before the next hop.

Honesty. The **vertical lift is the passive spring** (fast knee, not a commanded motor — legitimate
series elasticity, unlike the retracted 27 rad/s exploit). The **forward impulse is a motor-limited
hip push** (≤ a realistic ~5 N·m), so the horizontal drive is a real actuator, not an injected
velocity. The body stays upright every hop and re-settles to standing height between hops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .spring_leg import LEGS

_LEG_JOINTS: tuple[str, ...] = tuple(f"{part}_{leg}" for leg in LEGS
                                     for part in ("hip_abduct", "hip_flex", "knee"))
StepHook = Callable[[mujoco.MjData, str, int], None]


@dataclass(frozen=True)
class HopToGoalResult:
    """Outcome of a spring-hop-to-goal run."""

    reached: bool
    n_hops: int
    forward: float
    min_upright: float
    hop_x: tuple[float, ...]  # torso forward position at the end of each hop (for plotting)

    @property
    def fell(self) -> bool:
        return self.min_upright < 0.5


@dataclass
class SpringHopGait:
    """Drive a spring-legged model to a forward goal by repeated upright spring hops.

    # Preconditions
    ``model`` is a spring-legged quadruped (see :func:`spring_leg.build_spring_legged`) with a free
    ``base`` joint and ``hip_abduct/hip_flex/knee_{leg}`` + ``paw_{leg}`` bodies. ``goal_distance>0``
    (a forward goal), ``hip_tau_cap`` ≤ the realistic motor torque. # Invariant: only the hip-flex
    push (bounded by ``hip_tau_cap``) and the posture PD apply actuator torque; the knee launch is
    passive spring.
    """

    model: mujoco.MjModel
    goal_distance: float = 0.6
    reach_radius: float = 0.12
    crouch: float = -0.7
    hip_tau_cap: float = 5.0
    motor_tau_cap: float = 8.0   # realistic AIBO-class leg motor — ALL actuator torque is clamped here
    max_hops: int = 14
    launch_steps: int = 110
    push_steps: int = 60
    settle_steps: int = 360
    stand_settle: int = 250
    posture_kp: float = 25.0
    posture_kv: float = 1.5
    _qadr: dict = field(default_factory=dict, init=False, repr=False)
    _vadr: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        jid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self._qadr = {jn: int(self.model.jnt_qposadr[jid(jn)]) for jn in _LEG_JOINTS}
        self._vadr = {jn: int(self.model.jnt_dofadr[jid(jn)]) for jn in _LEG_JOINTS}
        if self.goal_distance <= 0.0:
            raise ValueError("goal_distance must be > 0 (a forward goal)")
        if self.hip_tau_cap > self.motor_tau_cap:
            raise ValueError("hip_tau_cap must not exceed the motor torque cap")

    def _torso(self) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    @staticmethod
    def _upright(d: mujoco.MjData, torso: int) -> float:
        return float(d.xmat[torso].reshape(3, 3)[2, 2])

    def _hold_posture(self, d: mujoco.MjData, q0: dict[str, float]) -> None:
        for jn in _LEG_JOINTS:
            tau = self.posture_kp * (q0[jn] - float(d.qpos[self._qadr[jn]])) \
                - self.posture_kv * float(d.qvel[self._vadr[jn]])
            d.qfrc_applied[self._vadr[jn]] = float(np.clip(tau, -self.motor_tau_cap,
                                                           self.motor_tau_cap))

    def run(self, on_step: StepHook | None = None) -> HopToGoalResult:
        """Hop to the goal; call ``on_step(data, phase, hop_idx)`` each sim step if provided.

        Postcondition: returns whether the goal was reached, the hop count, forward displacement,
        and the minimum torso uprightness over the whole run (``fell`` iff it dropped below 0.5).
        """
        m = self.model
        d = mujoco.MjData(m)
        torso = self._torso()
        mujoco.mj_forward(m, d)
        for _ in range(self.stand_settle):
            mujoco.mj_step(m, d)
            if on_step:
                on_step(d, "stand", 0)
        q0 = {jn: float(d.qpos[self._qadr[jn]]) for jn in _LEG_JOINTS}
        x_start = float(d.xpos[torso, 0])
        goal_x = x_start + self.goal_distance
        min_up, hops, hop_x = 1.0, 0, []
        while (goal_x - float(d.xpos[torso, 0])) > self.reach_radius and hops < self.max_hops:
            hops += 1
            remaining = goal_x - float(d.xpos[torso, 0])
            push = float(np.clip(6.0 * np.tanh(remaining), -self.hip_tau_cap, self.hip_tau_cap))
            for leg in LEGS:                                     # LOAD the springs
                d.qpos[self._qadr[f"knee_{leg}"]] = self.crouch
            mujoco.mj_forward(m, d)
            for k in range(self.launch_steps):                  # LAUNCH: passive spring + hip push
                d.qfrc_applied[:] = 0.0
                if k < self.push_steps:
                    for leg in LEGS:
                        d.qfrc_applied[self._vadr[f"hip_flex_{leg}"]] = push
                mujoco.mj_step(m, d)
                min_up = min(min_up, self._upright(d, torso))
                if on_step:
                    on_step(d, "launch", hops)
            for _ in range(self.settle_steps):                  # RESTABILIZE to standing posture
                self._hold_posture(d, q0)
                mujoco.mj_step(m, d)
                min_up = min(min_up, self._upright(d, torso))
                if on_step:
                    on_step(d, "settle", hops)
            hop_x.append(float(d.xpos[torso, 0]) - x_start)
        reached = (goal_x - float(d.xpos[torso, 0])) <= self.reach_radius
        return HopToGoalResult(reached=reached, n_hops=hops,
                               forward=float(d.xpos[torso, 0]) - x_start,
                               min_upright=min_up, hop_x=tuple(hop_x))
