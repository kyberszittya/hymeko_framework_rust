"""CIP-AIBO-01 adapter: constructed ERS-1000 quadruped sim under the CIP-0 protocol.

SIMULATION ONLY (no physical AIBO). Depends on the frozen core ``hymeko_control``
and on the constructed ``hymeko_rl`` quadruped sim; the core imports neither.

Semi-MDP: one CIP-0 tick == one hybrid-mode option (scripted FSM). Adds genuine
YAW via ``SteeredTrotGait``. The default waypoint is on-axis so the certified
trajectory is a real approach->stop->hold; robust off-axis align is a follow-up
(one-directional yaw stability), so AIBO-4 is not claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hymeko_control.cip.authority import (
    AuthorityChannel,
    AuthorityMap,
    AuthoritySource,
)
from hymeko_control.cip.certificate import CertificateResult
from hymeko_control.cip.option import (
    AffineAuthorityDecoder,
    ExecutableOption,
    OptionEnd,
    ResponseTrace,
)
from hymeko_control.cip.physical_intent import PhysicalIntent
from hymeko_control.cip.structured_state import ControlState, StructuredStateLike
from hymeko_control.language.ir import ControlModel

from .certificate import aibo_certificate_suite
from .locomotion_gait import SteeredTrotGait, heading_error

CHAIN_A = ("STAND", "ALIGN", "WALK", "DECELERATE", "STOP", "HOLD")
_INTENT_ORDER = (
    "forward_velocity", "yaw_rate", "alignment_demand", "support_margin",
    "gait_cadence", "deceleration_demand", "stop_demand",
)


@dataclass
class AIBOCIPAdapter:
    """CIP-0 adapter over the constructed ERS-1000 quadruped (simulation)."""

    model: ControlModel
    seed: int = 0
    goal_distance: float = 0.8
    reach_target: float = 0.42        # waypoint stop tolerance (env reach_radius is smaller)
    decel_dist: float = 0.52
    align_tol: float = 0.25
    stop_speed: float = 0.06
    settle_steps: int = 150
    dwell_steps: int = 200
    max_mode_steps: int = 2500

    _env: Any = field(default=None, init=False, repr=False)
    _gait: SteeredTrotGait = field(default_factory=SteeredTrotGait, init=False, repr=False)
    _mode: str = field(default="STAND", init=False, repr=False)
    _next: str = field(default="STAND", init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv
        self._env = QuadrupedGoalEnv(base="free", task="goal",
                                     goal_distance=self.goal_distance,
                                     reach_radius=0.12, max_steps=20000)
        self._env.reset(seed=self.seed)
        self._bounds = self.model.intent_bounds()

    # -- signals --------------------------------------------------------
    def _speed(self) -> float:
        v = np.asarray(self._env.data.qvel)[:2]
        return float(np.hypot(v[0], v[1]))

    def _signals(self) -> dict[str, float]:
        return {
            "dist_to_goal": float(self._env.dist_to_goal()),
            "heading_error": float(heading_error(self._env)),
            "speed": self._speed(),
            "uprightness": float(self._env._torso_uprightness()),
            "finite": 1.0 if np.all(np.isfinite(self._env.data.qacc)) else 0.0,
        }

    def _control(self, mode: str, s: dict[str, float]) -> np.ndarray:
        if mode in ("STAND", "STOP", "HOLD"):
            return self._gait.action(self._env, yaw_cmd=0.0, drive=0.0)
        if mode == "ALIGN":
            yaw = -0.5 if s["heading_error"] < -self.align_tol else (
                -0.5 if abs(s["heading_error"]) > self.align_tol else 0.0)
            return self._gait.action(self._env, yaw_cmd=yaw, drive=0.7)
        if mode == "WALK":
            return self._gait.action(self._env, yaw_cmd=0.0, drive=1.0)
        return self._gait.action(self._env, yaw_cmd=0.0, drive=0.7)  # DECELERATE

    def _exit(self, mode: str, s: dict[str, float], steps: int) -> bool:
        if mode == "STAND":
            return steps >= self.settle_steps
        if mode == "ALIGN":
            return abs(s["heading_error"]) <= self.align_tol or steps >= self.max_mode_steps
        if mode == "WALK":
            return s["dist_to_goal"] <= self.decel_dist
        if mode == "DECELERATE":
            return s["dist_to_goal"] <= self.reach_target
        if mode == "STOP":
            return s["speed"] <= self.stop_speed
        return steps >= self.dwell_steps  # HOLD

    # -- CIP-0 protocol -------------------------------------------------
    def observe(self) -> StructuredStateLike:
        s = self._signals()
        state = ControlState(t=self._tick, phase=self._mode, signals=s,
                             contact={"upright": s["uprightness"] > 0.5})
        self._tick += 1
        return state

    def identify_mode(self, state: StructuredStateLike) -> str:
        return self._mode

    def form_intent(self, state: StructuredStateLike, task: Any) -> PhysicalIntent:
        s = self._signals()
        comps = {
            "forward_velocity": 1.0 if self._mode == "WALK" else (
                0.4 if self._mode == "DECELERATE" else 0.0),
            "yaw_rate": float(np.clip(-s["heading_error"], -1.0, 1.0)) if self._mode == "ALIGN" else 0.0,
            "alignment_demand": 1.0 if self._mode == "ALIGN" else 0.0,
            "support_margin": 1.0,
            "gait_cadence": 1.0 if self._mode in ("WALK", "ALIGN", "DECELERATE") else 0.0,
            "deceleration_demand": 1.0 if self._mode in ("DECELERATE", "STOP") else 0.0,
            "stop_demand": 1.0 if self._mode in ("STOP", "HOLD") else 0.0,
        }
        return PhysicalIntent.clipped(components=comps, bounds=self._bounds)

    def measure_authority(self, state: StructuredStateLike, mode: str) -> AuthorityMap:
        s = self._signals()
        return AuthorityMap(channels={
            "forward_authority": AuthorityChannel(
                "forward_authority", 1.0, AuthoritySource.OBSERVED,
                "measured body forward velocity under the trot"),
            "yaw_authority": AuthorityChannel(
                "yaw_authority", 1.0, AuthoritySource.OBSERVED,
                "measured yaw rate under the steered trot (differential stride)"),
            "stability_authority": AuthorityChannel(
                "stability_authority", max(0.0, s["uprightness"]), AuthoritySource.OBSERVED,
                "torso uprightness (no-fall margin)"),
        })

    def decode(self, intent: PhysicalIntent, authority: AuthorityMap) -> ExecutableOption:
        dec = AffineAuthorityDecoder(
            name="aibo_steered_trot", order=_INTENT_ORDER,
            gain=tuple(1.0 for _ in _INTENT_ORDER), mode=self._mode,
            initiation=frozenset(CHAIN_A), termination="approach_align_stop")
        opt = dec.decode(intent, authority)
        prov = dict(opt.provenance)
        prov["realizer"] = "SIMULATED ERS-1000 + scripted SteeredTrotGait (NOT hardware)"
        return ExecutableOption(opt.name, opt.mode, opt.command, opt.initiation,
                                opt.termination, prov)

    def execute(self, option: ExecutableOption) -> ResponseTrace:
        idx = CHAIN_A.index(self._mode)
        signals: list[dict[str, float]] = []
        commands: list[tuple[float, ...]] = []
        fell = False
        min_upright = 1.0
        steps = 0
        for steps in range(1, self.max_mode_steps + 1):
            s = self._signals()
            a = self._control(self._mode, s)
            self._env.step(a)
            s2 = self._signals()
            signals.append(s2)
            commands.append(tuple(float(x) for x in a))
            min_upright = min(min_upright, s2["uprightness"])
            if s2["uprightness"] < 0.4 or s2["finite"] < 1.0:
                fell = True
                break
            if self._exit(self._mode, s2, steps):
                break
        if fell:
            self._next, end = self._mode, OptionEnd.ABORTED
        elif self._mode == "HOLD":
            self._next, end = "HOLD", OptionEnd.COMPLETED
        else:
            self._next = CHAIN_A[idx + 1]
            end = OptionEnd.HANDOFF
        last = signals[-1] if signals else self._signals()
        return ResponseTrace(
            option=option.name, commands=tuple(commands), signals=tuple(signals),
            end=end,
            provenance={
                "option": option.name, "mode": self._mode, "steps": steps,
                "dist_to_goal": last["dist_to_goal"], "speed": last["speed"],
                "heading_error": last["heading_error"], "min_upright": min_upright,
                "fell": fell,
                "reached": last["dist_to_goal"] <= self.reach_target,
                "halted": last["speed"] <= self.stop_speed,
                "held": self._mode == "HOLD" and steps >= self.dwell_steps and not fell,
            },
        )

    def certify(self, state: StructuredStateLike, trace: ResponseTrace) -> CertificateResult:
        return aibo_certificate_suite().evaluate(state, trace)

    def transition(self, mode: str, certificate: CertificateResult) -> str:
        nxt = self._next
        if nxt != mode and not self.model.is_legal_transition(mode, nxt):
            nxt = mode
        self._mode = nxt
        return nxt
