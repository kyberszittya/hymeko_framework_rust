"""CIP-PNP-01 adapter: PickPlaceEnv + scripted expert v3 under the CIP-0 protocol.

Dependency direction is one-way: this scenario module depends on the frozen core
``hymeko_control`` and on ``hymeko_rl`` (the embodiment), never the reverse.

Semi-MDP mapping: one CIP-0 tick == one hybrid-mode OPTION. ``execute`` runs the
embodiment's realizer (the scripted expert --- the embodiment-specific authority
decoder) until the classified mode advances, the object is dropped, or the
episode ends. ``decode`` is a pure, deterministic function of (intent, authority);
the low-level joint action is realized inside ``execute`` (v0 uses the scripted
expert; a learned decoder is future RL work).
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

from .certificate import pnp_certificate_suite

# Declared mode chain (matches cip_pnp_01.hymeko.yaml).
CHAIN = ("APPROACH", "ACQUIRE", "GRASP", "LIFT", "CARRY", "PLACE", "RELEASE", "SETTLE")
_INTENT_ORDER = (
    "approach_velocity", "acquisition_demand", "grasp_force", "lift_clearance",
    "transport_velocity", "target_correction", "placement_force", "release_demand",
)


def _mk_env(seed: int, expert_version: int):
    from hymeko_rl.viz.render_pick_place import fanuc_pick_env

    env = fanuc_pick_env(
        require_settle=True, target_bin=True, expert_version=expert_version
    )
    env.reset(seed=seed)
    return env


@dataclass
class PickPlaceCIPAdapter:
    """CIP-0 adapter over the arm+gripper pick-place embodiment."""

    model: ControlModel
    seed: int = 1
    expert_version: int = 3
    lift_thresh: float = 0.035
    place_radius: float = 0.075
    max_option_steps: int = 640

    _env: Any = field(default=None, init=False, repr=False)
    _info: dict = field(default_factory=dict, init=False, repr=False)
    _tick: int = field(default=0, init=False, repr=False)
    _mode: str = field(default="APPROACH", init=False, repr=False)
    _end_mode: str = field(default="APPROACH", init=False, repr=False)
    _was_lifted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._env = _mk_env(self.seed, self.expert_version)
        self._info = dict(self._env_last_info())
        self._bounds = self.model.intent_bounds()

    def _env_last_info(self) -> dict:
        # reset() already ran; pull a first info by reading env geometry cheaply.
        info = getattr(self._env, "_last_info", None)
        return dict(info) if isinstance(info, dict) else {}

    # -- signal extraction / mode classification ------------------------
    @staticmethod
    def _signals(info: dict) -> dict[str, float]:
        def f(key: str, default: float = 0.0) -> float:
            v = info.get(key, default)
            return float(v) if v is not None else default

        def b(key: str) -> float:
            return 1.0 if info.get(key) else 0.0

        return {
            "both_contact": b("both_contact"),
            "lifted": f("lifted"),
            "obj_to_target": f("obj_to_target", 1.0),
            "placed_stable": b("placed_stable"),
            "settled": b("settled"),
            "reached": b("reached"),
            "on_ground": b("on_ground"),
            "delivered": b("delivered"),
            "death": b("death"),
        }

    def _classify(self, info: dict) -> str:
        s = self._signals(info)
        if s["placed_stable"] or s["settled"]:
            return "SETTLE"
        if s["both_contact"]:
            if s["lifted"] >= self.lift_thresh:
                return "PLACE" if s["obj_to_target"] < self.place_radius else "CARRY"
            if s["lifted"] >= 0.008:
                return "LIFT"
            return "GRASP"
        if s["obj_to_target"] < self.place_radius:
            return "RELEASE"
        return "ACQUIRE" if s["reached"] else "APPROACH"

    # -- CIP-0 protocol (8 methods) -------------------------------------
    def observe(self) -> StructuredStateLike:
        state = ControlState(
            t=self._tick, phase=self._mode,
            signals=self._signals(self._info),
            contact={"bilateral": bool(self._info.get("both_contact"))},
        )
        self._tick += 1
        return state

    def identify_mode(self, state: StructuredStateLike) -> str:
        self._mode = self._classify(self._info)
        return self._mode

    def form_intent(self, state: StructuredStateLike, task: Any) -> PhysicalIntent:
        s = self._signals(self._info)
        comps = {
            "approach_velocity": 1.0 - s["reached"],
            "acquisition_demand": s["reached"],
            "grasp_force": s["both_contact"],
            "lift_clearance": max(self.lift_thresh - s["lifted"], 0.0),
            "transport_velocity": min(s["obj_to_target"], 1.0),
            "target_correction": 0.0,
            "placement_force": 1.0 if self._mode == "PLACE" else 0.0,
            "release_demand": 1.0 if self._mode in ("RELEASE", "SETTLE") else 0.0,
        }
        return PhysicalIntent.clipped(components=comps, bounds=self._bounds)

    def measure_authority(self, state: StructuredStateLike, mode: str) -> AuthorityMap:
        s = self._signals(self._info)
        return AuthorityMap(channels={
            "reach_authority": AuthorityChannel(
                "reach_authority", 1.0, AuthoritySource.OBSERVED,
                "IK reachability of the pre-grasp pose (fanuc annulus)"),
            "grip_authority": AuthorityChannel(
                "grip_authority", s["both_contact"], AuthoritySource.OBSERVED,
                "bilateral fingertip contact (both_contact)"),
            "transport_authority": AuthorityChannel(
                "transport_authority", 1.0 - s["on_ground"], AuthoritySource.OBSERVED,
                "drop margin: not on_ground while lifted"),
        })

    def decode(self, intent: PhysicalIntent, authority: AuthorityMap) -> ExecutableOption:
        decoder = AffineAuthorityDecoder(
            name="pnp_scripted_expert_v3",
            order=_INTENT_ORDER,
            gain=tuple(1.0 for _ in _INTENT_ORDER),
            mode=self._mode,
            initiation=frozenset(CHAIN),
            termination="placed_and_settled",
        )
        option = decoder.decode(intent, authority)
        # record the realizer honestly: v0 realizes the option via the scripted expert
        prov = dict(option.provenance)
        prov["realizer"] = "hymeko_rl scripted expert v3 (embodiment decoder)"
        return ExecutableOption(
            name=option.name, mode=option.mode, command=option.command,
            initiation=option.initiation, termination=option.termination,
            provenance=prov,
        )

    def execute(self, option: ExecutableOption) -> ResponseTrace:
        cur_i = CHAIN.index(self._mode)
        commands: list[tuple[float, ...]] = []
        signals: list[dict[str, float]] = []
        any_death = False
        dropped = False
        carry_contact_ok = True
        end = OptionEnd.TRUNCATED
        for _ in range(self.max_option_steps):
            action = np.asarray(self._env.expert_action, dtype=np.float64)
            _obs, _r, term, trunc, info = self._env.step(action)
            self._info = dict(info)
            s = self._signals(info)
            commands.append(tuple(float(x) for x in action))
            signals.append(s)
            self._was_lifted = self._was_lifted or (s["lifted"] >= self.lift_thresh)
            any_death = any_death or bool(s["death"])
            if self._was_lifted and s["on_ground"]:
                dropped = True
            if self._mode == "CARRY" and s["both_contact"] < 1.0:
                carry_contact_ok = False
            if any_death or dropped:
                end = OptionEnd.ABORTED
                break
            if term or trunc:
                end = (OptionEnd.COMPLETED
                       if (s["placed_stable"] or s["settled"]) else OptionEnd.TRUNCATED)
                break
            if CHAIN.index(self._classify(info)) != cur_i:
                end = OptionEnd.HANDOFF
                break
        self._end_mode = self._classify(self._info)
        last = signals[-1] if signals else self._signals(self._info)
        if end == OptionEnd.HANDOFF and self._end_mode == "SETTLE" and (
            last["placed_stable"] or last["settled"]
        ):
            end = OptionEnd.COMPLETED
        return ResponseTrace(
            option=option.name,
            commands=tuple(commands),
            signals=tuple(signals),
            end=end,
            provenance={
                "option": option.name,
                "steps": len(commands),
                "end_mode": self._end_mode,
                "placed_stable": bool(last["placed_stable"] or last["settled"]),
                "any_death": any_death,
                "dropped": dropped,
                "carry_contact_ok": carry_contact_ok,
                "min_lifted": min((s["lifted"] for s in signals), default=0.0),
                "max_lifted": max((s["lifted"] for s in signals), default=0.0),
            },
        )

    def certify(
        self, state: StructuredStateLike, trace: ResponseTrace
    ) -> CertificateResult:
        return pnp_certificate_suite().evaluate(state, trace)

    def transition(self, mode: str, certificate: CertificateResult) -> str:
        cur_i = CHAIN.index(mode)
        tgt_i = CHAIN.index(self._end_mode)
        if tgt_i > cur_i:
            nxt = CHAIN[cur_i + 1]
        elif tgt_i < cur_i:
            nxt = CHAIN[cur_i - 1]
        else:
            nxt = mode
        if nxt != mode and not self.model.is_legal_transition(mode, nxt):
            nxt = mode
        return nxt
