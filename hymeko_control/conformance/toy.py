"""A torch-free TOY reference adapter (generic 1-DOF reach), used only to
self-test the conformance battery. It is NOT a scenario: no physics engine, no
embodiment specifics -- just enough to drive the full CIP-0 lifecycle to a
certified stop so the battery has something legal to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..cip.authority import AuthorityChannel, AuthorityMap, AuthoritySource
from ..cip.certificate import Certificate, CertificateResult, CertificateSuite
from ..cip.option import (
    AffineAuthorityDecoder,
    ExecutableOption,
    OptionEnd,
    ResponseTrace,
)
from ..cip.physical_intent import PhysicalIntent
from ..cip.structured_state import ControlState, StructuredStateLike
from ..language.ir import ControlModel
from ..language.schema_v0 import CertificateKind
from ..language.validator import validate

_APPROACH, _TOUCH, _HOLD = "APPROACH", "TOUCH", "HOLD"


def toy_reach_spec() -> dict[str, Any]:
    """A valid schema-v0 scenario dict for a 1-DOF reach (APPROACH->TOUCH->HOLD)."""
    return {
        "schema_version": "v0",
        "name": "toy_reach",
        "entities": [
            {"id": "arm", "kind": "body"},
            {"id": "j0", "kind": "joint"},
            {"id": "goal", "kind": "target"},
            {"id": "enc", "kind": "sensor"},
        ],
        "morphology": [
            {"kind": "kinematic", "source": "arm", "target": "j0"},
            {"kind": "actuation", "source": "j0", "target": "arm"},
        ],
        "ports": [
            {"id": "j0.tau", "kind": "actuation", "owner": "j0"},
            {"id": "enc.q", "kind": "observation", "owner": "enc"},
            {"id": "arm.tip", "kind": "contact", "owner": "arm"},
        ],
        "physics": [
            {"name": "link_inertia", "role": "energy_storage"},
            {"name": "joint_friction", "role": "dissipation"},
            {"name": "tip_contact", "role": "contact"},
        ],
        "hybrid": {
            "modes": [
                {"name": _APPROACH},
                {"name": _TOUCH},
                {"name": _HOLD, "reset": {"integral_err": 0.0}},
            ],
            "transitions": [
                {"source": _APPROACH, "dest": _TOUCH, "guard": "near"},
                {"source": _TOUCH, "dest": _HOLD, "guard": "contact"},
                {"source": _TOUCH, "dest": _APPROACH, "guard": "lost"},
            ],
            "initial_mode": _APPROACH,
        },
        "task": {
            "intents": [
                {"name": "reach_velocity", "lower": 0.0, "upper": 1.0, "unit": "1/s"},
            ],
            "authorities": [
                {"name": "actuation", "source": "observed",
                 "provenance": "toy: unit actuator, measured"},
            ],
            "certificates": [
                {"name": "reached", "kind": "success", "predicate": "dist<=0.05"},
                {"name": "no_overshoot", "kind": "safety", "predicate": "dist>=0"},
            ],
        },
    }


def toy_model() -> ControlModel:
    return validate(toy_reach_spec())


@dataclass
class ToyReachAdapter:
    """Drives a 1-DOF reach to a certified HOLD. Deterministic; torch-free."""

    dist: float = 1.0
    step: float = 0.05
    _t: int = field(default=0, init=False)
    _mode_dist: float = field(default=1.0, init=False)
    _mode: str = field(default=_APPROACH, init=False)
    _bounds: dict = field(
        default_factory=lambda: {"reach_velocity": (0.0, 1.0)}, init=False
    )

    # 1. OBSERVE ----------------------------------------------------------
    def observe(self) -> StructuredStateLike:
        state = ControlState(
            t=self._t,
            phase=self._mode,
            signals={"dist": self.dist},
            contact={"tip": self.dist <= 0.01},
        )
        self._t += 1
        return state

    # 2. IDENTIFY MODE ----------------------------------------------------
    def identify_mode(self, state: StructuredStateLike) -> str:
        d = state.signal("dist")
        self._mode_dist = d
        if d > 0.1:
            self._mode = _APPROACH
        elif d > 0.01:
            self._mode = _TOUCH
        else:
            self._mode = _HOLD
        return self._mode

    # 3. FORM INTENT ------------------------------------------------------
    def form_intent(self, state: StructuredStateLike, task: Any) -> PhysicalIntent:
        demand = min(max(state.signal("dist"), 0.0), 1.0)
        return PhysicalIntent(
            components={"reach_velocity": demand}, bounds=self._bounds
        )

    # 4. MEASURE AUTHORITY ------------------------------------------------
    def measure_authority(self, state: StructuredStateLike, mode: str) -> AuthorityMap:
        return AuthorityMap(
            channels={
                "actuation": AuthorityChannel(
                    name="actuation",
                    value=1.0,
                    source=AuthoritySource.OBSERVED,
                    provenance="toy: unit actuator authority (measured)",
                )
            }
        )

    # 5. DECODE (deterministic) ------------------------------------------
    def decode(self, intent: PhysicalIntent, authority: AuthorityMap) -> ExecutableOption:
        decoder = AffineAuthorityDecoder(
            name="toy_affine",
            order=("reach_velocity",),
            gain=(1.0,),
            mode=self._mode,
            initiation=frozenset({_APPROACH, _TOUCH, _HOLD}),
            termination="reached",
        )
        return decoder.decode(intent, authority)

    # 6. EXECUTE ----------------------------------------------------------
    def execute(self, option: ExecutableOption) -> ResponseTrace:
        self.dist = max(0.0, self.dist - self.step)
        # an option COMPLETES only once we are holding in the terminal mode;
        # in APPROACH/TOUCH it merely HANDS OFF to the next mode's option.
        end = OptionEnd.COMPLETED if self._mode == _HOLD else OptionEnd.HANDOFF
        return ResponseTrace(
            option=option.name,
            commands=(option.command,),
            signals=({"dist": self.dist},),
            end=end,
            provenance={"option": option.name, "applied_command": option.command},
        )

    # 7. CERTIFY (reward-independent) ------------------------------------
    def certify(
        self, state: StructuredStateLike, trace: ResponseTrace
    ) -> CertificateResult:
        suite = CertificateSuite(
            certificates=(
                Certificate("reached", CertificateKind.SUCCESS,
                            lambda s, tr: s.signal("dist") <= 0.05),
                Certificate("no_overshoot", CertificateKind.SAFETY,
                            lambda s, tr: s.signal("dist") >= 0.0),
            )
        )
        return suite.evaluate(state, trace)

    # 8. TRANSITION -------------------------------------------------------
    def transition(self, mode: str, certificate: CertificateResult) -> str:
        if mode == _APPROACH:
            return _TOUCH if self._mode_dist <= 0.1 else _APPROACH
        if mode == _TOUCH:
            return _HOLD if self._mode_dist <= 0.01 else _TOUCH
        return _HOLD
