"""CIP-0 --- runtime lifecycle driver.

:class:`CIP0Runtime` binds a validated :class:`~hymeko_control.language.ir.ControlModel`
to a :class:`~hymeko_control.cip.protocol.CIP0Adapter` and drives the lifecycle
one ``tick`` at a time, enforcing the CIP-0 contract at every step:

* causal observation (``t`` non-decreasing);
* identified mode exists in the model;
* intent is bounded;
* authority carries provenance;
* decoding is deterministic (re-decoded and compared);
* the response trace references the executed option;
* mode transitions are legal in the declarative model.

The runtime is READ-ONLY with respect to the adapter's value objects: it never
mutates a state, intent, option, or trace it receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..language.ir import ControlModel
from .authority import AuthorityMap
from .certificate import CertificateResult
from .option import ExecutableOption, ResponseTrace
from .physical_intent import PhysicalIntent
from .protocol import CIP0Adapter
from .structured_state import StructuredStateLike


class CausalityError(RuntimeError):
    """Raised when an observation's tick index is not causal (decreases)."""


class ModeError(RuntimeError):
    """Raised when a mode is unknown or a transition is illegal."""


class DeterminismError(RuntimeError):
    """Raised when re-decoding the same intent/authority yields a different option."""


class ProvenanceError(RuntimeError):
    """Raised when a response trace does not reference its option."""


@dataclass(frozen=True)
class TickRecord:
    """Immutable provenance of one CIP-0 lifecycle step."""

    tick: int
    mode: str
    intent: PhysicalIntent
    authority: AuthorityMap
    option: ExecutableOption
    trace: ResponseTrace
    certificate: CertificateResult
    next_mode: str


@dataclass
class CIP0Runtime:
    """Drives a :class:`CIP0Adapter` against a :class:`ControlModel`.

    # Invariants
    ``_last_t`` only ever increases; ``mode`` is always a declared mode.
    """

    model: ControlModel
    adapter: CIP0Adapter
    mode: str = ""
    _last_t: int = field(default=-1, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.mode:
            self.mode = self.model.initial_mode
        if self.mode not in self.model.mode_names():
            raise ModeError(f"initial mode {self.mode!r} not in model")

    # -- per-step contract checks ----------------------------------------
    def _check_causal(self, state: StructuredStateLike) -> None:
        if state.t < self._last_t:
            raise CausalityError(
                f"observation tick {state.t} < last {self._last_t} (non-causal)"
            )
        self._last_t = state.t

    def _check_mode(self, mode: str) -> None:
        if mode not in self.model.mode_names():
            raise ModeError(f"identified mode {mode!r} not declared in model")

    def _check_intent(self, intent: PhysicalIntent) -> None:
        if not intent.is_bounded():
            raise self._intent_error(intent)

    @staticmethod
    def _intent_error(intent: PhysicalIntent) -> Exception:
        from .physical_intent import IntentBoundsError

        return IntentBoundsError(f"intent not bounded: {dict(intent.components)}")

    @staticmethod
    def _check_deterministic(
        adapter: CIP0Adapter,
        intent: PhysicalIntent,
        authority: AuthorityMap,
        option: ExecutableOption,
    ) -> None:
        again = adapter.decode(intent, authority)
        if (again.command, again.mode, again.name) != (
            option.command,
            option.mode,
            option.name,
        ):
            raise DeterminismError(
                f"decode not deterministic for option {option.name!r}"
            )

    def _check_transition(self, from_mode: str, to_mode: str) -> None:
        if to_mode not in self.model.mode_names():
            raise ModeError(f"transition target {to_mode!r} not a declared mode")
        if not self.model.is_legal_transition(from_mode, to_mode):
            raise ModeError(f"illegal transition {from_mode!r} -> {to_mode!r}")

    # -- one lifecycle step ----------------------------------------------
    def tick(self, task: Any = None) -> TickRecord:
        """Run one OBSERVE..TRANSITION cycle and return its provenance.

        # Postconditions
        Returns a :class:`TickRecord`; ``self.mode`` advances to ``next_mode``.
        Raises a specific contract error (Causality/Mode/Determinism/Provenance)
        on any violation, never a silent skip.
        """
        state = self.adapter.observe()
        self._check_causal(state)

        mode = self.adapter.identify_mode(state)
        self._check_mode(mode)

        intent = self.adapter.form_intent(state, task)
        self._check_intent(intent)

        authority = self.adapter.measure_authority(state, mode)
        authority.require_provenance()

        option = self.adapter.decode(intent, authority)
        self._check_deterministic(self.adapter, intent, authority, option)

        trace = self.adapter.execute(option)
        if not trace.references(option):
            raise ProvenanceError(
                f"response trace does not reference option {option.name!r}"
            )

        certificate = self.adapter.certify(state, trace)

        next_mode = self.adapter.transition(mode, certificate)
        self._check_transition(mode, next_mode)
        self.mode = next_mode

        return TickRecord(
            tick=state.t,
            mode=mode,
            intent=intent,
            authority=authority,
            option=option,
            trace=trace,
            certificate=certificate,
            next_mode=next_mode,
        )

    def run(self, max_ticks: int, task: Any = None) -> list[TickRecord]:
        """Tick until a terminal self-loop with a passing certificate, an
        ABORTED option, or ``max_ticks``. Returns the record list.
        """
        from .option import OptionEnd

        records: list[TickRecord] = []
        for _ in range(max_ticks):
            rec = self.tick(task)
            records.append(rec)
            if rec.trace.end in (OptionEnd.ABORTED, OptionEnd.COMPLETED):
                break
            if (
                rec.next_mode == rec.mode
                and self.model.is_terminal_mode(rec.next_mode)
                and rec.certificate.passed
            ):
                break
        return records
