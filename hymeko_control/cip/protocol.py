"""CIP-0 --- the common scenario-adapter Protocol.

Every embodiment scenario implements :class:`CIP0Adapter`. The runtime
(:mod:`hymeko_control.cip.runtime`) drives an adapter through the lifecycle::

    OBSERVE -> IDENTIFY MODE -> FORM INTENT -> MEASURE AUTHORITY
    -> DECODE -> EXECUTE OPTION -> MEASURE RESPONSE -> CERTIFY -> TRANSITION

The adapter depends on this core; the core never imports the adapter.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .authority import AuthorityMap
from .certificate import CertificateResult
from .option import ExecutableOption, ResponseTrace
from .physical_intent import PhysicalIntent
from .structured_state import StructuredStateLike


@runtime_checkable
class CIP0Adapter(Protocol):
    """The eight-method contract a scenario adapter must satisfy.

    Method contracts (enforced by the runtime):

    * ``observe`` returns a state whose ``t`` is non-decreasing across a run.
    * ``identify_mode`` returns a mode name that exists in the ``ControlModel``.
    * ``form_intent`` returns a :class:`PhysicalIntent` that is bounded.
    * ``measure_authority`` returns an :class:`AuthorityMap` with provenance.
    * ``decode`` is deterministic in ``(intent, authority)``.
    * ``execute`` returns a :class:`ResponseTrace` that references the option.
    * ``certify`` is reward-independent (state + trace only).
    * ``transition`` returns a mode reachable by a legal transition (or a
      self-loop / terminal mode).
    """

    def observe(self) -> StructuredStateLike: ...

    def identify_mode(self, state: StructuredStateLike) -> str: ...

    def form_intent(self, state: StructuredStateLike, task: Any) -> PhysicalIntent: ...

    def measure_authority(self, state: StructuredStateLike, mode: str) -> AuthorityMap: ...

    def decode(self, intent: PhysicalIntent, authority: AuthorityMap) -> ExecutableOption: ...

    def execute(self, option: ExecutableOption) -> ResponseTrace: ...

    def certify(
        self, state: StructuredStateLike, trace: ResponseTrace
    ) -> CertificateResult: ...

    def transition(self, mode: str, certificate: CertificateResult) -> str: ...
