"""CIP-0 --- executable option + decoder + response trace.

An ``ExecutableOption`` is a first-class hybrid option bundling
*initiation* (which modes it may start in), *policy* (the decoded command),
and *termination* (the certificate that ends it), plus full provenance of how
it was decoded. Decoding is DETERMINISTIC: the same ``(intent, authority)``
must always yield the same option -- no RNG, no local search inside the core.
(A scenario may still layer stochastic search on top, but the core contract is
a pure function.)

``OptionEnd`` mirrors ``hymeko_rl.option_rl.core.OptionEnd`` semantics
(HANDOFF / COMPLETED / ABORTED / TRUNCATED) but is re-declared here to keep the
core torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from .._frozen import freeze_mapping
from .authority import AuthorityMap
from .physical_intent import PhysicalIntent


class OptionEnd(str, Enum):
    """Semi-MDP option termination semantics."""

    HANDOFF = "handoff"
    COMPLETED = "completed"
    ABORTED = "aborted"  # safety abort
    TRUNCATED = "truncated"


@dataclass(frozen=True)
class ExecutableOption:
    """A decoded, executable hybrid option.

    # Invariants
    Frozen. ``provenance`` records the decoder id and snapshots of the intent
    and authority it was decoded from, so an execution is fully auditable.
    """

    name: str
    mode: str
    command: tuple[float, ...]
    initiation: frozenset[str]
    termination: str  # certificate name that terminates this option
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", tuple(float(c) for c in self.command))
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance))

    def may_initiate(self, mode: str) -> bool:
        return mode in self.initiation


@dataclass(frozen=True)
class ResponseTrace:
    """The measured response of executing an option.

    ``provenance`` must reference the option that produced it (``option`` name +
    ``command``), so response and command are never decoupled.
    """

    option: str
    commands: tuple[tuple[float, ...], ...]
    signals: tuple[Mapping[str, float], ...]
    end: OptionEnd
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signals",
            tuple(freeze_mapping(s) for s in self.signals),
        )
        object.__setattr__(self, "provenance", freeze_mapping(self.provenance))

    def references(self, option: ExecutableOption) -> bool:
        """True iff this trace provably came from ``option``."""
        return self.option == option.name and self.provenance.get("option") == option.name


@runtime_checkable
class Decoder(Protocol):
    """Deterministic intent+authority -> option map.

    # Contract
    ``decode`` is a pure function: no RNG, no I/O, no mutation. Calling it twice
    with equal arguments returns equal options.
    """

    def decode(self, intent: PhysicalIntent, authority: AuthorityMap) -> ExecutableOption: ...


@dataclass(frozen=True)
class AffineAuthorityDecoder:
    """A deterministic reference decoder: authority-weighted affine map.

    ``command[i] = gain[i] * intent[order[i]] * authority_weight(order[i])``.
    Authority scales the demand by the (clipped-to-[0,1]) available capability on
    the matching channel, so a demand is never decoded beyond measured authority.

    This is the torch-free example decoder used by conformance tests; scenarios
    supply their own embodiment-specific decoder implementing :class:`Decoder`.
    """

    name: str
    order: tuple[str, ...]
    gain: tuple[float, ...]
    mode: str
    initiation: frozenset[str]
    termination: str

    def __post_init__(self) -> None:
        if len(self.order) != len(self.gain):
            raise ValueError("AffineAuthorityDecoder: order/gain length mismatch")

    def _weight(self, authority: AuthorityMap, channel: str) -> float:
        if channel not in authority.channels:
            return 1.0
        return min(max(authority.value(channel), 0.0), 1.0)

    def decode(self, intent: PhysicalIntent, authority: AuthorityMap) -> ExecutableOption:
        command = tuple(
            g * intent.get(name) * self._weight(authority, name)
            for name, g in zip(self.order, self.gain)
        )
        provenance = {
            "decoder": self.name,
            "deterministic": True,
            "intent": dict(intent.components),
            "authority": {k: authority.value(k) for k in authority.channels},
            "order": self.order,
        }
        return ExecutableOption(
            name=f"{self.name}:{self.mode}",
            mode=self.mode,
            command=command,
            initiation=self.initiation,
            termination=self.termination,
            provenance=provenance,
        )
