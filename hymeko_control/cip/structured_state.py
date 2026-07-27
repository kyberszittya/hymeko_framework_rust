"""CIP-0 --- structured observation state.

The runtime input is an explicit STRUCTURED state (named signals + discrete
phase + contact flags + geometry + task metadata + a causal tick index), never
a bare vector. This mirrors ``hymeko_rl.option_rl.state.StructuredState`` by
DUCK TYPE (see :class:`StructuredStateLike`): a scenario may hand the CIP-0
runtime that class directly, or the torch-free :class:`ControlState` here, or
anything exposing the same surface. The core never imports ``hymeko_rl`` to
stay dependency-clean.

Immutability is load-bearing: the runtime reads state and must never mutate it
("no hidden state modification"). ``ControlState`` is frozen and wraps its
mappings read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from .._frozen import freeze_mapping


@runtime_checkable
class StructuredStateLike(Protocol):
    """Structural contract for a CIP-0 observation.

    ``t`` is the causal tick index: it must be non-decreasing across a run so the
    runtime can reject non-causal observations.
    """

    @property
    def t(self) -> int: ...

    @property
    def phase(self) -> str: ...

    def signal(self, name: str) -> float: ...

    def flat(self) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class ControlState:
    """A torch-free structured observation.

    # Invariants
    All mapping fields are read-only after construction; the instance is frozen.
    """

    t: int
    phase: str
    signals: Mapping[str, float] = field(default_factory=dict)
    contact: Mapping[str, bool] = field(default_factory=dict)
    geometry: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t < 0:
            raise ValueError(f"ControlState.t must be >= 0, got {self.t}")
        object.__setattr__(self, "signals", freeze_mapping(self.signals))
        object.__setattr__(self, "contact", freeze_mapping(self.contact))
        object.__setattr__(self, "geometry", freeze_mapping(self.geometry))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))

    def signal(self, name: str) -> float:
        """Return a named scalar observation.

        # Preconditions
        ``name`` is a declared signal; raises ``KeyError`` otherwise.
        """
        return self.signals[name]

    def flat(self) -> tuple[float, ...]:
        """Deterministic flat view: signals sorted by name, then contact flags."""
        sig = tuple(self.signals[k] for k in sorted(self.signals))
        con = tuple(1.0 if self.contact[k] else 0.0 for k in sorted(self.contact))
        return sig + con
