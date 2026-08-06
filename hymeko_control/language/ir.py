"""HyMeKo control language --- in-memory IR (the validated ``ControlModel``).

These frozen dataclasses are what a scenario adapter reads at runtime. They are
produced by :func:`hymeko_control.language.validator.validate`; construct them
directly only in tests. All mapping fields are wrapped read-only, so a model is
genuinely immutable once built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .._frozen import freeze_mapping
from .schema_v0 import (
    CertificateKind,
    EntityKind,
    PhysicsRole,
    PortKind,
    RelationKind,
)


@dataclass(frozen=True)
class Entity:
    """A declarable entity (body / joint / object / target / sensor / controller)."""

    eid: str
    kind: EntityKind
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", freeze_mapping(self.attributes))


@dataclass(frozen=True)
class Port:
    """A typed port owned by an entity (effort/flow/actuation/observation/...)."""

    pid: str
    kind: PortKind
    owner: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", freeze_mapping(self.attributes))


@dataclass(frozen=True)
class MorphologyRelation:
    """A directed morphology relation ``source -> target`` of a given kind."""

    kind: RelationKind
    source: str
    target: str


@dataclass(frozen=True)
class PhysicsElement:
    """A bond-graph physics element (storage / dissipation / interconnection / ...)."""

    name: str
    role: PhysicsRole
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", freeze_mapping(self.attributes))


@dataclass(frozen=True)
class Mode:
    """A hybrid-control discrete mode.

    ``reset`` maps continuous-state names to the value they are reset to when the
    automaton *enters* this mode (the reset map of hybrid-automaton theory).
    """

    name: str
    invariant: str | None = None
    reset: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reset", freeze_mapping(self.reset))


@dataclass(frozen=True)
class Transition:
    """A legal discrete transition ``source -> dest`` gated by a named guard."""

    source: str
    dest: str
    guard: str
    event: str | None = None


@dataclass(frozen=True)
class IntentSpec:
    """A physical-intent component: a named, bounded scalar demand."""

    name: str
    lower: float
    upper: float
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"IntentSpec {self.name!r}: lower {self.lower} > upper {self.upper}"
            )


@dataclass(frozen=True)
class AuthoritySpec:
    """A declared authority channel and where its value comes from."""

    name: str
    source: str  # observed | modeled | assumed
    provenance: str


@dataclass(frozen=True)
class CertificateSpec:
    """A declared external certificate (success or safety)."""

    name: str
    kind: CertificateKind
    predicate: str  # symbolic reference resolved by the scenario adapter


@dataclass(frozen=True)
class ControlModel:
    """The validated declarative model an adapter runs against.

    # Invariants
    Every reference is internally consistent (guaranteed by the validator):
    port owners, transition endpoints and ``initial_mode`` all resolve.
    """

    name: str
    entities: tuple[Entity, ...]
    morphology: tuple[MorphologyRelation, ...]
    ports: tuple[Port, ...]
    physics: tuple[PhysicsElement, ...]
    modes: tuple[Mode, ...]
    transitions: tuple[Transition, ...]
    initial_mode: str
    intents: tuple[IntentSpec, ...]
    authorities: tuple[AuthoritySpec, ...]
    certificates: tuple[CertificateSpec, ...]

    # -- lookups ----------------------------------------------------------
    def entity(self, eid: str) -> Entity:
        for e in self.entities:
            if e.eid == eid:
                return e
        raise KeyError(f"no entity {eid!r}")

    def mode(self, name: str) -> Mode:
        for m in self.modes:
            if m.name == name:
                return m
        raise KeyError(f"no mode {name!r}")

    def mode_names(self) -> frozenset[str]:
        return frozenset(m.name for m in self.modes)

    def ports_of(self, eid: str) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.owner == eid)

    def legal_transitions(self, from_mode: str) -> tuple[Transition, ...]:
        """Transitions whose ``source`` is ``from_mode``."""
        return tuple(t for t in self.transitions if t.source == from_mode)

    def is_legal_transition(self, from_mode: str, to_mode: str) -> bool:
        """A mode may always self-loop; otherwise a declared transition must exist."""
        if from_mode == to_mode:
            return True
        return any(
            t.source == from_mode and t.dest == to_mode for t in self.transitions
        )

    def is_terminal_mode(self, name: str) -> bool:
        """True iff ``name`` has no declared transition to a *different* mode.

        A terminal mode can only self-loop; reaching it with a passing
        certificate is what completes an episode.
        """
        return not any(
            t.source == name and t.dest != name for t in self.transitions
        )

    def intent_bounds(self) -> dict[str, tuple[float, float]]:
        return {s.name: (s.lower, s.upper) for s in self.intents}
