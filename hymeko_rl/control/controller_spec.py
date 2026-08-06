"""A scripted controller's FSM + parameters as a declarative spec — the in-memory form of the
``meta_controller.hymeko`` vocabulary (the demonstrator side of the substrate).

A :class:`ControllerSpec` is read from a ``.hymeko`` controller profile's ``controller_spec`` bundle:
``fsm_phase`` members are the FSM vertices (each binding a target LAW by name and declaring its outgoing
transitions ``on <guard-event> to <phase>;``), all other members are field-carrying parameter terms whose
scalars merge into :attr:`params`. Python binds the named guards/laws (behaviour registries); the FSM
*topology* lives in the profile. Same reader (``read_bundle``) and shape as
:class:`hymeko_rl.agents.strategy_spec.StrategySpec`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from hymeko_rl.env._profile import read_bundle

_PHASE_KIND = "fsm_phase"
_LAW = re.compile(r"\blaw\s+(\w+)\s*;")
_TRANSITION = re.compile(r"\bon\s+(\w+)\s+to\s+(\w+)\s*;")
_FIELD = re.compile(r"(\w+)\s+(-?[\d.]+)\s*;")


@dataclass(frozen=True)
class PhaseSpec:
    """One FSM vertex: its target-law binding and its outgoing ``(event, destination)`` arcs, in
    declaration order (the order is the guard-evaluation priority)."""

    name: str
    law: str
    transitions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ControllerSpec:
    """Declarative scripted-controller FSM + parameters.

    # Preconditions (``from_hymeko``) the profile declares one ``controller_spec`` whose bundle contains
      at least one ``fsm_phase`` member; every transition destination is a declared phase.
    # Invariants ``phases`` is non-empty; :attr:`initial` is ``phases[0].name``; every transition
      destination names a phase in ``phases``.
    """

    phases: tuple[PhaseSpec, ...]
    params: dict[str, float]

    @property
    def initial(self) -> str:
        return self.phases[0].name

    def phase(self, name: str) -> PhaseSpec:
        """# Errors ``KeyError`` if ``name`` is not a declared phase."""
        return {p.name: p for p in self.phases}[name]

    def step(self, phase: str, fires: Callable[[str], bool]) -> tuple[str, str]:
        """One FSM transition: the first declared outgoing event of ``phase`` whose guard ``fires`` wins.

        # Postconditions returns ``(next_phase, event)``; ``event == ""`` iff no guard fired (self-loop).
        """
        for event, dst in self.phase(phase).transitions:
            if fires(event):
                return dst, event
        return phase, ""

    @classmethod
    def from_hymeko(cls, profile: str | Path, spec_kind: str = "controller_spec") -> "ControllerSpec":
        """Build the spec from a controller profile's bundle.

        # Errors ``FileNotFoundError``; ``ValueError`` (no/multiple bundle, no phase member, a phase
          without a ``law``, or a transition to an undeclared phase).
        """
        phases: list[PhaseSpec] = []
        params: dict[str, float] = {}
        for name, kind, body, _w in read_bundle(profile, spec_kind):
            if kind == _PHASE_KIND:
                law = _LAW.search(body)
                if law is None:
                    raise ValueError(f"{profile}: phase {name!r} declares no `law <name>;`")
                phases.append(PhaseSpec(name=name, law=law.group(1),
                                        transitions=tuple((e, d) for e, d in _TRANSITION.findall(body))))
            else:
                params.update({k: float(v) for k, v in _FIELD.findall(body)})
        if not phases:
            raise ValueError(f"{profile}: {spec_kind} bundles no {_PHASE_KIND} member")
        declared = {p.name for p in phases}
        dangling = [(p.name, d) for p in phases for _e, d in p.transitions if d not in declared]
        if dangling:
            raise ValueError(f"{profile}: transitions to undeclared phase(s): {dangling}")
        return cls(phases=tuple(phases), params=params)
