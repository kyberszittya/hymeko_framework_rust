"""Task-independent hierarchical skill routing: an upstream (learned) option policy drives the system until a handoff
certificate fires, then control passes to a FROZEN downstream skill that carries the episode to termination. This is the
general skill-composition mechanism the coin carry→settling pipeline instantiates (carry option → K6-handoff → frozen pi_0).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SkillRoute:
    """One upstream→downstream handoff route. ``handed_off(outcome)`` decides when the upstream option yields; ``downstream``
    is a frozen policy/skill invoked only after a valid handoff. Kept data-only so any task wires its own certificate + skill.

    # Invariant: downstream is never trained by the engine (frozen skill); the upstream option owns the pre-handoff segment."""

    name: str
    handed_off: Callable[[dict], bool]
    downstream: Any                       # a frozen callable/skill; engine treats it as opaque
    upstream_owns_until_handoff: bool = True

    def route(self, outcome: dict) -> str:
        """Return which stage owns continuation given an executed-option outcome."""
        return "downstream_frozen_skill" if self.handed_off(outcome) else "upstream_option_redecide"
