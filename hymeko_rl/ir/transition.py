"""Hybrid transitions: a guard, an optional reset map, a handoff descriptor, and (later) an energy certificate per edge.

Each edge M_i -> M_{i+1} carries a :class:`TransitionGuard` (when it may fire), a reset map (the post-jump state; identity
by default), and a :class:`HandoffDescriptor` that records the full boundary state so a downstream stage can be replayed
from it. R11.2 wires the structure and the (measured) energy certificate; the reward/optimization that rides on these
edges arrives at later boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from hymeko_rl.ir.energy import EnergyTransitionCertificate
from hymeko_rl.ir.hybrid_mode import HybridMode
from hymeko_rl.ir.rollout import RolloutState, Vec


class TransitionGuardError(RuntimeError):
    """Raised when a transition is fired from a state its guard does not admit."""


@dataclass(frozen=True)
class TransitionGuard:
    """A named firing condition for the edge ``source -> target``."""

    name: str
    source: HybridMode
    target: HybridMode
    condition: str
    predicate: Callable[[RolloutState], bool]

    def __post_init__(self) -> None:
        assert self.target - self.source == 1, "a guard connects consecutive modes M_i -> M_{i+1}"

    def allows(self, state: RolloutState) -> bool:
        return bool(self.predicate(state))


@dataclass(frozen=True)
class HandoffDescriptor:
    """The complete boundary state handed from one mode to the next: configuration, rates, last torque, object pose and
    velocity, goal, contact count, phase-energy scalar, and the mode pair. ``is_complete`` is the wiring gate."""

    q: Vec
    qdot: Vec
    prev_tau: Vec
    x_coin: Vec
    xdot_coin: Vec
    x_goal: Vec
    n_contacts: int
    e_phase: float
    mode_from: HybridMode
    mode_to: HybridMode

    def is_complete(self) -> bool:
        """Postcondition: True iff every array field is finite and non-empty, ``e_phase`` is finite, and the modes are a
        consecutive forward pair."""
        arrays = (self.q, self.qdot, self.prev_tau, self.x_coin, self.xdot_coin, self.x_goal)
        if any(a.size == 0 or not bool(np.all(np.isfinite(a))) for a in arrays):
            return False
        if not np.isfinite(self.e_phase) or self.n_contacts < 0:
            return False
        return self.mode_to - self.mode_from == 1


@dataclass(frozen=True)
class HybridTransition:
    """A guarded edge with a reset map and optional descriptors. ``fire`` applies the reset map only if the guard admits
    the pre-jump state (raising otherwise), so a fired transition is always guard-consistent."""

    source: HybridMode
    target: HybridMode
    guard: TransitionGuard
    reset_map: Callable[[RolloutState], RolloutState] = lambda s: s
    handoff: Optional[HandoffDescriptor] = None
    energy_cert: Optional[EnergyTransitionCertificate] = None

    def __post_init__(self) -> None:
        assert self.guard.source == self.source and self.guard.target == self.target, "guard must match the edge"

    def fire(self, state: RolloutState) -> RolloutState:
        """Preconditions: ``guard.allows(state)``. Postconditions: returns the post-jump state from the reset map."""
        if not self.guard.allows(state):
            raise TransitionGuardError(f"{self.guard.name}: guard blocked firing {self.source.name}->{self.target.name}")
        return self.reset_map(state)
