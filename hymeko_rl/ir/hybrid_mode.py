"""The hybrid mode automaton M0..M7 and its per-mode state invariants + trace validation.

The deployed episode is a hybrid system: EXACT_ZERO_HOME -> FREE_REACH -> PRECONTACT_ALIGNMENT -> CAPTURE ->
CONTROLLED_DELIVERY -> TARGET_ENTRY -> SETTLE -> K6_SUCCESS. Modes are an ``IntEnum`` so ordering is total and a trace is
just a monotone walk M0..M7. ``StateInvariant`` attaches a predicate over a :class:`RolloutState` to a mode (e.g.
ZERO_HOME requires q and qdot at rest); ``ModeTrace.is_valid`` rejects skips and regressions.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Sequence

import numpy as np

from hymeko_rl.ir.rollout import RolloutState, Vec


class HybridMode(IntEnum):
    """Ordered discrete modes of the delivery hybrid system; the integer value is the canonical sequence index."""

    ZERO_HOME = 0
    FREE_REACH = 1
    PRECONTACT_ALIGNMENT = 2
    CAPTURE = 3
    CONTROLLED_DELIVERY = 4
    TARGET_ENTRY = 5
    SETTLE = 6
    K6_SUCCESS = 7


@dataclass(frozen=True)
class StateInvariant:
    """A named predicate that must hold while the system is in ``mode``."""

    name: str
    mode: HybridMode
    predicate: Callable[[RolloutState], bool]

    def holds(self, state: RolloutState) -> bool:
        return bool(self.predicate(state))


def zero_home_invariant(q_atol: float = 1e-9, vel_atol: float = 1e-9) -> StateInvariant:
    """The M0 invariant: the configuration is at the (any-D) origin and every velocity is zero."""

    def _pred(s: RolloutState) -> bool:
        at_origin = bool(np.allclose(s.q, np.zeros_like(s.q), atol=q_atol, rtol=0.0))
        still = float(np.abs(s.qdot).max(initial=0.0)) <= vel_atol and \
            float(np.abs(s.object_vel).max(initial=0.0)) <= vel_atol
        return at_origin and still

    return StateInvariant("ZERO_HOME_AT_REST", HybridMode.ZERO_HOME, _pred)


@dataclass(frozen=True)
class ModeTrace:
    """An observed sequence of modes over one episode. Valid iff it starts at ZERO_HOME, never regresses, and advances one
    mode at a time (repeats allowed, skips forbidden)."""

    modes: tuple[HybridMode, ...]

    def __post_init__(self) -> None:
        assert len(self.modes) >= 1, "a trace has at least one mode"

    @staticmethod
    def canonical() -> "ModeTrace":
        """The full nominal walk ZERO_HOME .. K6_SUCCESS, one entry per mode."""
        return ModeTrace(tuple(HybridMode))

    def first_invalid_step(self) -> "int | None":
        """Index of the first illegal step (regression or skip), or None if the whole trace is valid.

        Postcondition: returns None iff :meth:`is_valid`.
        """
        if self.modes[0] != HybridMode.ZERO_HOME:
            return 0
        for i, (a, b) in enumerate(zip(self.modes[:-1], self.modes[1:]), start=1):
            if b - a not in (0, 1):
                return i
        return None

    def is_valid(self) -> bool:
        return self.first_invalid_step() is None


def build_mode_trace(modes: Sequence[HybridMode]) -> ModeTrace:
    """Adapter helper: wrap a recorded mode sequence into a :class:`ModeTrace`."""
    return ModeTrace(tuple(modes))


# a small alias so adapters can annotate the configuration vector without importing rollout directly
ConfigVec = Vec
