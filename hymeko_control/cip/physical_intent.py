"""CIP-0 --- physical intent.

A ``PhysicalIntent`` is a typed, *bounded* bundle of named scalar demands
(approach direction, grasp force, lift clearance, forward velocity, yaw rate,
...). The vocabulary and bounds are declared per-embodiment in the scenario
language (``task.intents``); this value type carries the runtime demand and
enforces the bounds.

Bounding is a hard contract: an intent that exceeds its declared range is a
programming error, caught at construction. ``clipped`` produces a saturated
intent for controllers that prefer to saturate rather than fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .._frozen import freeze_mapping


class IntentBoundsError(ValueError):
    """Raised when an intent component falls outside its declared bounds."""


@dataclass(frozen=True)
class PhysicalIntent:
    """A bounded set of named physical demands.

    # Preconditions
    Every key in ``components`` has a ``(lo, hi)`` entry in ``bounds`` with
    ``lo <= hi``, and (unless built via :meth:`clipped`) every component lies
    within its bounds.

    # Invariants
    Frozen and read-only; ``is_bounded()`` holds for any instance that did not
    come from a bounds-violating direct construction.
    """

    components: Mapping[str, float]
    bounds: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", freeze_mapping(self.components))
        object.__setattr__(self, "bounds", freeze_mapping(self.bounds))
        for name, value in self.components.items():
            if name not in self.bounds:
                raise IntentBoundsError(f"intent {name!r} has no declared bounds")
            lo, hi = self.bounds[name]
            if lo > hi:
                raise IntentBoundsError(f"intent {name!r}: inverted bounds ({lo}, {hi})")
            if not lo <= value <= hi:
                raise IntentBoundsError(
                    f"intent {name!r} = {value} outside bounds [{lo}, {hi}]"
                )

    def is_bounded(self) -> bool:
        """True iff every component lies within its declared bounds."""
        return all(
            self.bounds[n][0] <= v <= self.bounds[n][1]
            for n, v in self.components.items()
        )

    def get(self, name: str) -> float:
        return self.components[name]

    def vector(self, order: tuple[str, ...]) -> tuple[float, ...]:
        """Deterministic ordered vector view over the named components."""
        return tuple(self.components[k] for k in order)

    @classmethod
    def clipped(
        cls,
        components: Mapping[str, float],
        bounds: Mapping[str, tuple[float, float]],
    ) -> "PhysicalIntent":
        """Build an intent, saturating each component into its bounds.

        # Postconditions
        The result satisfies :meth:`is_bounded`.
        """
        clamped = {
            n: min(max(v, bounds[n][0]), bounds[n][1]) for n, v in components.items()
        }
        return cls(components=clamped, bounds=bounds)
