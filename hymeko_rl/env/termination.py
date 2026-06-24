"""Episode-termination conditions as a declarative spec — the model's failure predicate.

A :class:`TerminationSpec` is a tuple of condition kinds; the episode ends ("death") if **any**
condition holds against the env's :class:`~hymeko_rl.env.safety.SafetyState`. Read from a
``.hymeko`` task profile's ``termination_spec`` (``meta_task.hymeko``'s ``termination`` vocab), so
*what counts as failure* lives in the model, beside the reward — not hard-coded in ``step()``.

Mirrors :class:`~hymeko_rl.env.reward.RewardSpec`: a kind→predicate Strategy registry + a small
bundle reader over the same narrow ``.hymeko`` parse.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hymeko_rl.env._profile import read_bundle
from hymeko_rl.env.safety import SafetyState

TerminationCond = Callable[[SafetyState], bool]


# kind -> predicate over the safety state. Mirrors meta_task.hymeko's `termination` vocab.
_TERMINATION_CONDS: dict[str, TerminationCond] = {
    "ground_contact": lambda s: s.ground_contact,
    "self_collision": lambda s: s.self_collision,
}


@dataclass(frozen=True)
class TerminationSpec:
    """An (unordered) set of death conditions → ``is_terminal`` = OR over their predicates.

    # Preconditions every kind is in :data:`_TERMINATION_CONDS`.
    # Postconditions an empty spec is never terminal; otherwise terminal iff any predicate holds.
    """

    conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = [c for c in self.conditions if c not in _TERMINATION_CONDS]
        if unknown:
            raise ValueError(
                f"unknown termination condition(s) {unknown}; known: {sorted(_TERMINATION_CONDS)}")

    def is_terminal(self, safety: SafetyState) -> bool:
        """True iff any declared condition holds against ``safety`` (a death)."""
        return any(_TERMINATION_CONDS[c](safety) for c in self.conditions)

    @classmethod
    def from_hymeko(cls, profile_path: str | Path) -> "TerminationSpec":
        """Build from a ``.hymeko`` task profile's ``termination_spec`` bundle.

        # Errors ``FileNotFoundError``; ``ValueError`` (no/multiple ``termination_spec``).
        """
        return cls(conditions=tuple(
            kind for _name, kind, _body, _w in read_bundle(profile_path, "termination_spec")))


# The default death predicate when a profile declares none — ground contact or self-collision
# (the conditions the env used before termination was nameable in the model).
DEATH_ON_CONTACT = TerminationSpec(("ground_contact", "self_collision"))
