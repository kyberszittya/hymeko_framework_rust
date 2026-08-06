"""Semantic actor command + embodiment adapter contract for the coin-delivery scenario framework.

A :class:`SemanticCommand` is the *embodiment-independent* per-step intent (where to push, how much to
preload, whether to release / recontact, and — for a distal-pad embodiment — how to orient each pad). A
:class:`KinematicAdapter` maps that intent onto a concrete embodiment's action vector; the mapping is where
an embodiment declares what it can and cannot honor. The **canonical K0** embodiment cannot honor pad
orientation, and it says so explicitly (:attr:`KinematicMapping.rejected`) rather than silently folding the
command into the cooperative action.

The authoritative delivery rollout still drives the FROZEN :class:`~hymeko_rl.train.coin_delivery_actor.DeliveryActor`
through the frozen monitor; this semantic layer is a parallel, inspectable representation (used by the demo,
the K1 pad-orientation extension, and the adapter tests). :class:`DeliveryActorSemantic` wraps a frozen actor
so the round-trip ``K0Adapter.map_command(actor.command(inner, t)).action`` reproduces ``actor_action`` exactly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from hymeko_rl.train.coin_delivery_actor import ActorParams, DeliveryActor, actor_action

BASE_ACTION_DIM = 6   # the frozen 6-DoF cooperative action a=[mid_x, mid_y, aperture, squeeze, differential, rotate]


@dataclass(frozen=True)
class SemanticCommand:
    """One step of embodiment-independent delivery intent.

    # Preconditions ``push_direction`` is a finite 2-vector; ``aperture``/``contact_preload``/``differential``/
      ``rotate`` are finite scalars; pad orientations are finite scalars or ``None`` (no pad command).
    # Postconditions immutable; ``as_dict`` is JSON-serializable.
    # Invariants the six cooperative fields are sufficient to reconstruct a frozen actor's 6-DoF action."""

    push_direction: tuple[float, float]
    aperture: float
    contact_preload: float                       # squeeze of both tips toward the coin
    differential: float = 0.0                    # left/right balance shift
    rotate: float = 0.0                          # fingertip-axis rotation
    approach_direction: "tuple[float, float] | None" = None
    left_pad_orientation: "float | None" = None
    right_pad_orientation: "float | None" = None
    release: bool = False
    recontact: bool = False

    @property
    def pad_orientation_commanded(self) -> bool:
        return self.left_pad_orientation is not None or self.right_pad_orientation is not None

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["approach_direction"] = self.approach_direction or self.push_direction
        return d


@runtime_checkable
class SemanticActor(Protocol):
    """A per-step semantic policy. ``delivery_actor`` names the FROZEN actor used by the authoritative monitor."""

    name: str
    delivery_actor: DeliveryActor

    def command(self, inner: Any, t: int) -> SemanticCommand: ...


@dataclass(frozen=True)
class KinematicMapping:
    """Result of mapping a :class:`SemanticCommand` onto an embodiment.

    ``action`` is the embodiment's action vector; ``rejected`` names the semantic fields the embodiment could
    NOT honor (dropped explicitly, never silently reinterpreted). # Postconditions ``action`` is 1-D float32."""

    action: np.ndarray
    rejected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", np.asarray(self.action, dtype=np.float32).reshape(-1))


class KinematicAdapter(ABC):
    """Maps a :class:`SemanticCommand` to a concrete embodiment's action vector."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Length of the embodiment action vector."""

    @abstractmethod
    def map_command(self, cmd: SemanticCommand) -> KinematicMapping:
        """Map ``cmd`` to this embodiment. # Postconditions ``result.action`` has length :attr:`action_dim`."""

    @staticmethod
    def _cooperative(cmd: SemanticCommand) -> np.ndarray:
        """The frozen 6-DoF cooperative action encoded by ``cmd`` (embodiment-independent core)."""
        return np.array([cmd.push_direction[0], cmd.push_direction[1], cmd.aperture,
                         cmd.contact_preload, cmd.differential, cmd.rotate], dtype=np.float32)


@dataclass(frozen=True)
class DeliveryActorSemantic:
    """Wrap a FROZEN :class:`DeliveryActor` as a :class:`SemanticActor`, decoding its 6-DoF action into a
    :class:`SemanticCommand`. The decoding is exact, so ``K0Adapter.map_command(command(inner, t)).action``
    equals ``actor_action(inner, t, delivery_actor, params)`` — the existing actors run unchanged."""

    name: str
    delivery_actor: DeliveryActor
    params: ActorParams = field(default_factory=ActorParams)

    def command(self, inner: Any, t: int) -> SemanticCommand:
        a = actor_action(inner, t, self.delivery_actor, self.params)
        return SemanticCommand(push_direction=(float(a[0]), float(a[1])), aperture=float(a[2]),
                               contact_preload=float(a[3]), differential=float(a[4]), rotate=float(a[5]),
                               approach_direction=(float(a[0]), float(a[1])),
                               release=bool(a[3] < 0.0))


# ── actor registry (semantic wrappers over the frozen delivery actors) ─────────────────────────────────────────────
_ACTOR_SPECS: tuple[tuple[str, DeliveryActor], ...] = (
    ("symmetric_push", DeliveryActor.A0_SYM_PUSH),
    ("v_plow", DeliveryActor.A1_VPLOW),
    ("asymmetric_push", DeliveryActor.A2_ASYM_PUSH),
    ("setup_push", DeliveryActor.A3_SETUP_PUSH),
    ("recovery", DeliveryActor.A4_RECOVERY),
    ("neutral", DeliveryActor.A5_NEUTRAL),
)


def build_actor_registry() -> dict[str, DeliveryActorSemantic]:
    """The named semantic actors available to every scenario (the delivery baseline ``neutral`` included)."""
    return {name: DeliveryActorSemantic(name=name, delivery_actor=actor) for name, actor in _ACTOR_SPECS}


def cooperative_scramble(action: np.ndarray, t: int) -> np.ndarray:
    """Structural correct-vs-scramble control for the cooperative action: rotate the commanded midpoint
    direction 90° (destroys the toward-zone push structure at matched magnitude). Deterministic in ``(action, t)``
    so paired seeds compare like-for-like. # Postconditions ``len(result) == len(action)`` (capacity-matched)."""
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    a[0], a[1] = a[1], -a[0]
    return a


def orientation_scramble(action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Capacity-matched pad-orientation scramble: permute + randomly re-sign the pad-orientation tail while
    preserving the cooperative head AND the vector length (a K0 action has no tail → a no-op of equal length).

    # Preconditions ``action`` is 1-D of length >= :data:`BASE_ACTION_DIM`.
    # Postconditions ``len(result) == len(action)`` (the scramble is capacity-matched)."""
    a = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    tail = a[BASE_ACTION_DIM:]
    if tail.size:
        perm = rng.permutation(tail.size)
        signs = np.sign(rng.uniform(-1.0, 1.0, tail.size)).astype(np.float32)
        a[BASE_ACTION_DIM:] = tail[perm] * signs
    return a
