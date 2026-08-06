"""Affective component — let an external human or agent shape the reward **at runtime**.

The reward is a declarative ``Σ weight·term`` over the env state (:mod:`hymeko_rl.env.reward`). This module
adds a runtime **valence** signal ``v ∈ [-1, 1]`` (approval … disapproval) supplied by a pluggable
:class:`AffectiveSource`, and two ways to couple it in — chosen per task, both honoured:

  * **Additive term** — the ``affective`` reward term (in the registry) returns ``v``; declared in a
    ``.hymeko`` like any other term (``@approval: rew.affective { weight 2.0; }``) → adds ``weight·v``.
  * **Modulator** — :class:`AffectiveModulator` wraps a :class:`RewardSpec` and scales it live:
    ``(1 + gain·v)·Σ weight·term`` → the human/agent reshapes *how much the whole task reward counts*.

The source is **injected** into the env (Strategy/Observer), never a global — the only way a runtime-mutable
reward stays testable and thread-safe (CLAUDE.md §6.5 #11). The env polls it once per step into ``_affect``;
the additive term and the modulator both read that.

Representation is a scalar valence today (extensible to valence+arousal); ``poll`` is given the live env so an
agentic source can read state. Sources:

  * :class:`NeutralAffect` — always 0 (default; the feature is inert unless a source is supplied).
  * :class:`HumanAffect`   — a thread-safe channel a human writes at runtime (key / slider / socket / file).
  * :class:`AgentAffect`   — wraps a callable/model (another agent, an LLM-judge, a discriminator).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hymeko_rl.env.reward import RewardSpec


def _clamp_valence(v: float) -> float:
    """Clamp to the valence range ``[-1, 1]``. # Postconditions ``-1 <= result <= 1``."""
    return -1.0 if v < -1.0 else 1.0 if v > 1.0 else float(v)


@runtime_checkable
class AffectiveSource(Protocol):
    """A runtime source of affective valence ``v ∈ [-1, 1]`` (Strategy/Observer).

    # Postconditions ``poll`` returns a finite valence in ``[-1, 1]``; it may read ``env`` (an agentic
    source scores the live state) but must not mutate it."""

    def poll(self, env: Any) -> float: ...


class NeutralAffect:
    """The inert default: no affect. # Postconditions always returns ``0.0``."""

    def poll(self, env: Any) -> float:
        return 0.0


class HumanAffect:
    """A thread-safe valence channel a human writes at runtime (key press, slider, socket, file-tail).

    The training/rollout thread calls :meth:`poll`; an external thread calls :meth:`set` / :meth:`nudge`.
    An optional per-poll ``decay`` lets a transient approval fade toward 0 (set ``decay=1.0`` to hold the
    last value until changed).

    # Preconditions ``0 < decay <= 1``. # Invariants the stored valence stays in ``[-1, 1]``.
    """

    def __init__(self, *, decay: float = 1.0, initial: float = 0.0) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        self._lock = threading.Lock()
        self._v = _clamp_valence(initial)
        self._decay = float(decay)

    def set(self, valence: float) -> None:
        """Set the current valence (clamped to ``[-1, 1]``). Thread-safe."""
        with self._lock:
            self._v = _clamp_valence(valence)

    def nudge(self, delta: float) -> None:
        """Add to the current valence (clamped). Thread-safe — e.g. a ``+``/``-`` key."""
        with self._lock:
            self._v = _clamp_valence(self._v + delta)

    def poll(self, env: Any) -> float:
        with self._lock:
            v = self._v
            self._v *= self._decay
            return v


class AgentAffect:
    """An agentic affect source: another agent/model produces the valence from the live env state.

    Wraps any ``score(env) -> float`` — a learned critic, a discriminator, or an LLM-judge adapter — and
    clamps the output to ``[-1, 1]``. Ties directly into the shared-agent-reward-model line: the scorer can
    be *another policy* rating this one.

    # Preconditions ``score`` is callable and returns a real number.
    """

    def __init__(self, score: Callable[[Any], float]) -> None:
        self._score = score

    def poll(self, env: Any) -> float:
        return _clamp_valence(float(self._score(env)))


@dataclass(frozen=True)
class AffectiveModulator:
    """Runtime affective **modulation** of a task reward: ``(1 + gain·v)·base``.

    Wraps a :class:`RewardSpec` (or any object with ``evaluate(env, dist, action) -> float``) and scales it by
    the env's current valence ``v`` (set by the env from its :class:`AffectiveSource`). ``gain·v > 0``
    amplifies how much the whole task reward counts (approval), ``< 0`` attenuates it (disapproval). Drop-in
    for an env's ``reward_spec`` — same ``evaluate`` signature.

    Compose with the additive ``affective`` term for the "both" coupling: wrap a spec that already declares
    ``affective`` → the approval bonus is the additive nudge, the modulation amplifies the rest.

    # Preconditions ``base`` exposes ``evaluate(env, dist, action)``. # Invariants the gain factor
    ``1 + gain·v`` is clamped to ``>= 0`` so the reward never flips sign from modulation alone.
    """

    base: "RewardSpec"
    gain: float = 1.0

    def evaluate(self, env: Any, dist: float, action: Any) -> float:
        v = float(getattr(env, "_affect", 0.0))
        factor = max(0.0, 1.0 + self.gain * v)
        return factor * self.base.evaluate(env, dist, action)
