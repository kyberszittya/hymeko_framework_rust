"""Task-independent semi-MDP option core.

An *option* is a temporally-extended action: from an initiation state the executor runs a committed policy until a valid
handoff / completion / abort, optionally continues through a frozen downstream skill to a terminal, and yields one
`OptionTransition`. The RL is semi-MDP: the Bellman target discounts the bootstrap by γ^τ (the option's duration), never a
single step. Nothing here knows about any concrete task — coin, grasp, navigation, etc. plug in via the Protocols.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


class OptionEnd(str, Enum):
    """How an option terminated — the handoff/completion/abort semantics, task-independent."""

    HANDOFF = "handoff"          # reached a valid handoff → (usually) frozen downstream skill → terminal
    COMPLETED = "completed"      # the committed macro finished without handoff → re-decide (recovery)
    ABORTED = "aborted"          # safety abort / irreversible failure / divergence → terminal
    TRUNCATED = "truncated"      # option/episode budget exhausted


@dataclass
class OptionTransition:
    """One semi-MDP option transition. ``action`` is the Bellman action (e.g. the proposal center); ``provenance`` carries
    the executed sub-choice (e.g. the search-selected candidate) and certificate — NEVER used as the Bellman action."""

    s: np.ndarray
    action: np.ndarray
    reward: float                # R_option = Σ_{j<τ} γ^j r_{t+j} (discounted return through the whole option)
    tau: float                   # option duration (steps) — the γ^τ exponent
    s_next: np.ndarray
    terminal: float              # 1.0 if no bootstrap after this option
    end: OptionEnd
    provenance: dict[str, Any] = field(default_factory=dict)


def smdp_target(reward, gamma: float, tau, terminal, q_next):
    """Semi-MDP Bellman target: ``R_option + (1−terminal)·γ^τ·Q_next``. Uses γ^τ (option duration), NOT one-step γ. Works on
    scalars or broadcastable tensors/arrays.

    # Preconditions: 0<gamma≤1; tau≥0; terminal∈{0,1}. # Postcondition: terminal ⇒ target==reward."""
    return reward + (1.0 - terminal) * (gamma ** tau) * q_next


@dataclass
class OptionReplayBuffer:
    """FIFO replay of `OptionTransition`s. Samples return tensors (s, a, r, tau, s2, terminal) for the semi-MDP loss; the
    provenance stays out of the Bellman batch (it is diagnostics, not an action)."""

    cap: int = 20000
    _s: list = field(default_factory=list); _a: list = field(default_factory=list); _r: list = field(default_factory=list)
    _tau: list = field(default_factory=list); _s2: list = field(default_factory=list); _term: list = field(default_factory=list)
    _prov: list = field(default_factory=list)

    def add(self, tr: OptionTransition):
        self._s.append(np.asarray(tr.s, np.float32)); self._a.append(np.asarray(tr.action, np.float32))
        self._r.append(float(tr.reward)); self._tau.append(float(tr.tau)); self._s2.append(np.asarray(tr.s_next, np.float32))
        self._term.append(float(tr.terminal)); self._prov.append(tr.provenance)
        if len(self._s) > self.cap:
            for buf in (self._s, self._a, self._r, self._tau, self._s2, self._term, self._prov):
                del buf[0]

    def __len__(self):
        return len(self._s)

    @property
    def provenance(self):
        return self._prov

    def sample(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self._s), n)

        def t(buf, dt):
            return torch.as_tensor(np.asarray([buf[i] for i in idx], dt))
        return (t(self._s, np.float32), t(self._a, np.float32), t(self._r, np.float32),
                t(self._tau, np.float32), t(self._s2, np.float32), t(self._term, np.float32))


@runtime_checkable
class OptionEnv(Protocol):
    """A task's option-level environment. ``reset`` → initiation observation; ``step(action)`` executes ONE option and
    returns (s_next, R_option, done, info) where info carries at least ``tau`` and any certificate/provenance the task
    exposes. Implementations own their physics; the engine only sees this interface."""

    def reset(self, idx: int | None = None) -> np.ndarray: ...

    def step(self, action: np.ndarray, *, search_seed: int | None = None) -> tuple[np.ndarray, float, bool, dict]: ...


@runtime_checkable
class HandoffCertificate(Protocol):
    """Task-defined success/handoff certificate over an executed option outcome (e.g. coin K6). Kept a Protocol so the
    engine never hard-codes a success rule."""

    def success(self, outcome: dict) -> bool: ...

    def handed_off(self, outcome: dict) -> bool: ...
