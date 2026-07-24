"""Task-independent proposal + fixed-budget local-search wrapper.

The deployed/RL controller is `state → learned proposal center → fixed-budget local search around it → selected committed
option`. The RL **action is the proposal center**; the search-selected candidate is recorded as *provenance*, never as the
Bellman action (that would break the Bellman semantics — the critic must value the proposal, whose stationary search
response is part of the environment). These Protocols let any task supply its own candidate space and score while reusing
the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ProposalPolicy(Protocol):
    """Maps an initiation observation to a single proposal center (a legal task action vector)."""

    def center(self, obs: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class CandidateGenerator(Protocol):
    """Samples ``n`` candidate options around a center (task-defined perturbation, e.g. structured jitter)."""

    def sample(self, center: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class CandidateScorer(Protocol):
    """Scores a candidate by rolling it (the frozen, predefined local-search score). Higher = better."""

    def score(self, candidate: np.ndarray, rng: np.random.Generator) -> tuple[float, dict]: ...


@dataclass
class SelectedActionProvenance:
    """What the fixed-budget search chose, kept separate from the Bellman action (the center)."""

    center: np.ndarray
    selected: np.ndarray
    score: float
    outcome: dict
    budget: int

    def as_dict(self) -> dict[str, Any]:
        return {"theta_center": np.asarray(self.center, np.float32), "theta_selected": np.asarray(self.selected, np.float32),
                "score": float(self.score), "budget": int(self.budget), **{k: self.outcome.get(k) for k in ("k6", "reached_handoff", "tau")}}


@dataclass
class FixedBudgetSearch:
    """Sample ``budget`` candidates around a center, keep the argmax-score one. ``budget==0`` executes the center directly
    (no rescue). The budget/generator/scorer are FROZEN across an RL run — the wrapper must be a stationary env response."""

    generator: CandidateGenerator
    scorer: CandidateScorer
    budget: int = 8

    def select(self, center: np.ndarray, rng: np.random.Generator) -> SelectedActionProvenance:
        center = np.asarray(center, np.float32)
        if self.budget <= 0:
            sc, out = self.scorer.score(center, rng)
            return SelectedActionProvenance(center, center, sc, out, 0)
        cands = self.generator.sample(center, self.budget, rng)
        best_i, best_sc, best_out = 0, -np.inf, {}
        for i, c in enumerate(cands):
            sc, out = self.scorer.score(c, rng)
            if sc > best_sc:
                best_i, best_sc, best_out = i, sc, out
        return SelectedActionProvenance(center, np.asarray(cands[best_i], np.float32), best_sc, best_out, self.budget)
