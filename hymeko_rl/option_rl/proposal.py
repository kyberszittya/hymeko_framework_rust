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


# ─────────────────────────── multimodal proposal + policy search (ARCHITECTURAL_ASSIMILATION_V1) ───────────────────────────
# O2 (square/rectangle, fresh-reconstruct) established MULTIMODAL_OPTION_STRUCTURE_DOMINANT: near-identical states have
# DISTANT winning options that one deterministic MSE centre averages into a weak guess (obs+geom did not help; within-shape
# nearest-obs θ-distance 2.88). The fix is a proposal that emits SEVERAL strategy modes and a search that explores across
# AND within them. The single-centre `ProposalPolicy`/`FixedBudgetSearch` above are the K=1 special case (kept, unchanged).


@dataclass
class ProposalMode:
    """One strategy mode: a probability weight, a centre (a legal task action), an optional per-mode candidate spread, and
    an id. A `MultimodalProposalPolicy` emits a list of these; the single-θ proposal is one mode with prob 1."""

    prob: float
    center: np.ndarray
    std: "np.ndarray | float | None" = None      # per-mode spread; None ⇒ the generator's own default
    mode_id: int = 0


@runtime_checkable
class MultimodalProposalPolicy(Protocol):
    """Maps an observation to a LIST of proposal modes (K distinct strategy centres with probabilities). Task-independent;
    the O2-justified replacement for a single deterministic centre. K=1 recovers `ProposalPolicy`."""

    def modes(self, obs: np.ndarray) -> "list[ProposalMode]": ...


@dataclass
class SingleModeProposal:
    """Adapt any single-centre `ProposalPolicy` to the multimodal interface (K=1) — so existing proposals plug into the
    multimodal search unchanged, and a task can migrate one head at a time."""

    policy: ProposalPolicy

    def modes(self, obs: np.ndarray) -> "list[ProposalMode]":
        return [ProposalMode(1.0, np.asarray(self.policy.center(obs), np.float32), None, 0)]


@dataclass
class MultimodalProvenance:
    """What the multimodal search chose: the proposed modes, the per-mode budget split, the selected mode + action. The
    Bellman action for RL is the SELECTED MODE's centre (provenance keeps the concrete selected candidate separate)."""

    mode_centers: "list[np.ndarray]"
    mode_probs: "list[float]"
    selected_mode: int
    selected: np.ndarray
    score: float
    outcome: dict
    budget: int
    per_mode_budget: "list[int]"

    def as_dict(self) -> "dict[str, Any]":
        return {"selected_mode": int(self.selected_mode), "n_modes": len(self.mode_centers),
                "mode_probs": [round(float(p), 3) for p in self.mode_probs],
                "theta_selected": np.asarray(self.selected, np.float32), "score": float(self.score),
                "budget": int(self.budget), "per_mode_budget": list(self.per_mode_budget),
                **{k: self.outcome.get(k) for k in ("k6", "reached_handoff", "tau")}}


def allocate_budget(probs: "list[float]", budget: int) -> "list[int]":
    """Split a total ``budget`` across K modes: ≥1 candidate each (so no mode is silently dropped), the remainder ∝ prob.
    # Preconditions ``budget ≥ 0``; probs non-negative. # Postconditions ``sum == max(budget, K)`` when budget≥K else 1 each
    for the top-budget modes. Deterministic (largest-remainder)."""
    k = len(probs)
    if k == 0:
        return []
    if budget <= 0:
        return [0] * k
    if budget <= k:                                          # too small to give every mode one — fund the top-prob modes
        order = sorted(range(k), key=lambda i: -probs[i])
        out = [0] * k
        for i in order[:budget]:
            out[i] = 1
        return out
    base = [1] * k                                           # ≥1 each
    rem = budget - k
    p = np.asarray(probs, np.float64)
    p = p / p.sum() if p.sum() > 0 else np.full(k, 1.0 / k)
    exact = p * rem
    add = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - add))[: rem - int(add.sum())]:   # largest-remainder rounding
        add[i] += 1
    return [int(b + a) for b, a in zip(base, add)]


@dataclass
class MultimodalBudgetSearch:
    """Generalises `FixedBudgetSearch` to K modes: allocate the total budget across the proposal's modes (∝ prob, ≥1 each),
    generate candidates around EACH mode centre (with that mode's spread), score all, keep the global argmax, and record
    which mode won. budget / generator / scorer / allocation are FROZEN across a run (stationary env response — the same
    Bellman-safety contract as the single-centre search). This IS the task-independent `PolicySearch` the tasks bind to."""

    generator: CandidateGenerator
    scorer: CandidateScorer
    budget: int = 8

    def select(self, proposal: MultimodalProposalPolicy, obs: np.ndarray, rng: np.random.Generator) -> MultimodalProvenance:
        modes = list(proposal.modes(np.asarray(obs, np.float32)))
        if not modes:
            raise ValueError("MultimodalProposalPolicy.modes returned no modes")
        centers = [np.asarray(m.center, np.float32) for m in modes]
        probs = [float(max(0.0, m.prob)) for m in modes]
        per_mode = allocate_budget(probs, self.budget)
        best = (0, np.asarray(centers[0], np.float32), -np.inf, {})   # (mode, action, score, outcome)
        for mi, (c, n) in enumerate(zip(centers, per_mode)):
            if n <= 0:
                continue
            if n == 1:                                        # a single candidate = the mode centre itself (no wasted jitter)
                cand = c[None, :]
            else:
                cand = self.generator.sample(c, n, rng)       # score at the generator's native precision (K=1 ≡ FixedBudgetSearch)
            for cc in cand:
                sc, out = self.scorer.score(cc, rng)
                if sc > best[2]:
                    best = (mi, np.asarray(cc, np.float32), sc, out)
        return MultimodalProvenance(centers, probs, best[0], best[1], best[2], best[3], self.budget, per_mode)
