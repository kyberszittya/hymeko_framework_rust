"""Deterministic mechanism subset *selection* — LiNGAM-SH step 3B (spec §5/§7).

Given candidate :class:`~hymeko_rl.eval.causal.mechanism_proposal.MechanismProposal`\\s and the pairwise LiNGAM
matrix ``B``, pick the subset that best trades reconstruction (least-squares Σ, from
:func:`~hymeko_rl.eval.causal.mechanism_factorization.fit_sigma_least_squares`) against mechanism complexity.
**Deterministic only** — exhaustive over all subsets when small, else greedy forward selection; no stochastic
search, no discovery, no DirectLiNGAM change.

Selection score (documented, simple):

    selection_score = fro_error + complexity_penalty · n_parameters · ln(n_observed + 1) / max(n_observed, 1)

where ``n_observed`` is the number of non-zero entries of ``B``. Fewer mechanisms are preferred when the
reconstruction is comparable (the penalty rises with ``n_parameters``); the empty subset is a valid baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Mapping, Sequence

import numpy as np

from .mechanism_factorization import (
    MechanismFactorization,
    _variable_universe,
    build_pairwise_b,
    fit_sigma_least_squares,
)
from .mechanism_proposal import MechanismProposal


@dataclass(frozen=True)
class MechanismSelectionResult:
    """The selected mechanism subset, the rejected candidates, and the fitted factorization over the selection."""

    variables: tuple[str, ...]
    selected: tuple[MechanismProposal, ...]
    rejected: tuple[MechanismProposal, ...]
    factorization: MechanismFactorization
    candidate_scores: Mapping[str, Mapping[str, float]]
    selection_score: float
    method: str


def _n_observed(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]",
                proposals: "Sequence[MechanismProposal]") -> int:
    """Number of non-zero entries of ``B`` (invariant to zero-padding by extra variables)."""
    universe = _variable_universe(variables, pairwise_edges, proposals)
    return int(np.count_nonzero(build_pairwise_b(universe, pairwise_edges)))


def _selection_score(fro_error: float, n_parameters: int, n_observed: int, complexity_penalty: float) -> float:
    """``fro_error + penalty · n_parameters · ln(n_observed+1) / max(n_observed, 1)`` (see module docstring)."""
    return float(fro_error + complexity_penalty * n_parameters * np.log(n_observed + 1) / max(n_observed, 1))


def _score_subset(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]",
                  subset: "Sequence[MechanismProposal]", complexity_penalty: float, n_observed: int,
                  ) -> "tuple[float, MechanismFactorization]":
    fac = fit_sigma_least_squares(variables, pairwise_edges, subset)
    return _selection_score(fac.metrics["fro_error"], len(subset), n_observed, complexity_penalty), fac


def _candidate_scores(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]",
                      proposals: "Sequence[MechanismProposal]") -> dict[str, dict[str, float]]:
    """Per-candidate diagnostics: the single-mechanism least-squares fit (fitted σ, sign match, error alone)."""
    scores: dict[str, dict[str, float]] = {}
    for p in proposals:
        fac = fit_sigma_least_squares(variables, pairwise_edges, [p])
        sigma = float(fac.sigma[0, 0]) if fac.sigma.size else 0.0
        scores[p.name] = {
            "fitted_sigma": round(sigma, 6), "proposal_sign": float(p.sign),
            "sign_match": float(sigma == 0.0 or int(np.sign(sigma)) == p.sign),
            "fro_error_alone": fac.metrics["fro_error"], "explained_energy_alone": fac.metrics["explained_energy"]}
    return scores


def _select_exhaustive(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]",
                       proposals: "Sequence[MechanismProposal]", complexity_penalty: float, n_observed: int,
                       ) -> tuple[int, ...]:
    """Test every subset (including empty); pick min score, tie-break to fewer mechanisms then lower index."""
    m = len(proposals)
    best_key: "tuple[float, int, tuple[int, ...]] | None" = None
    best_combo: tuple[int, ...] = ()
    for r in range(m + 1):
        for combo in combinations(range(m), r):
            score, _ = _score_subset(variables, pairwise_edges, [proposals[i] for i in combo],
                                     complexity_penalty, n_observed)
            key = (round(score, 9), len(combo), combo)
            if best_key is None or key < best_key:
                best_key, best_combo = key, combo
    return best_combo


def _select_greedy(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]",
                   proposals: "Sequence[MechanismProposal]", complexity_penalty: float, n_observed: int,
                   ) -> tuple[int, ...]:
    """Deterministic forward selection: add the improving candidate with the lowest resulting score; stop at none."""
    selected: list[int] = []
    remaining = list(range(len(proposals)))
    cur_score, _ = _score_subset(variables, pairwise_edges, [], complexity_penalty, n_observed)
    while remaining:
        best_key: "tuple[float, int] | None" = None
        for i in remaining:
            score, _ = _score_subset(variables, pairwise_edges, [proposals[j] for j in (*selected, i)],
                                     complexity_penalty, n_observed)
            if best_key is None or (round(score, 9), i) < best_key:
                best_key = (round(score, 9), i)
        if best_key is None or best_key[0] >= round(cur_score, 9):
            break
        selected.append(best_key[1])
        remaining.remove(best_key[1])
        cur_score = best_key[0]
    return tuple(selected)


def select_mechanism_subset(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]",
                            proposals: "Sequence[MechanismProposal]", *,
                            method: 'Literal["exhaustive", "greedy"]' = "greedy", max_exhaustive: int = 12,
                            complexity_penalty: float = 1.0) -> MechanismSelectionResult:
    """Select the mechanism subset that minimizes the selection score (see module docstring).

    Exhaustive when ``method == "exhaustive"`` and ``len(proposals) <= max_exhaustive`` (all subsets, empty
    included); otherwise deterministic greedy forward selection. # Postconditions the returned ``factorization``
    is the least-squares fit over ``selected``; ``selected`` may be empty (baseline).
    """
    n_observed = _n_observed(variables, pairwise_edges, proposals)
    if method == "exhaustive" and len(proposals) <= max_exhaustive:
        chosen = _select_exhaustive(variables, pairwise_edges, proposals, complexity_penalty, n_observed)
    else:
        chosen = _select_greedy(variables, pairwise_edges, proposals, complexity_penalty, n_observed)
    selected = tuple(proposals[i] for i in chosen)
    rejected = tuple(proposals[i] for i in range(len(proposals)) if i not in set(chosen))
    score, fac = _score_subset(variables, pairwise_edges, list(selected), complexity_penalty, n_observed)
    return MechanismSelectionResult(
        variables=tuple(fac.variables), selected=selected, rejected=rejected, factorization=fac,
        candidate_scores=_candidate_scores(variables, pairwise_edges, proposals),
        selection_score=round(score, 6), method=method)


__all__ = ["MechanismSelectionResult", "select_mechanism_subset"]
