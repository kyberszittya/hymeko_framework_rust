"""Deterministic mechanism factorization ``B ≈ A_out · Σ · A_inᵀ`` — LiNGAM-SH step 3A (spec §5).

Given the pairwise LiNGAM coefficient matrix ``B`` and a **fixed candidate set** of proposed mechanisms
(:class:`~hymeko_rl.eval.causal.mechanism_proposal.MechanismProposal` from step 2), evaluate the signed-incidence
factorization: each mechanism ``k`` is a rank-1 block ``A_out[:,k] · Σ[k,k] · A_in[:,k]ᵀ`` contributing its
strength·sign to every tail×head pairwise entry of ``B_hat``. **No search, no optimization, no discovery** — the
structure (``A_in``, ``A_out``) is the proposal support and ``Σ`` comes straight from the proposal scores. This is
the deterministic evaluation that Step 3B (group-sparse fit / selection) will optimize over.

Convention: ``B[effect, cause] = weight`` (rows = effects, cols = causes), matching DirectLiNGAM's ``adjacency``.
Overlapping mechanisms that touch the same ``(effect, cause)`` entry **sum** their contributions (spec §8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .hymeko_emit import CausalHypergraph
from .mechanism_proposal import MechanismProposal, proposals_to_causal_hypergraph


@dataclass(frozen=True, eq=False)
class MechanismFactorization:
    """The evaluated factorization ``B_hat = A_out · Σ · A_inᵀ`` over a fixed proposal set (eq disabled: arrays)."""

    variables: tuple[str, ...]
    mechanisms: tuple[MechanismProposal, ...]
    a_in: np.ndarray                    # (d, m): A_in[i,k] != 0 iff variable i is a TAIL of mechanism k
    a_out: np.ndarray                   # (d, m): A_out[j,k] != 0 iff variable j is a HEAD of mechanism k
    sigma: np.ndarray                   # (m, m) diagonal: Σ[k,k] = sign·strength (scaled if normalized)
    b_hat: np.ndarray                   # (d, d) reconstruction, B_hat[effect, cause]
    residual: np.ndarray                # (d, d) = B - B_hat
    metrics: Mapping[str, float]

    def to_causal_hypergraph(self, name: str) -> CausalHypergraph:
        """The mechanism-form :class:`CausalHypergraph` for this factorization's proposals (star-expand + verify)."""
        return proposals_to_causal_hypergraph(self.variables, self.mechanisms, name=name)


def build_pairwise_b(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]") -> np.ndarray:
    """Assemble the pairwise coefficient matrix with ``B[effect, cause] = weight`` (duplicates sum).

    # Preconditions every edge's cause/effect is in ``variables``. # Postconditions shape ``(d, d)``.
    """
    idx = {v: i for i, v in enumerate(variables)}
    d = len(variables)
    b = np.zeros((d, d), dtype=np.float64)
    for cause, effect, weight in pairwise_edges:
        b[idx[effect], idx[cause]] += float(weight)     # B[effect, cause]; overlapping edges accumulate
    return b


def _variable_universe(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]",
                       proposals: "Sequence[MechanismProposal]") -> list[str]:
    """Caller variables first, then any variable referenced by an edge or proposal (deterministic order)."""
    referenced = [v for c, e, _w in pairwise_edges for v in (c, e)]
    referenced += [v for p in proposals for v in (*p.tail, *p.head)]
    return list(dict.fromkeys([*variables, *referenced]))


def _build_factors(variables: "Sequence[str]", proposals: "Sequence[MechanismProposal]", *, normalize: bool,
                   ) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Build ``(A_in, A_out, Σ)``. ``normalize`` unit-scales the incidence columns (Σ absorbs the scale) — a pure
    representation choice; ``B_hat`` is identical either way."""
    idx = {v: i for i, v in enumerate(variables)}
    d, m = len(variables), len(proposals)
    a_in = np.zeros((d, m), dtype=np.float64)
    a_out = np.zeros((d, m), dtype=np.float64)
    sig = np.zeros(m, dtype=np.float64)
    for k, p in enumerate(proposals):
        scale = float(p.sign) * abs(float(p.strength))
        tnorm = np.sqrt(len(p.tail)) if normalize else 1.0
        hnorm = np.sqrt(len(p.head)) if normalize else 1.0
        for t in p.tail:
            a_in[idx[t], k] = 1.0 / tnorm
        for h in p.head:
            a_out[idx[h], k] = 1.0 / hnorm
        sig[k] = scale * tnorm * hnorm                  # (=scale when not normalized) → B_hat invariant
    return a_in, a_out, np.diag(sig)


def score_mechanism_set(b: np.ndarray, b_hat: np.ndarray, *, n_mechanisms: int,
                        n_parameters: int) -> dict[str, float]:
    """Reconstruction metrics of ``b_hat`` vs ``b`` (baseline ``b_hat = 0``).

    Returns ``fro_error``, ``fro_error_baseline`` (``‖B‖``), ``relative_error``, ``explained_energy``
    (``1 − ‖resid‖²/‖B‖²``), ``n_mechanisms``, ``n_parameters``, and a rough ``bic_like_score``
    (``n·ln(RSS/n) + k·ln(n)`` over the ``n = d²`` matrix entries).
    """
    resid = b - b_hat
    fro_error = float(np.linalg.norm(resid))
    fro_baseline = float(np.linalg.norm(b))
    b_energy = float(np.sum(b * b))
    resid_energy = float(np.sum(resid * resid))
    n = max(1, b.size)
    rss = max(resid_energy, 1e-12)
    bic_like = float(n * np.log(rss / n) + n_parameters * np.log(n))
    return {
        "fro_error": round(fro_error, 6),
        "fro_error_baseline": round(fro_baseline, 6),
        "relative_error": round(fro_error / fro_baseline, 6) if fro_baseline > 0 else 0.0,
        "explained_energy": round(1.0 - resid_energy / b_energy, 6) if b_energy > 0 else 0.0,
        "n_mechanisms": float(n_mechanisms),
        "n_parameters": float(n_parameters),
        "bic_like_score": round(bic_like, 6),
    }


def factorize_from_proposals(variables: "Sequence[str]", pairwise_edges: "Sequence[tuple[str, str, float]]",
                             proposals: "Sequence[MechanismProposal]", *, normalize: bool = True,
                             ) -> MechanismFactorization:
    """Evaluate ``B ≈ A_out · Σ · A_inᵀ`` for the fixed proposal set (deterministic; no fit/search).

    # Postconditions ``b_hat == a_out @ sigma @ a_in.T``; ``b_hat``'s nonzero support is the pairwise projection
      of the proposed mechanisms; overlapping mechanisms sum into shared entries.
    """
    universe = _variable_universe(variables, pairwise_edges, proposals)
    b = build_pairwise_b(universe, pairwise_edges)
    a_in, a_out, sigma = _build_factors(universe, proposals, normalize=normalize)
    b_hat = a_out @ sigma @ a_in.T
    residual = b - b_hat
    metrics = score_mechanism_set(b, b_hat, n_mechanisms=len(proposals), n_parameters=len(proposals))
    return MechanismFactorization(variables=tuple(universe), mechanisms=tuple(proposals), a_in=a_in, a_out=a_out,
                                  sigma=sigma, b_hat=b_hat, residual=residual, metrics=metrics)


__all__ = ["MechanismFactorization", "build_pairwise_b", "score_mechanism_set", "factorize_from_proposals"]
