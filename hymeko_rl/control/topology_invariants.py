"""Graph invariants for the topology→control law (Phase 4). For each topology we compute the spectral and
combinatorial quantities that the isomorphic-controllers hypotheses say should predict control performance ---
algebraic connectivity / adjacency spectral gap (expansion), signed frame coherence (the ETF/Steiner result),
Z2 balance frustration (holonomy), and degree regularity. These become the regressors against the
``controller_bench`` MSE: which invariant predicts which plant's controllability.
"""
from __future__ import annotations

import itertools

import numpy as np

from hymeko_rl.agents.hypergraph_state import HypergraphState


def _adjacencies(hg: HypergraphState) -> "tuple[np.ndarray, np.ndarray]":
    """Unsigned ``A`` and signed ``S`` adjacency matrices ``(n, n)`` from the directed signed arcs."""
    n = hg.n_vertices
    a = np.zeros((n, n))
    s = np.zeros((n, n))
    for (u, v), sg in zip(hg.edges.tolist(), hg.signs.tolist()):
        a[int(u), int(v)] = a[int(v), int(u)] = 1.0
        s[int(u), int(v)] = s[int(v), int(u)] = float(sg)
    return a, s


def algebraic_connectivity(a: np.ndarray) -> float:
    """The Fiedler value lambda_2 of the unsigned Laplacian (graph connectivity / expansion floor)."""
    lap = np.diag(a.sum(1)) - a
    return float(np.sort(np.linalg.eigvalsh(lap))[1])


def adjacency_spectral_gap(a: np.ndarray) -> float:
    """lambda_1 - lambda_2 of the adjacency (the expander gap — large for Petersen/Ramanujan)."""
    ev = np.sort(np.linalg.eigvalsh(a))[::-1]
    return float(ev[0] - ev[1])


def frame_coherence(s: np.ndarray) -> float:
    """Max normalised inner product between the points' signed-adjacency rows (the ETF/Steiner measure;
    lower = tighter, more equiangular)."""
    norms = np.linalg.norm(s, axis=1)
    n = s.shape[0]
    coh = 0.0
    for i, j in itertools.combinations(range(n), 2):
        if norms[i] > 0 and norms[j] > 0:
            coh = max(coh, abs(float(s[i] @ s[j])) / (norms[i] * norms[j]))
    return coh


def balance_frustration(hg: HypergraphState) -> int:
    """Harary frustration index of the signed graph: min edges to flip for a balanced 2-colouring (Z2 holonomy
    trivial). Brute-force 2-colouring (n small). 0 = balanced."""
    n = hg.n_vertices
    if n > 18:                                            # 2^n brute-force infeasible (e.g. Kneser K(9,2)=36)
        return -1
    edges = {}
    for (u, v), sg in zip(hg.edges.tolist(), hg.signs.tolist()):
        edges[(min(int(u), int(v)), max(int(u), int(v)))] = int(sg)
    if not edges:
        return 0
    best = len(edges)
    for mask in range(1 << n):
        color = [(mask >> i) & 1 for i in range(n)]
        viol = sum(not ((sg > 0) == (color[i] == color[j])) for (i, j), sg in edges.items())
        best = min(best, viol)
        if best == 0:
            break
    return best


def degree_irregularity(a: np.ndarray) -> float:
    """Std of the degree sequence (0 = regular — Petersen/expander; large = star)."""
    return float(a.sum(1).std())


def invariants(hg: HypergraphState) -> "dict[str, float]":
    a, s = _adjacencies(hg)
    return {"n": float(hg.n_vertices),
            "alg_conn": round(algebraic_connectivity(a), 3),
            "spectral_gap": round(adjacency_spectral_gap(a), 3),
            "frame_coherence": round(frame_coherence(s), 3),
            "frustration": float(balance_frustration(hg)),
            "deg_irregularity": round(degree_irregularity(a), 3)}


def run_invariants(names: "list[str] | None" = None, *, n_nodes: int = 9, seed: int = 0) -> "dict[str, dict]":
    """Invariants for every topology in the zoo (designs included)."""
    from hymeko_rl.control.hypergraph_designs import DESIGNS
    from hymeko_rl.control.topology_zoo import TOPOLOGIES
    builders: "dict[str, object]" = {**{k: (lambda b=b: b(n_nodes, seed=seed)) for k, b in TOPOLOGIES.items()},
                                     **{k: b for k, b in DESIGNS.items()}}
    names = names or list(builders)
    out = {}
    for nm in names:
        try:
            out[nm] = invariants(builders[nm]())   # type: ignore[operator]
        except Exception as e:   # noqa: BLE001 — grid needs a perfect square etc.; skip with a note
            out[nm] = {"error": f"{type(e).__name__}: {str(e)[:40]}"}
    return out
