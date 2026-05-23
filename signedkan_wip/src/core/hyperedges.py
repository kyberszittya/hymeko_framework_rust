"""SignedKAN — Phase 1.5–1.9: hyperedge construction by sign-balance triads.

Cartwright--Harary 1956 structural balance: a signed triangle
(v_i, v_j, v_k) with signs (s_ij, s_jk, s_ik) is *balanced* iff
the product s_ij · s_jk · s_ik = +1, i.e. the triangle has either
zero or two negative edges. Otherwise it is unbalanced.

Hyperedge construction:
  - Enumerate all triangles in the underlying signed graph
    (ignoring direction; symmetric edge set).
  - For each triangle, classify as balanced or unbalanced.
  - Apply the σ assignment rule (DECISIONS.md):
      apex vertex  →  σ = +1
      base vertices → σ = -1
    Apex defined as the vertex incident to both negative edges
    in an unbalanced triangle (or arbitrarily, the lowest-index
    vertex in a balanced triangle).
  - Emit a list of hyperedges, each carrying:
      (v_i, v_j, v_k, σ_i, σ_j, σ_k, balanced: bool)

Run:
    python3 -m src.hyperedges --dataset bitcoin_alpha
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..datasets import SignedGraph, load


@dataclass
class SignedTriad:
    v: tuple[int, int, int]            # vertex IDs in canonical order
    sigma: tuple[int, int, int]        # σ ∈ {+1, -1} per vertex
    edge_signs: tuple[int, int, int]   # signs of (v0v1, v1v2, v0v2)
    balanced: bool


def _adjacency(g: SignedGraph,
                directed: bool = False) -> dict[int, dict[int, int]]:
    """Return adjacency as a nested dict.

    ``directed=False`` (default): symmetrise — ``adj[u][v]`` AND
    ``adj[v][u]`` both set to ``sign(uv)``. If the same pair appears
    twice with conflicting signs, the most recent one wins (rare in
    Bitcoin data; tracked but not flagged).

    ``directed=True``: preserve the stored direction. Only ``adj[u][v]``
    is set for each stored edge ``(u, v)``. The reverse direction may
    or may not be present as a separate entry.
    """
    adj: dict[int, dict[int, int]] = defaultdict(dict)
    for (s, t), sg in zip(g.edges, g.signs):
        adj[int(s)][int(t)] = int(sg)
        if not directed:
            adj[int(t)][int(s)] = int(sg)
    return adj


def _enumerate_triangles(adj: dict[int, dict[int, int]]) -> list[tuple[int,int,int]]:
    """Triangles (i, j, k) with i < j < k. O(d_max² · |V|) for sparse
    Bitcoin-scale graphs; tractable on a laptop.

    Iteration is deterministic: outer loop sorted by `i`, inner
    over `sorted(nbrs[i])`. CPython set iteration is hash-randomised
    under PYTHONHASHSEED=random; sorting at every step makes the
    triad list reproducible across re-imports — a Phase 8 requirement
    for the m_per_vertex cap to be deterministic.
    """
    out: list[tuple[int,int,int]] = []
    nbrs = {v: set(d.keys()) for v, d in adj.items()}
    for i in sorted(adj):
        for j in sorted(nbrs[i]):
            if j <= i:
                continue
            common = nbrs[i] & nbrs[j]
            for k in sorted(common):
                if k <= j:
                    continue
                out.append((i, j, k))
    return out


def _classify(adj: dict[int, dict[int, int]],
              tri: tuple[int,int,int]) -> SignedTriad:
    i, j, k = tri
    s_ij = adj[i][j]
    s_jk = adj[j][k]
    s_ik = adj[i][k]
    balanced = (s_ij * s_jk * s_ik) == 1

    # σ assignment rule from DECISIONS.md.
    # In an UNBALANCED triangle, exactly one vertex sits between the two
    # negative edges — this is the apex. In a BALANCED triangle there is
    # no canonical apex; we pick i (lowest index) as the apex.
    if not balanced:
        # Find the vertex incident to both negative edges in {i, j, k}.
        # An unbalanced triangle has either 1 or 3 negative edges.
        edges = [(s_ij, i, j), (s_jk, j, k), (s_ik, i, k)]
        neg = [(u, v) for s, u, v in edges if s == -1]
        if len(neg) == 1:
            # Single negative edge ⇒ each endpoint is incident to exactly
            # one negative; apex is the THIRD vertex (incident to none).
            in_neg = set(neg[0])
            apex = (set([i, j, k]) - in_neg).pop()
        else:  # three negatives
            apex = i  # all three are equally apex; pick lowest index
    else:
        apex = i

    sigma_i = 1 if i == apex else -1
    sigma_j = 1 if j == apex else -1
    sigma_k = 1 if k == apex else -1
    return SignedTriad(
        v=(i, j, k),
        sigma=(sigma_i, sigma_j, sigma_k),
        edge_signs=(s_ij, s_jk, s_ik),
        balanced=balanced,
    )


def construct(
    g: SignedGraph,
    m_per_vertex: int | None = None,
) -> list[SignedTriad]:
    """Sign-balance triad construction from a signed graph.

    Parameters
    ----------
    g
        Signed graph to enumerate triangles over.
    m_per_vertex
        Optional cap on the number of triads kept *per apex vertex*.
        ``None`` (default) keeps every enumerated triad — back-compat
        with the original construction. ``m_per_vertex = M > 0``
        groups triads by apex (the vertex with $\\sigma = +1$ under
        :func:`_classify`) and keeps the first ``M`` triads per
        bucket in enumeration order. The ordering is deterministic
        (see :func:`_enumerate_triangles`).

        Wired in 2026-05-19 Phase 8 to give the HSIKAN P-graph
        framework's ``cycle_topk_m{4,16,64}`` axis real training-
        time effect: ``m_per_vertex`` is the per-anchor cycle-pool
        size the architecture-search picks.
    """
    adj = _adjacency(g)
    tris = _enumerate_triangles(adj)
    triads = [_classify(adj, t) for t in tris]
    if m_per_vertex is None or m_per_vertex <= 0:
        return triads
    # Group by apex, keep at most M per bucket in enumeration order.
    bucket: dict[int, int] = defaultdict(int)
    out: list[SignedTriad] = []
    for t in triads:
        apex_idx = t.sigma.index(1)
        apex = t.v[apex_idx]
        if bucket[apex] < m_per_vertex:
            bucket[apex] += 1
            out.append(t)
    return out


def stats(triads: list[SignedTriad]) -> dict:
    n = len(triads)
    if n == 0:
        return {"n_triads": 0, "balanced_frac": 0.0}
    n_bal = sum(1 for t in triads if t.balanced)
    return {
        "n_triads": n,
        "n_balanced": n_bal,
        "n_unbalanced": n - n_bal,
        "balanced_frac": n_bal / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha",
                    choices=["bitcoin_alpha", "bitcoin_otc"])
    args = ap.parse_args()
    g = load(args.dataset)
    print(f"\n{args.dataset}: {g.stats()}")
    print("  enumerating triangles...")
    triads = construct(g)
    s = stats(triads)
    print(f"  {s['n_triads']} triads  "
          f"({s['n_balanced']} balanced, {s['n_unbalanced']} unbalanced; "
          f"balanced_frac={s['balanced_frac']:.3f})")
    # Show first few triads as sanity check.
    print("\n  first 3 triads:")
    for t in triads[:3]:
        print(f"    {t}")


if __name__ == "__main__":
    main()
