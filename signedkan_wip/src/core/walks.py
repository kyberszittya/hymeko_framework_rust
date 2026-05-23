"""Walk-HSiKAN — open-walk constructor analogous to ``construct_k`` /
``construct_2`` in :mod:`n_tuples`.

A length-$L$ simple walk is a sequence $(v_0, v_1, \\ldots, v_L)$ of
$L{+}1$ distinct vertices with each consecutive pair an edge in $G$.
Per-vertex σ assignment matches the cycle convention:

  $\\sigma_i = (-1)^{n_i^-}$

where $n_i^-$ is the number of *negative walk-edges incident to vertex
$i$ within the walk*.  For interior vertices ($0 < i < L$) this is at
most $2$; for endpoints ($i \\in \\{0, L\\}$) it is at most $1$.

The output ``SignedNTuple`` is the same dataclass cycles use, so the
existing ``MixedAritySignedKAN`` / ``SignedKANLayer`` consume walks
unchanged: the model is structurally agnostic between cycles and
walks, the difference is only in what topological structure feeds it.

Walk-HSiKAN is the architectural fit for **walk-rich** signed graphs
(Slashdot, Epinions) where cycles are too rare or too sample-starved
for the cycle bias to help.
"""
from __future__ import annotations

import numpy as np

from ..datasets import SignedGraph
from .n_tuples import SignedNTuple


def _build_sign_lookup(g: SignedGraph) -> dict[tuple[int, int], int]:
    """``(min(u,v), max(u,v)) → sign`` table from the signed graph.

    Used to look up the sign of each walk-edge during σ-assignment.
    O(1) per query, O($|E|$) memory.
    """
    out: dict[tuple[int, int], int] = {}
    for i in range(len(g.edges)):
        u = int(g.edges[i, 0])
        v = int(g.edges[i, 1])
        s = int(g.signs[i])
        key = (min(u, v), max(u, v))
        out[key] = s
    return out


def _classify_walk(walk: tuple[int, ...],
                    sign_of: dict[tuple[int, int], int],
                    ) -> SignedNTuple | None:
    """Compute per-vertex σ + balance flag for one walk.

    Returns ``None`` if the walk references an edge not in the graph
    (shouldn't happen for walks built via ``enumerate_k_walks_rs`` on
    the same graph, but be defensive).
    """
    L_plus_1 = len(walk)
    L = L_plus_1 - 1
    edge_signs: list[int] = []
    for j in range(L):
        u, v = int(walk[j]), int(walk[j + 1])
        key = (min(u, v), max(u, v))
        s = sign_of.get(key)
        if s is None:
            return None
        edge_signs.append(s)
    # Per-vertex negative-incident counts (within the walk).
    neg_counts = [0] * L_plus_1
    for j in range(L):
        if edge_signs[j] == -1:
            neg_counts[j]     += 1
            neg_counts[j + 1] += 1
    # σ_i = +1 iff incident negative-count is even (incl. 0).
    sigma = tuple(1 if (c % 2) == 0 else -1 for c in neg_counts)
    n_neg = sum(1 for s in edge_signs if s == -1)
    balanced = (n_neg % 2) == 0   # Davis-style — balanced iff even negs
    return SignedNTuple(
        v=tuple(int(x) for x in walk),
        sigma=sigma,
        edge_signs=tuple(edge_signs),
        balanced=balanced,
        arity=L_plus_1,         # walk has L+1 vertices
    )


def construct_walks(g: SignedGraph, walk_len: int,
                     max_walks: int | None = None,
                     seed: int = 0) -> list[SignedNTuple]:
    """Build the list of length-$L$ simple walks for the signed graph.

    walk_len = $L$ ⇒ each walk is a tuple of $L{+}1$ vertices.
    Returns ``list[SignedNTuple]`` with ``arity = L+1`` so the
    existing ``construct_k``-aware code path consumes them unchanged.

    Uses ``hymeko.enumerate_k_walks_rs`` (Rust DFS, canonical-form
    emission) for enumeration; falls back to a pure-Python DFS if
    the Rust binding is not available.
    """
    if walk_len < 1:
        raise ValueError(f"walk_len must be >= 1, got {walk_len}")

    sign_of = _build_sign_lookup(g)

    try:
        import hymeko  # type: ignore
        if hasattr(hymeko, "enumerate_k_walks_rs"):
            eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.uint32)
            ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.uint32)
            arr = hymeko.enumerate_k_walks_rs(
                eu.tolist(), ev.tolist(), g.n_nodes,
                int(walk_len),
                int(max_walks) if max_walks else None,
                int(seed),
            )
            walks = [tuple(row) for row in arr.tolist()]
        else:
            walks = _python_walks(g, walk_len)
    except ImportError:
        walks = _python_walks(g, walk_len)

    if max_walks is not None and len(walks) > max_walks:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(walks), size=max_walks, replace=False)
        walks = [walks[int(i)] for i in idx]

    out: list[SignedNTuple] = []
    for w in walks:
        nt = _classify_walk(w, sign_of)
        if nt is not None:
            out.append(nt)
    return out


def _python_walks(g: SignedGraph, walk_len: int):
    """Pure-Python fallback for the walk enumerator (small graphs only).

    Same canonicalisation as :func:`hymeko.enumerate_k_walks_rs`:
    only emit walks with ``walk[0] <= walk[-1]``.
    """
    n = g.n_nodes
    adj: dict[int, list[int]] = {}
    for i in range(len(g.edges)):
        u = int(g.edges[i, 0]); v = int(g.edges[i, 1])
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    out: list[tuple[int, ...]] = []

    def dfs(path: list[int], visited: set[int]):
        if len(path) == walk_len + 1:
            if path[0] <= path[-1]:
                out.append(tuple(path))
            return
        tail = path[-1]
        for nxt in sorted(adj.get(tail, [])):
            if nxt in visited:
                continue
            path.append(nxt); visited.add(nxt)
            dfs(path, visited)
            path.pop(); visited.remove(nxt)

    for s in range(n):
        dfs([s], {s})
    return out
