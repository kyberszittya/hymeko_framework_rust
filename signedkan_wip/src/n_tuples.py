"""N-tuple (k-uniform) hyperedge construction for signed graphs.

Generalises the 3-uniform triad construction to $k$-uniform
hyperedges using **Davis 1967 weakly-balanced** $k$-cycles:

    A signed $k$-cycle is *balanced* iff the number of negative
    edges in the cycle is even.

Per-vertex $\\sigma$ assignment generalises the triad apex rule:

    σ_i = +1  if vertex $i$ is incident to an EVEN number of negative
             edges in the cycle; -1 otherwise.

For $k=3$ this reduces exactly to the existing apex rule
(unbalanced triad: apex is incident to two negatives → σ=+1).

Returned ``SignedNTuple`` objects are a strict superset of
``SignedTriad`` — the same training pipeline can consume mixed-arity
sets by stacking the same-arity bins separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .datasets import SignedGraph
from .hyperedges import _adjacency


@dataclass
class SignedNTuple:
    """A signed $k$-uniform hyperedge.

    v          : vertex IDs, length $k$, sorted ascending
    sigma      : per-vertex sign $\\sigma_i \\in \\{+1, -1\\}$, length $k$
    edge_signs : signs of the $k$ edges of the cycle
                 (in the same canonical-edge ordering used to compute
                 balance)
    balanced   : True iff product of edge_signs is +1
    arity      : len(v) — useful for filtering by k
    """
    v: tuple[int, ...]
    sigma: tuple[int, ...]
    edge_signs: tuple[int, ...]
    balanced: bool
    arity: int


def _cycle_edges(adj: dict[int, dict[int, int]],
                 cycle: tuple[int, ...]) -> tuple[int, ...] | None:
    """Edge signs of a candidate cycle, in cycle-order
    $(v_0, v_1), (v_1, v_2), \\ldots, (v_{k-1}, v_0)$.

    Returns None if the cycle has any non-edge.
    """
    k = len(cycle)
    out = []
    for i in range(k):
        u, v = cycle[i], cycle[(i + 1) % k]
        if v not in adj.get(u, {}):
            return None
        out.append(adj[u][v])
    return tuple(out)


def enumerate_k_cycles(adj: dict[int, dict[int, int]],
                        k: int) -> list[tuple[int, ...]]:
    """Enumerate all unique chordless or non-chordless $k$-cycles.

    Uses a DFS from each starting vertex with rooted-cycle
    canonicalisation: starting from the smallest vertex of the cycle,
    the second vertex must be the smaller of the two neighbours in
    the cycle. This emits each cycle exactly twice (clockwise +
    counter-clockwise from the same root); we deduplicate via a
    canonical sorted-rotation hash.

    Complexity: $O(|E| \\cdot d^{k-1})$, tractable for $k \\leq 5$
    on Bitcoin-scale fixtures.
    """
    if k < 3:
        return []
    nbrs = {v: sorted(d.keys()) for v, d in adj.items()}

    def _canonical(cycle: tuple[int, ...]) -> tuple[int, ...]:
        """Canonical form: rotate so smallest-index is first; if both
        directions are valid, pick the lexicographically smaller."""
        kk = len(cycle)
        rmin = min(range(kk), key=lambda i: cycle[i])
        fwd = tuple(cycle[(rmin + j) % kk] for j in range(kk))
        rev = tuple(cycle[(rmin - j) % kk] for j in range(kk))
        return min(fwd, rev)

    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for start in adj:
        # DFS up to depth k from start. Only continue when each new
        # vertex is > start (cycle root must be smallest); the
        # closing edge must return to start.
        stack: list[tuple[list[int], set[int]]] = [([start], {start})]
        while stack:
            path, visited = stack.pop()
            if len(path) == k:
                # Check closure: last vertex must connect back to start.
                if path[0] in adj.get(path[-1], {}):
                    cyc = _canonical(tuple(path))
                    if cyc not in seen:
                        seen.add(cyc)
                        out.append(cyc)
                continue
            tail = path[-1]
            for nxt in nbrs.get(tail, ()):
                if nxt < start:
                    continue            # root-canonicalisation
                if nxt in visited:
                    continue
                stack.append((path + [nxt], visited | {nxt}))
    return out


def _vertex_negative_count(cycle: tuple[int, ...],
                            edge_signs: tuple[int, ...]) -> list[int]:
    """For each vertex in the cycle, count negative edges incident
    to that vertex *within the cycle*."""
    k = len(cycle)
    counts = [0] * k
    for i in range(k):
        if edge_signs[i] == -1:
            counts[i] += 1
            counts[(i + 1) % k] += 1
    return counts


def _classify_n_tuple(cycle: tuple[int, ...],
                       edge_signs: tuple[int, ...]) -> SignedNTuple:
    """Compute σ assignment and balance flag from cycle edge signs."""
    k = len(cycle)
    n_neg = sum(1 for s in edge_signs if s == -1)
    balanced = (n_neg % 2) == 0   # Davis 1967 weak balance
    neg_counts = _vertex_negative_count(cycle, edge_signs)
    sigma = tuple(1 if (c % 2) == 0 else -1 for c in neg_counts)
    return SignedNTuple(v=cycle, sigma=sigma, edge_signs=edge_signs,
                        balanced=balanced, arity=k)


# Hard memory cap for the unbounded code path. The Rust enumerator
# returns a (N, k) numpy ndarray, so the wire-crossing cost is
# negligible — but downstream classification builds a SignedNTuple per
# cycle (~400 B each in CPython), and unbounded enumeration on Slashdot
# k=4 produces 55 M cycles → ~22 GB of dataclass wrappers. Force the
# caller to opt into a bigger budget if they actually want it.
_DEFAULT_CYCLE_CAP = 2_000_000


def _enumerate_cycles_fast(g: SignedGraph, k: int,
                            max_cycles: int | None = None,
                            seed: int = 0,
                            directed: bool = False,
                            early_stop: bool = False):
    """Use the Rust enumerator from hymeko if available; otherwise fall
    back to the pure-Python DFS in :func:`enumerate_k_cycles`.

    ``directed=False`` (default): treat the graph as undirected (each
    stored (u,v) edge is symmetrised at adjacency-construction time).
    ``directed=True``: enumerate *directed* k-cycles — each cycle step
    must follow an out-edge. Stored ``(u,v)`` is u→v only.

    When ``max_cycles`` is set, the Rust enumerator reservoir-samples
    down to that many cycles. With ``early_stop=True`` it instead keeps
    the first ``max_cycles`` cycles encountered and aborts the DFS —
    much faster (10²–10³× on dense graphs at high k) but biased toward
    cycles starting at small-indexed vertices.

    If ``max_cycles is None``, a default cap of ``_DEFAULT_CYCLE_CAP``
    (2 M) is applied to prevent OOM on graphs whose cycle space exceeds
    that. Pass an explicit ``max_cycles`` (or ``-1`` for unbounded) to
    override.
    """
    if max_cycles is None:
        cap = _DEFAULT_CYCLE_CAP
    elif max_cycles < 0:
        cap = None
    else:
        cap = max_cycles
    try:
        import hymeko  # type: ignore
        # ── Axiom-aware top-K path: opt-in via env var ──
        # HSIKAN_TOPK_MODE = "global" | "per_vertex"   (otherwise → off)
        # HSIKAN_TOPK_K    = K when global; m when per_vertex
        # HSIKAN_TOPK_SCORER = "fraction_negative" (default) | "balance"
        #                     | "sign_product_abs" | "low_root"
        import os
        topk_mode = os.environ.get("HSIKAN_TOPK_MODE", "").strip()
        if topk_mode in ("global", "per_vertex"):
            scorer = os.environ.get("HSIKAN_TOPK_SCORER", "fraction_negative")
            pruner = os.environ.get("HSIKAN_TOPK_PRUNER", "none")
            K = int(os.environ.get("HSIKAN_TOPK_K", "16"))
            eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.uint32)
            ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.uint32)
            es = np.ascontiguousarray(g.signs, dtype=np.int8)
            if topk_mode == "global" and hasattr(
                    hymeko, "enumerate_top_k_cycles_signed_rs"):
                arr, _scores = hymeko.enumerate_top_k_cycles_signed_rs(
                    eu, ev, es, g.n_nodes, k, K, scorer, pruner,
                )
                return [tuple(row) for row in arr.tolist()]
            if topk_mode == "per_vertex" and hasattr(
                    hymeko, "enumerate_top_k_per_vertex_cycles_signed_rs"):
                arr, _scores = (
                    hymeko.enumerate_top_k_per_vertex_cycles_signed_rs(
                        eu, ev, es, g.n_nodes, k, K, scorer, pruner,
                    )
                )
                return [tuple(row) for row in arr.tolist()]
        if hasattr(hymeko, "enumerate_k_cycles_rs"):
            eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.uint32)
            ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.uint32)
            arr = hymeko.enumerate_k_cycles_rs(
                eu, ev, g.n_nodes, k,
                cap, int(seed), directed, early_stop,
            )
            # arr is a (N, k) uint32 ndarray. Convert via .tolist() —
            # the C-level path is ~10× faster than the per-element
            # int(x) loop and produces native Python ints suitable for
            # downstream classification (which expects hashable tuples).
            return [tuple(row) for row in arr.tolist()]
    except ImportError:
        pass
    adj = _adjacency(g, directed=directed)
    cycles = enumerate_k_cycles(adj, k)
    if cap is not None and len(cycles) > cap:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(cycles), size=cap, replace=False)
        cycles = [cycles[int(i)] for i in idx]
    return cycles


def construct_2(g: SignedGraph) -> list[SignedNTuple]:
    """k=2 hyperedges = the edges themselves, treated as 2-uniform
    "cycles" with σ assigned by Davis-style parity at each endpoint
    (a single negative edge sets σ_u = σ_v = -1; positive sets +1).

    These are NOT cycles in the graph-theoretic sense — they're the
    raw signed edges packaged so the existing HSiKAN layer can consume
    them at arity k=2. The arity-mixing apparatus (αₖ) then decides
    how much weight raw edges get vs higher-order cycles.

    LEAKAGE NOTE: built from g.edges. The downstream M_e construction
    must exclude the self-edge from each query's incidence to avoid
    reading the answer from the input.
    """
    out = []
    for i in range(len(g.edges)):
        u = int(g.edges[i, 0])
        v = int(g.edges[i, 1])
        s = int(g.signs[i])
        sigma_v = 1 if s == 1 else -1   # both endpoints share parity
        out.append(SignedNTuple(
            v=(u, v),
            sigma=(sigma_v, sigma_v),
            edge_signs=(s,),
            balanced=(s == 1),
            arity=2,
        ))
    return out


def construct_k(g: SignedGraph, k: int,
                 max_cycles: int | None = None,
                 seed: int = 0,
                 directed: bool = False,
                 early_stop: bool = False) -> list[SignedNTuple]:
    """Build the list of signed $k$-tuples for the graph.

    For $k=3$ this is mathematically equivalent to the existing
    ``construct()`` in ``hyperedges.py``, modulo the σ-tie-breaking
    convention on balanced triads (the existing code picks the
    lowest-index vertex as apex; here we use Davis-style parity which
    gives the same result for unbalanced triads but a different σ
    pattern on balanced ones — both are valid).

    ``max_cycles`` subsamples the raw cycle list **before**
    classification — essential when classification of all cycles would
    OOM (Slashdot k=4 = 55M cycles → ~25GB of SignedNTuple wrappers).
    """
    if k < 3:
        raise ValueError(f"k must be >= 3, got {k}")
    adj = _adjacency(g, directed=directed)
    # Push the cap into the enumerator so we never materialise more
    # than ``max_cycles`` Python tuples in memory.
    cycles = _enumerate_cycles_fast(g, k, max_cycles=max_cycles,
                                       seed=seed, directed=directed,
                                       early_stop=early_stop)
    out = []
    for cyc in cycles:
        es = _cycle_edges(adj, cyc)
        if es is None:
            continue
        out.append(_classify_n_tuple(cyc, es))
    return out


def stats(tuples: list[SignedNTuple]) -> dict:
    if not tuples:
        return {"n": 0, "k": None, "n_balanced": 0, "balanced_frac": 0.0}
    ks = {t.arity for t in tuples}
    n_bal = sum(1 for t in tuples if t.balanced)
    return {
        "n": len(tuples),
        "k": tuples[0].arity if len(ks) == 1 else f"mixed{sorted(ks)}",
        "n_balanced": n_bal,
        "balanced_frac": n_bal / len(tuples),
    }
