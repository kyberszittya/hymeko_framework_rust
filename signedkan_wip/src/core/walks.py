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
            walks = _python_walks(g, walk_len, max_walks=max_walks,
                                   seed=seed)
    except ImportError:
        walks = _python_walks(g, walk_len, max_walks=max_walks, seed=seed)

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


def _classify_walks_arrays(
    g: SignedGraph, walks_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised σ + edge_signs computation from a ``(N, walk_len+1)``
    numpy array of walk vertex indices. Returns
    ``(v_int32, sigma_int8, edge_signs_int8)`` parallel arrays —
    bypasses the SignedNTuple-list materialisation that costs the
    Komondor walk-k=5 path 30+ GB of Python objects (job 13883876
    OOM, 2026-06-03).

    Walks are OPEN: ``arity_edges = walk_len = walks_arr.shape[1] - 1``.
    σ-recurrence has NO modular wrap at the endpoints — vertex 0 sees
    only edge 0, vertex L sees only edge L-1.

    Mirrors :func:`signedkan_wip.src.core.n_tuples._classify_k_arrays`
    for cycles; the two differ only in (1) ``edge_signs`` width and
    (2) absence of the seam edge.
    """
    if walks_arr.size == 0:
        empty_i = np.zeros((0, 0), dtype=np.int32)
        empty_s = np.zeros((0, 0), dtype=np.int8)
        return empty_i, empty_s, empty_s.copy()

    N, L_plus_1 = walks_arr.shape
    L = L_plus_1 - 1
    walks_np = np.ascontiguousarray(walks_arr, dtype=np.int64)

    # Sparse edge → sign lookup. Symmetrise undirected edges so both
    # (u, v) and (v, u) keys hit the same sign. Linear key
    # ``u * n_nodes + v`` packs both directions into one sorted vector.
    e = np.asarray(g.edges, dtype=np.int64)
    s = np.asarray(g.signs, dtype=np.int8)
    n_nodes = int(g.n_nodes)
    keys_fwd = e[:, 0] * n_nodes + e[:, 1]
    keys_rev = e[:, 1] * n_nodes + e[:, 0]
    all_keys = np.concatenate([keys_fwd, keys_rev])
    all_signs = np.concatenate([s, s])
    order = np.argsort(all_keys, kind="stable")
    sorted_keys = all_keys[order]
    sorted_signs = all_signs[order]

    # For each walk, look up the L edge signs in cycle order. Open
    # walk → no modular wrap, indices 0..L-1 only.
    u_mat = walks_np[:, :L]                              # (N, L)
    v_mat = walks_np[:, 1:L_plus_1]                      # (N, L)
    query_keys = u_mat * n_nodes + v_mat                 # (N, L)
    pos = np.searchsorted(sorted_keys, query_keys.ravel())
    pos = np.clip(pos, 0, sorted_keys.size - 1)
    found = sorted_keys[pos] == query_keys.ravel()
    edge_signs_flat = np.where(found, sorted_signs[pos], 0).astype(np.int8)
    edge_signs = edge_signs_flat.reshape(N, L)

    # Filter walks that touch a non-edge (any zero in edge_signs).
    valid = (edge_signs != 0).all(axis=1)
    if not valid.all():
        walks_np = walks_np[valid]
        edge_signs = edge_signs[valid]
        N = walks_np.shape[0]
        if N == 0:
            empty_i = np.zeros((0, L_plus_1), dtype=np.int32)
            empty_s_v = np.zeros((0, L_plus_1), dtype=np.int8)
            empty_s_e = np.zeros((0, L), dtype=np.int8)
            return empty_i, empty_s_v, empty_s_e

    # Per-vertex negative-incidence count within each walk.
    # Vertex 0    sees edge 0 only.
    # Vertex i,   0 < i < L,  sees edges i-1 and i.
    # Vertex L    sees edge L-1 only.
    is_neg = (edge_signs == -1).astype(np.int8)          # (N, L)
    neg_counts = np.zeros((N, L_plus_1), dtype=np.int8)
    neg_counts[:, 0]     = is_neg[:, 0]                  # endpoint
    neg_counts[:, L]     = is_neg[:, L - 1]              # endpoint
    if L > 1:
        # interior vertices: sum of two adjacent edges' is_neg
        neg_counts[:, 1:L] = is_neg[:, :L - 1] + is_neg[:, 1:L]
    sigma = np.where((neg_counts % 2) == 0, 1, -1).astype(np.int8)
    v_out = walks_np.astype(np.int32)
    return v_out, sigma, edge_signs


def construct_walks_arrays(
    g: SignedGraph,
    walk_len: int,
    max_walks: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memory-optimised walks constructor: returns
    ``(v, sigma, edge_signs)`` packed numpy arrays directly,
    skipping the legacy SignedNTuple-list path.

    Subsamples to ``max_walks`` IN NUMPY SPACE before any Python
    object materialisation — caps peak RSS at
    ``raw_arr.nbytes + max_walks * 4·L``, whereas the legacy
    :func:`construct_walks` materialised every Rust-enumerated walk as
    a Python tuple *before* subsampling. On bitcoin_alpha walk_len=5
    that peak is multiple GB (Komondor job 13883876 OOM,
    walk k=5 build_me, 2026-06-03).

    Same enumeration semantics as :func:`construct_walks`:
    canonical-form filter (``walk[0] <= walk[-1]``), Davis-style σ.

    Returns
    -------
    v : (N, walk_len + 1) int32
        Walk vertex indices in traversal order.
    sigma : (N, walk_len + 1) int8
        Per-vertex sign, +1 / -1.
    edge_signs : (N, walk_len) int8
        Per-edge sign of each walk-edge.
    """
    if walk_len < 1:
        raise ValueError(f"walk_len must be >= 1, got {walk_len}")

    try:
        import hymeko  # type: ignore
        if hasattr(hymeko, "enumerate_k_walks_rs"):
            eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.uint32)
            ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.uint32)
            # Rust enumerator returns (N_total, walk_len+1) uint32.
            # ``max_walks`` IS honoured by the Rust side (capped emission)
            # — we still subsample defensively in Python to handle older
            # Rust binaries that may have ignored the cap.
            arr_raw = hymeko.enumerate_k_walks_rs(
                eu.tolist(), ev.tolist(), g.n_nodes,
                int(walk_len),
                int(max_walks) if max_walks else None,
                int(seed),
            )
            arr = np.asarray(arr_raw, dtype=np.int64)
        else:
            arr = _python_walks_array(g, walk_len, max_walks=max_walks,
                                       seed=seed)
    except ImportError:
        # Forward max_walks so the Python DFS stays memory-bounded
        # via the reservoir sampler (2026-06-03 Komondor fix).
        arr = _python_walks_array(g, walk_len, max_walks=max_walks,
                                   seed=seed)

    # Defensive numpy-space subsample BEFORE classification.
    if max_walks is not None and arr.shape[0] > max_walks:
        rng = np.random.default_rng(seed)
        idx = rng.choice(arr.shape[0], size=max_walks, replace=False)
        arr = arr[idx]

    return _classify_walks_arrays(g, arr)


def _python_walks_array(
    g: SignedGraph, walk_len: int,
    max_walks: int | None = None, seed: int = 0,
) -> np.ndarray:
    """Pure-Python fallback for the walk enumerator, numpy-direct.

    When ``max_walks`` is given, runs a :class:`NumpyReservoirSampler`
    INSIDE the DFS — no Python-tuple allocation per visit, Algorithm L
    skip-sampling so most visits incur only a counter bump. Returns
    the ``(min(cap, seen), walk_len+1)`` numpy array.

    When ``max_walks`` is ``None``, uses an unbounded ``out`` list-of-
    tuples and converts to numpy at the end (legacy compatibility
    path, only safe on small fixtures).

    Komondor fix 2026-06-03: walk_len=4 bitcoin_alpha visited billions
    of length-5 paths through high-degree nodes; the legacy unbounded
    fallback OOM-killed at 32 GB. With the numpy reservoir the same
    case is bounded at ``max_walks × (walk_len+1) × 4`` bytes ≈ 2 MB.
    """
    from .reservoir import NumpyReservoirSampler

    n = g.n_nodes
    adj: dict[int, list[int]] = {}
    for i in range(len(g.edges)):
        u = int(g.edges[i, 0]); v = int(g.edges[i, 1])
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    if max_walks is None:
        # Unbounded path — only safe when caller knows the population
        # size is tractable. Kept for parity with the pre-2026-06-03
        # behaviour on small fixtures / unit-test code paths.
        walks = _python_walks(g, walk_len, max_walks=None, seed=seed)
        if not walks:
            return np.zeros((0, walk_len + 1), dtype=np.int64)
        return np.asarray(walks, dtype=np.int64)

    sampler = NumpyReservoirSampler(
        cap=int(max_walks), k=walk_len + 1,
        dtype=np.int64, seed=seed,
    )

    def dfs(path: list[int], visited: set[int]):
        if len(path) == walk_len + 1:
            if path[0] <= path[-1]:
                sampler.offer(path)        # no tuple alloc
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
    # Copy out of the preallocated buffer so the caller can mutate
    # without affecting the sampler. The (min(cap, seen), k) slice
    # is the uniform sample.
    return np.array(sampler.to_array(), dtype=np.int64, copy=True)


def _python_walks(
    g: SignedGraph, walk_len: int,
    max_walks: int | None = None, seed: int = 0,
):
    """List-of-tuples variant of the Python walks fallback.

    Memory-bounded since 2026-06-03: when ``max_walks`` is set, the
    DFS feeds a :class:`NumpyReservoirSampler` (zero per-visit Python
    alloc) and converts only the surviving rows to tuples at the end.
    For an unbounded call (``max_walks=None``) the legacy
    ``out.append(tuple(path))`` path is retained — only safe on small
    fixtures where the population fits in memory.

    Without the reservoir, walk_len=4 on bitcoin_alpha generates
    billions of length-5 tuples through high-degree nodes — Komondor
    job 13883886 OOM-killed at 32 GB on that exact case.
    """
    if max_walks is not None:
        arr = _python_walks_array(g, walk_len, max_walks=max_walks,
                                   seed=seed)
        return [tuple(int(x) for x in row) for row in arr.tolist()]

    # Unbounded legacy path.
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
