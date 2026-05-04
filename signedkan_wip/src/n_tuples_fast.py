"""Fast 4-cycle enumeration via sparse matrix algebra + numpy
vectorisation.

Two entry points:
  - ``construct_4_fast_arrays(g)`` — returns raw numpy arrays
    ``(cycle_v, edge_signs, sigma, balanced)``. Skips per-cycle
    Python object construction; the dominant overhead at
    Slashdot scale (10M+ cycles).
  - ``construct_4_fast(g)`` — backward-compatible wrapper that
    constructs ``SignedNTuple`` objects from the arrays.

Algorithm:

  A 4-cycle (u, m_1, w, m_2) — vertices in cycle order — has
  edges (u,m_1), (m_1,w), (w,m_2), (m_2,u). u and w are at the
  diagonal; m_1, m_2 are the two common neighbours of u and w.
  Each unordered pair {m_1, m_2} ⊂ N(u) ∩ N(w) yields exactly one
  4-cycle.

Canonical dedup without a `seen` set:
  Enforce u < w, m_1 > u, m_2 > u (so u is the smallest of all
  four vertices), and m_1 < m_2 (canonical orientation). Each
  cycle is then emitted exactly once.

Vectorisation:
  - Common-neighbour intersection via `numpy.intersect1d`.
  - Pair generation via `numpy.triu_indices`.
  - Edge-sign lookup vectorised via scipy CSR fancy indexing on
    the signed adjacency.
  - σ assignment computed in bulk via numpy boolean ops.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .datasets import SignedGraph
from .hyperedges import _adjacency
from .n_tuples import SignedNTuple


def construct_4_fast_arrays(
    g: SignedGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate all 4-cycles, return as raw numpy arrays.

    Returns
    -------
    cycle_v   : (T, 4) int64 — vertex IDs in cycle order, canonical
                 (smallest vertex first, lex-smaller orientation)
    edge_signs: (T, 4) int8  — signs of cycle edges in cycle order
    sigma     : (T, 4) int8  — Davis per-vertex σ assignment (±1)
    balanced  : (T,)   bool  — Davis weak-balance flag
    """
    adj = _adjacency(g)
    nbrs = {v: set(d.keys()) for v, d in adj.items()}
    n = g.n_nodes

    # Sorted per-vertex neighbour arrays for fast intersection.
    nbr_arr: dict[int, np.ndarray] = {}
    for v in range(n):
        if v in nbrs and nbrs[v]:
            nbr_arr[v] = np.fromiter(sorted(nbrs[v]),
                                       dtype=np.int64,
                                       count=len(nbrs[v]))

    # Sparse symmetric binary adjacency for A² (common-neighbour count).
    rows, cols, signs = [], [], []
    for u, ns in adj.items():
        for v, s in ns.items():
            rows.append(u); cols.append(v); signs.append(int(s))
    rows_np = np.asarray(rows, dtype=np.int64)
    cols_np = np.asarray(cols, dtype=np.int64)
    signs_np = np.asarray(signs, dtype=np.int8)
    A = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows_np, cols_np)),
        shape=(n, n),
    ).tocsr()
    A.data[:] = 1
    A2 = (A @ A).tocsr()
    A2.setdiag(0)
    A2.eliminate_zeros()
    indptr, indices, data = A2.indptr, A2.indices, A2.data

    # Signed adjacency for vectorised edge-sign lookup later.
    A_sgn = sp.coo_matrix(
        (signs_np, (rows_np, cols_np)), shape=(n, n), dtype=np.int8,
    ).tocsr()

    # Accumulate cycle batches.
    cycle_chunks: list[np.ndarray] = []

    for u in range(n):
        if u not in nbr_arr:
            continue
        nbr_u = nbr_arr[u]
        nbr_u_above = nbr_u[nbr_u > u]
        if len(nbr_u_above) < 2:
            continue
        start, end = indptr[u], indptr[u + 1]
        for k_idx in range(start, end):
            w = int(indices[k_idx])
            if w <= u:
                continue
            if data[k_idx] < 2:
                continue
            if w not in nbr_arr:
                continue
            common = np.intersect1d(nbr_u_above, nbr_arr[w],
                                     assume_unique=True)
            if len(common) < 2:
                continue
            # Drop w if present (m must not equal w).
            common = common[common != w]
            if len(common) < 2:
                continue
            pi, pj = np.triu_indices(len(common), k=1)
            n_pairs = len(pi)
            cyc = np.empty((n_pairs, 4), dtype=np.int64)
            cyc[:, 0] = u
            cyc[:, 1] = common[pi]
            cyc[:, 2] = w
            cyc[:, 3] = common[pj]
            cycle_chunks.append(cyc)

    if not cycle_chunks:
        return (np.zeros((0, 4), dtype=np.int64),
                np.zeros((0, 4), dtype=np.int8),
                np.zeros((0, 4), dtype=np.int8),
                np.zeros((0,), dtype=bool))

    cycle_v = np.concatenate(cycle_chunks, axis=0)
    T = cycle_v.shape[0]

    # Vectorised edge-sign lookup via CSR fancy indexing.
    # Cycle edges: (cyc[:,0], cyc[:,1]), (cyc[:,1], cyc[:,2]),
    #              (cyc[:,2], cyc[:,3]), (cyc[:,3], cyc[:,0]).
    edge_signs = np.empty((T, 4), dtype=np.int8)
    for j in range(4):
        u_col = cycle_v[:, j]
        v_col = cycle_v[:, (j + 1) % 4]
        edge_signs[:, j] = np.asarray(A_sgn[u_col, v_col]).flatten()

    # σ_i = +1 iff vertex i is incident to an EVEN number of negative
    # cycle edges (Davis 1967 weak balance, vertex parity rule).
    is_neg = (edge_signs == -1).astype(np.int8)        # (T, 4)
    # Vertex j (j-th in cycle) is incident to cycle-edges (j-1) and j.
    neg_counts = np.empty((T, 4), dtype=np.int8)
    for j in range(4):
        neg_counts[:, j] = is_neg[:, (j - 1) % 4] + is_neg[:, j]
    sigma = np.where(neg_counts % 2 == 0,
                      np.int8(1), np.int8(-1)).astype(np.int8)

    # Davis weak balance: total negative count even.
    balanced = (is_neg.sum(axis=1) % 2 == 0)

    return cycle_v, edge_signs, sigma, balanced


def construct_4_fast(g: SignedGraph) -> list[SignedNTuple]:
    """Backward-compatible wrapper. Constructs ``SignedNTuple`` objects
    from the raw arrays. For Slashdot-scale fixtures, prefer
    ``construct_4_fast_arrays`` directly to avoid the per-cycle Python
    object overhead."""
    cycle_v, edge_signs, sigma, balanced = construct_4_fast_arrays(g)
    return [
        SignedNTuple(
            v=tuple(int(x) for x in cycle_v[i]),
            sigma=tuple(int(x) for x in sigma[i]),
            edge_signs=tuple(int(x) for x in edge_signs[i]),
            balanced=bool(balanced[i]),
            arity=4,
        )
        for i in range(cycle_v.shape[0])
    ]
