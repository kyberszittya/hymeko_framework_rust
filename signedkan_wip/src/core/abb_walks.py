"""Accelerated Branch and Bound (ABB) walk enumeration with top-K
score-driven pruning, in the Friedler P-graph tradition.

This is the Python-side reference / fallback for the same algorithm
the cycle path uses in Rust (``hymeko_graph::topk_cycles`` with
``BoundedScorer`` upper-bound pruning). When the ``hymeko`` Rust
wheel is unavailable inside the Singularity container (Komondor
2026-06-03), this module gives the user a memory- and score-bounded
walk enumerator without paying the OOM cost of the legacy unbounded
``_python_walks`` DFS.

Design — composable under the Strategy + Adapter pattern in
:mod:`signedkan_wip.src.cycle_cache.strategies`:

    enum = ABBWalkEnumerator(
        walk_len=4, top_k=100,
        scorer=BalanceScorer(),
    )
    arr = enum.enumerate(g, seed=0)
    # arr.v.shape == (≤ top_k, walk_len + 1)
    # arr.v contains the top-K walks by ``scorer.score``.

The DFS maintains a min-heap of size ``top_k`` of ``(score, walk)``
pairs. At each extension, the scorer's ``upper_bound`` on the current
prefix is compared against the heap's minimum: if the bound falls
below the current top-K threshold, the entire subtree is pruned.

Reports the ABB pruning rate so callers can quantify the
acceleration vs the vanilla MSG (unbounded) path. Precedent on
cycles: 25× speedup on Epinions k=4 K=10K
(reports/2026-05-10-abb-global-topk.md).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .path_scorers import PathScorer


@dataclass
class ABBStats:
    """Pruning rate + visit counts for the ABB DFS run."""

    n_visited: int = 0
    n_extended: int = 0
    n_pruned_branches: int = 0
    n_emitted: int = 0
    n_replaced_in_heap: int = 0

    @property
    def prune_rate(self) -> float:
        """Fraction of DFS-extension attempts that were pruned."""
        total = self.n_extended + self.n_pruned_branches
        if total == 0:
            return 0.0
        return self.n_pruned_branches / total


def _try_rust_top_k_walks(
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    edge_signs: np.ndarray,
    n_nodes: int,
    walk_len: int,
    top_k: int,
    scorer: PathScorer,
):
    """Attempt the Rust ``enumerate_top_k_walks_rs`` path. Returns the
    ``(walks_v, walks_signs)`` numpy pair on success or ``None`` if the
    Rust extension is unavailable or the scorer name has no Rust analog.

    Promoted 2026-06-03 as the production runner. The Python DFS in
    :func:`abb_enumerate_walks` remains the correctness reference; the
    Rust path is used when the wheel is installed (local dev box,
    Komondor after the wheel ship) and silently bypassed in
    Singularity-style minimal containers."""
    try:
        import hymeko as _hk  # type: ignore
    except ImportError:
        return None
    if not hasattr(_hk, "enumerate_top_k_walks_rs"):
        return None
    name = scorer.name()
    if name not in {"balance", "fraction_negative", "sign_product_abs"}:
        return None
    eu = np.ascontiguousarray(edges_u, dtype=np.uint32)
    ev = np.ascontiguousarray(edges_v, dtype=np.uint32)
    es = np.ascontiguousarray(edge_signs, dtype=np.int8)
    walks, signs, _scores = _hk.enumerate_top_k_walks_rs(
        eu.tolist(), ev.tolist(), es.tolist(),
        int(n_nodes), int(walk_len), int(top_k), name,
    )
    return (
        np.asarray(walks, dtype=np.int32),
        np.asarray(signs, dtype=np.int8),
    )


def abb_enumerate_walks(
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    edge_signs: np.ndarray,
    n_nodes: int,
    walk_len: int,
    top_k: int,
    scorer: PathScorer,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, ABBStats]:
    """Enumerate top-K walks of length ``walk_len`` under
    ``scorer.score``, with admissible-UB DFS pruning.

    Canonicalisation: only emit walks with ``walk[0] <= walk[-1]`` so
    each open walk is counted once (same convention as
    :func:`signedkan_wip.src.core.walks.construct_walks`).

    Parameters
    ----------
    edges_u, edges_v
        ``(E,)`` arrays of edge endpoints (undirected; both directions
        are inserted into the adjacency list internally).
    edge_signs
        ``(E,)`` int8 array of edge signs (``±1``).
    n_nodes
        Number of vertices in the graph.
    walk_len
        Number of edges per walk (``arity_edges``); vertex count is
        ``walk_len + 1``.
    top_k
        Cap on the returned walk count. Internally drives the min-heap
        whose threshold is the ABB pruning bound.
    scorer
        Any :class:`PathScorer`. The ABB DFS calls
        ``scorer.upper_bound`` at each extension and ``scorer.score``
        at each closed walk.
    seed
        Deterministic tie-break for equal-score walks (the heap is
        ordered by (score, secondary_key); secondary_key is
        ``random.Random(seed).random()`` to break ties uniformly).

    Returns
    -------
    walks_v
        ``(min(top_k, n_complete), walk_len + 1)`` int32 array of walk
        vertex sequences.
    walks_signs
        ``(min(top_k, n_complete), walk_len)`` int8 array of per-edge
        signs aligned to ``walks_v[:, :-1] / walks_v[:, 1:]``.
    stats
        :class:`ABBStats` with the prune rate + visit counts.

    Memory: O(top_k · walk_len) for the heap; O(walk_len) DFS stack +
    O(E) adjacency. Crucially independent of the visited-walk count,
    so the same algorithm runs on bitcoin_alpha (k=4 → 4·10⁹ visits)
    without OOM.
    """
    if walk_len < 1:
        raise ValueError(f"walk_len must be >= 1, got {walk_len}")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")
    if top_k == 0:
        return (
            np.zeros((0, walk_len + 1), dtype=np.int32),
            np.zeros((0, walk_len), dtype=np.int8),
            ABBStats(),
        )

    # Production runner: delegate to the Rust enumerator when the
    # hymeko wheel is installed AND the scorer has a Rust analog.
    # Falls back to the pure-Python DFS below otherwise (correctness
    # reference; required for the hymeko-missing Singularity case).
    rust = _try_rust_top_k_walks(
        edges_u, edges_v, edge_signs, n_nodes, walk_len, top_k, scorer,
    )
    if rust is not None:
        walks_v, walks_signs = rust
        # Rust path doesn't track per-DFS stats; report a sentinel.
        return walks_v, walks_signs, ABBStats(n_emitted=int(walks_v.shape[0]))

    # Build undirected CSR-ish adjacency with sign lookup.
    # adj[u] -> list of (v, sign).
    adj: dict[int, list[tuple[int, int]]] = {}
    for i in range(len(edges_u)):
        u = int(edges_u[i])
        v = int(edges_v[i])
        s = int(edge_signs[i])
        adj.setdefault(u, []).append((v, s))
        adj.setdefault(v, []).append((u, s))
    # Sort neighbours by index for deterministic DFS.
    for u in adj:
        adj[u].sort()

    # Min-heap of (score, secondary_tiebreak, vs_tuple, signs_tuple).
    # Length ≤ top_k; the minimum is the current threshold.
    import random as _rnd
    rng = _rnd.Random(int(seed))

    # Heap entries are (score, tiebreak, vs_tuple, signs_tuple).
    # heapq is a MIN-heap; the smallest score is at heap[0] — that
    # IS the threshold (the worst surviving top-K walk).
    heap: list[tuple[float, float, tuple[int, ...], tuple[int, ...]]] = []
    stats = ABBStats()

    # Mutable DFS scratch — avoids per-step list allocation.
    path: list[int] = []
    signs: list[int] = []
    visited: set[int] = set()

    def _emit_walk():
        """Score and offer the currently-complete walk to the heap."""
        # Canonical form: only emit when walk[0] <= walk[-1]; this
        # halves the count of equivalent open walks.
        if path[0] > path[-1]:
            return
        stats.n_emitted += 1
        score = scorer.score(path, signs)
        tiebreak = rng.random()
        vs_tuple = tuple(path)
        signs_tuple = tuple(signs)
        if len(heap) < top_k:
            heapq.heappush(
                heap, (score, tiebreak, vs_tuple, signs_tuple),
            )
        elif score > heap[0][0]:
            heapq.heapreplace(
                heap, (score, tiebreak, vs_tuple, signs_tuple),
            )
            stats.n_replaced_in_heap += 1

    def _current_threshold() -> float:
        """Return the score of the worst walk in the heap (or
        ``-inf`` if the heap isn't full yet — every walk is a
        candidate during heap fill)."""
        if len(heap) < top_k:
            return float("-inf")
        return heap[0][0]

    def dfs(n_neg_so_far: int):
        """Recursive DFS with ABB pruning."""
        stats.n_visited += 1
        if len(path) == walk_len + 1:
            _emit_walk()
            return
        tail = path[-1]
        steps_remaining = (walk_len + 1) - len(path)
        # Edges remaining = vertices remaining (open walk: edges =
        # vertices - 1; we have len(signs) edges so far; need
        # walk_len - len(signs) more).
        edges_remaining = walk_len - len(signs)
        for nxt, edge_sign in adj.get(tail, ()):
            if nxt in visited:
                continue
            # Pre-extension ABB check.
            new_n_neg = n_neg_so_far + (1 if edge_sign < 0 else 0)
            new_steps_left = edges_remaining - 1
            ub = scorer.upper_bound(
                new_n_neg, new_steps_left, walk_len,
            )
            if ub <= _current_threshold():
                stats.n_pruned_branches += 1
                continue
            stats.n_extended += 1
            path.append(nxt)
            signs.append(edge_sign)
            visited.add(nxt)
            dfs(new_n_neg)
            path.pop()
            signs.pop()
            visited.remove(nxt)

    for start in range(n_nodes):
        path.append(start)
        visited.add(start)
        dfs(n_neg_so_far=0)
        path.pop()
        visited.remove(start)

    # Drain the heap into output arrays. Heap is min-ordered; we
    # return walks in DESCENDING score order so the caller can take
    # ``[:k_keep]`` directly.
    heap.sort(key=lambda e: -e[0])
    n_out = len(heap)
    walks_v = np.zeros((n_out, walk_len + 1), dtype=np.int32)
    walks_signs = np.zeros((n_out, walk_len), dtype=np.int8)
    for i, (_score, _tiebreak, vs_tuple, signs_tuple) in enumerate(heap):
        walks_v[i] = vs_tuple
        walks_signs[i] = signs_tuple
    return walks_v, walks_signs, stats


def msg_enumerate_walks(
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    n_nodes: int,
    walk_len: int,
    max_walks: int | None = None,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Maximum Structure Generation — enumerate all feasible walks,
    reservoir-sample to ``max_walks`` for memory bound.

    P-graph terminology: MSG is the "generate every feasible
    structure" pass. For walks on a signed graph, "feasibility" is
    just the graph topology (no axiomatic exclusion beyond simple-
    walk: no vertex revisits within a walk). Output is unbiased by
    construction (every visited walk has equal acceptance probability
    under Vitter Algorithm L).

    Delegates to :class:`signedkan_wip.src.core.reservoir.NumpyReservoirSampler`
    via :func:`signedkan_wip.src.core.walks._python_walks_array`. Kept
    as a public name so MSG / ABB / SSG present a uniform API surface.
    """
    from .walks import _python_walks_array
    from ..datasets import SignedGraph

    # walks._python_walks_array takes a SignedGraph fixture; assemble
    # one from the array inputs. The sign vector is unused by walks
    # enumeration (only by classification), so an all-+1 vector is
    # fine — the classification step is the caller's responsibility.
    g = SignedGraph(
        edges=np.column_stack([edges_u, edges_v]).astype(np.int64),
        signs=np.ones(len(edges_u), dtype=np.int8),
        n_nodes=int(n_nodes),
    )
    return _python_walks_array(
        g, walk_len, max_walks=max_walks, seed=seed,
    )


def ssg_pareto_filter(
    walks_v: np.ndarray,
    walks_signs: np.ndarray,
    score_axes: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subset Structure Generation — Pareto filter walks across
    multiple score axes.

    P-graph terminology: SSG selects the Pareto-optimal subset of the
    structures emitted by MSG. For walks, the typical axes are
    (score, cost) — e.g. (balance, walk_length) when the enumerator
    mixes walk lengths. With a single score axis, SSG degenerates to
    "take top-K by score"; we keep the multi-axis API so the same
    code path serves future multi-objective HSiKAN sweeps.

    Parameters
    ----------
    walks_v
        ``(N, k+1)`` int32 vertex arrays from ABB / MSG.
    walks_signs
        ``(N, k)`` int8 sign arrays parallel to ``walks_v``.
    score_axes
        List of ``(N,)`` float arrays. A walk is on the Pareto
        frontier iff no other walk strictly dominates it on every
        axis. Convention: HIGHER is BETTER on every axis (negate the
        cost axis before passing if minimising).

    Returns
    -------
    pareto_v, pareto_signs, mask
        Filtered arrays and the boolean mask ``(N,)`` indicating which
        rows of the input survived.

    Notes
    -----
    Dispatches to a $O(N \\log N)$ sort-and-sweep
    (Kung--Luccio--Preparata 1975) at $D = 2$ -- the practical
    HSiKAN case (score vs cost / walk_len). $D \\ne 2$ falls back to
    the $O(N^2 D)$ brute path; both produce the same boolean mask.
    See `docs/plans/2026-06-04-klp-skyline/plan.pdf` for the
    correctness argument and benchmark.
    """
    if not score_axes:
        return walks_v, walks_signs, np.ones(len(walks_v), dtype=bool)
    N = walks_v.shape[0]
    if N == 0:
        return walks_v, walks_signs, np.zeros(0, dtype=bool)
    if any(ax.shape != (N,) for ax in score_axes):
        raise ValueError(
            f"every score axis must have shape ({N},); got "
            f"{[ax.shape for ax in score_axes]}"
        )
    scores = np.column_stack(score_axes)
    if len(score_axes) == 2:
        mask = _ssg_pareto_filter_sweep_2d(scores)
    else:
        mask = _ssg_pareto_filter_brute(scores)
    return walks_v[mask], walks_signs[mask], mask


def _ssg_pareto_filter_brute(scores: np.ndarray) -> np.ndarray:
    """Reference $O(N^2 D)$ Pareto filter. The boolean output is the
    specification the sort-and-sweep path is tested against."""
    N = scores.shape[0]
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        if not mask[i]:
            continue
        ge = (scores >= scores[i]).all(axis=1)
        gt_any = (scores > scores[i]).any(axis=1)
        dominators = ge & gt_any
        dominators[i] = False
        if dominators.any():
            mask[i] = False
    return mask


def _ssg_pareto_filter_sweep_2d(scores: np.ndarray) -> np.ndarray:
    """Kung--Luccio--Preparata 1975 sort-and-sweep skyline for D = 2.

    Higher is better on every axis. A point is on the skyline iff no
    other point STRICTLY DOMINATES it (Pareto: $\\geq$ on every axis,
    $>$ on at least one). Complexity: $O(N \\log N)$.

    Algorithm
    ---------
    1. Sort points by ``a_0`` desc; tie-break ``a_1`` desc.
    2. Walk the sort in $a_0$-groups (one group per distinct $a_0$).
    3. For each group, compute ``group_max_a1``. The points in the
       group with ``a_1 == group_max_a1`` are on the skyline iff
       ``group_max_a1 > running_max_a1``, where ``running_max_a1`` is
       the max $a_1$ seen across all STRICTLY EARLIER groups
       (i.e.\\ groups with strictly larger $a_0$). Other points in
       the group are dominated by the group's max (same $a_0$, lower
       $a_1$ -- strict on $a_1$).
    4. Update ``running_max_a1 <- max(running_max_a1, group_max_a1)``.
    5. Restore original index order via the inverse permutation.

    Tie handling -- the load-bearing edge case:

    * Two points with identical $(a_0, a_1)$: neither dominates the
      other (no strict $>$). The algorithm keeps BOTH (they share the
      group max and the group max passes the strict check).
    * Same $a_0$, different $a_1$: the lower-$a_1$ one is dominated
      by the higher within the same group.
    * Same $a_1$, different $a_0$: the lower-$a_0$ one's group has
      ``group_max_a1 == running_max_a1`` (the previously-seen larger
      $a_0$ point's $a_1$). The strict $>$ check fails; the lower-
      $a_0$ point is dominated.

    Bit-for-bit equivalent to the brute reference for all randomised
    + integer-tied + degenerate inputs in
    ``signedkan_wip/tests/test_ssg_pareto_filter.py``.
    """
    if scores.ndim != 2 or scores.shape[1] != 2:
        raise ValueError(
            f"sweep_2d requires (N, 2) scores; got {scores.shape}"
        )
    N = scores.shape[0]
    if N == 0:
        return np.zeros(0, dtype=bool)
    if N == 1:
        return np.ones(1, dtype=bool)

    # Sort: primary -a_0 (i.e. a_0 desc), tie-break -a_1 (a_1 desc).
    # lexsort takes keys low-priority first, so primary key goes LAST.
    order = np.lexsort((-scores[:, 1], -scores[:, 0]))
    sorted_a0 = scores[order, 0]
    sorted_a1 = scores[order, 1]

    # Group boundaries: a new group starts wherever a_0 changes.
    group_breaks = np.concatenate(
        ([0], np.where(sorted_a0[1:] != sorted_a0[:-1])[0] + 1, [N])
    )

    on_skyline_sorted = np.zeros(N, dtype=bool)
    running_max_a1 = -np.inf
    for gi in range(len(group_breaks) - 1):
        gs = group_breaks[gi]
        ge = group_breaks[gi + 1]
        group_max_a1 = sorted_a1[gs:ge].max()
        if group_max_a1 > running_max_a1:
            # The group's max-a_1 row(s) survive; ties on a_1 within
            # the group all survive (neither dominates the other).
            on_skyline_sorted[gs:ge] = sorted_a1[gs:ge] == group_max_a1
            running_max_a1 = group_max_a1
        # else: the entire group is dominated by an earlier group's
        # point (strict on a_0, weak on a_1) -- skip it.

    # Invert the permutation so the mask aligns with the input order.
    mask = np.empty(N, dtype=bool)
    mask[order] = on_skyline_sorted
    return mask
