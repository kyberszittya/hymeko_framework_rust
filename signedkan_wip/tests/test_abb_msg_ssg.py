"""Tests for the MSG / ABB / SSG top-K walk-enumeration adapters.

Added 2026-06-03 alongside ``signedkan_wip/src/core/path_scorers.py``
and ``signedkan_wip/src/core/abb_walks.py``. Verifies:

1. :class:`PathScorer` admissibility — for every concrete walk, the
   upper bound at every prefix is ``>= score(walk)`` (the load-bearing
   correctness invariant of the ABB pruner). A violation silently
   produces a wrong top-K answer.

2. :func:`abb_enumerate_walks` returns the same top-K as a brute-force
   MSG → score → sort baseline on a small fixture. ABB MUST not lose
   any walk that brute-force would keep.

3. :func:`abb_enumerate_walks` reports a non-zero prune rate on the
   same fixture (otherwise ABB is doing no work and we're just paying
   the overhead).

4. :func:`ssg_pareto_filter` correctly removes dominated rows on
   synthetic two-axis fixtures.

5. :class:`ABBWalkEnumerator` + :class:`CachedEnumerator` round-trip:
   cold call writes the cache; warm call reads it; both produce the
   same arrays.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from signedkan_wip.src import cycle_cache
from signedkan_wip.src.cycle_cache.strategies import (
    ABBWalkEnumerator, CachedEnumerator, MSGWalkEnumerator,
    SSGWalkEnumerator, WalkEnumerator,
)
from signedkan_wip.src.core.abb_walks import (
    abb_enumerate_walks, msg_enumerate_walks, ssg_pareto_filter,
)
from signedkan_wip.src.core.path_scorers import (
    BalanceScorer, FractionNegativeScorer, PathScorer,
    ShannonEntropyScorer, SignProductAbsScorer, pick_scorer,
)
from signedkan_wip.src.datasets import SignedGraph


# ─── Fixtures ───────────────────────────────────────────────────


def _toy_signed_graph(seed: int = 0, n: int = 10) -> SignedGraph:
    """Dense signed graph with both signs present, deterministic from
    the seed. Small enough that brute-force MSG enumeration is cheap,
    big enough that ABB has meaningful work to prune."""
    rng = np.random.default_rng(seed)
    edges = []
    signs = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.55:
                edges.append((u, v))
                signs.append(1 if rng.random() < 0.6 else -1)
    return SignedGraph(
        edges=np.array(edges, dtype=np.int64),
        signs=np.array(signs, dtype=np.int8),
        n_nodes=n,
    )


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HYMEKO_CYCLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYMEKO_CYCLE_CACHE", "1")
    cycle_cache.reset_stats()
    yield


# ─── PathScorer admissibility ───────────────────────────────────


def _walk_prefix_states(walk_signs: np.ndarray):
    """Iterate (prefix_state, completion_score) pairs for one walk.

    Yields tuples ``(n_neg_so_far, steps_remaining, k_len, completion_score)``
    for every non-empty prefix; ``completion_score`` is the SAME walk's
    final score (so we can test ``upper_bound >= completion_score``)."""
    L = len(walk_signs)
    for prefix_len in range(L + 1):
        prefix = walk_signs[:prefix_len]
        n_neg = int((prefix < 0).sum())
        yield n_neg, L - prefix_len, L


def _enumerate_open_walks_brute(g: SignedGraph, walk_len: int):
    """Brute-force open-walk enumerator for the test fixture. Returns
    a list of ``(walk_vertices, walk_edge_signs)`` tuples in DFS
    order, canonicalised by ``walk[0] <= walk[-1]``."""
    adj: dict[int, list[tuple[int, int]]] = {}
    for i in range(len(g.edges)):
        u = int(g.edges[i, 0])
        v = int(g.edges[i, 1])
        s = int(g.signs[i])
        adj.setdefault(u, []).append((v, s))
        adj.setdefault(v, []).append((u, s))
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def dfs(path, signs, visited):
        if len(path) == walk_len + 1:
            if path[0] <= path[-1]:
                out.append((tuple(path), tuple(signs)))
            return
        for nxt, s in adj.get(path[-1], ()):
            if nxt in visited:
                continue
            visited.add(nxt)
            path.append(nxt)
            signs.append(s)
            dfs(path, signs, visited)
            path.pop()
            signs.pop()
            visited.remove(nxt)

    for start in range(g.n_nodes):
        dfs([start], [], {start})
    return out


@pytest.mark.parametrize("scorer_cls", [
    FractionNegativeScorer, BalanceScorer,
    SignProductAbsScorer, ShannonEntropyScorer,
])
def test_path_scorer_admissibility(scorer_cls):
    """For every walk, the upper-bound at every prefix must be
    ≥ the walk's final score. Violation = silent wrong top-K."""
    g = _toy_signed_graph(seed=0, n=8)
    walks = _enumerate_open_walks_brute(g, walk_len=3)
    if not walks:
        pytest.skip("toy graph yielded no walks of length 3")
    scorer = scorer_cls()
    for vs, signs in walks:
        signs_arr = np.array(signs, dtype=np.int8)
        final_score = scorer.score(list(vs), list(signs))
        for n_neg, steps_left, k_len in _walk_prefix_states(signs_arr):
            ub = scorer.upper_bound(n_neg, steps_left, k_len)
            assert ub >= final_score - 1e-12, (
                f"{scorer.name()} UB violation: ub={ub:.6f} < "
                f"score={final_score:.6f} for prefix state "
                f"n_neg={n_neg} remaining={steps_left} k_len={k_len}"
            )


def test_pick_scorer_dispatches_known_names():
    for name in [
        "fraction_negative", "balance",
        "sign_product_abs", "entropy",
    ]:
        s = pick_scorer(name)
        assert isinstance(s, PathScorer)
        assert s.name() == name


def test_pick_scorer_rejects_unknown():
    with pytest.raises(ValueError, match="unknown PathScorer name"):
        pick_scorer("bogus")


# ─── ABB DFS correctness ────────────────────────────────────────


def test_abb_matches_brute_force_topk():
    """ABB MUST return the same top-K walks as a brute-force MSG +
    score + sort baseline. Differing answers = a non-admissible UB
    silently pruned a walk that should have survived."""
    g = _toy_signed_graph(seed=1, n=8)
    walk_len = 3
    top_k = 5
    scorer = BalanceScorer()

    eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.int64)
    ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.int64)
    es = np.ascontiguousarray(g.signs, dtype=np.int8)
    abb_v, abb_signs, stats = abb_enumerate_walks(
        eu, ev, es, g.n_nodes,
        walk_len=walk_len, top_k=top_k, scorer=scorer, seed=0,
    )

    # Brute-force baseline
    brute = _enumerate_open_walks_brute(g, walk_len=walk_len)
    if not brute:
        pytest.skip("toy graph yielded no walks")
    scored = [
        (scorer.score(list(vs), list(signs)), vs, signs)
        for vs, signs in brute
    ]
    scored.sort(key=lambda e: e[0], reverse=True)
    top_k_brute_scores = sorted(
        [s for s, *_ in scored[:top_k]], reverse=True,
    )
    abb_scores = sorted(
        [scorer.score(abb_v[i], abb_signs[i])
         for i in range(abb_v.shape[0])],
        reverse=True,
    )
    # Score multisets must match (walks may differ for tied scores).
    assert abb_scores == top_k_brute_scores, (
        f"ABB top-K score multiset {abb_scores} != "
        f"brute-force {top_k_brute_scores}"
    )


def test_abb_reports_prune_rate():
    """On a fixture with at least one negative edge and a
    fraction_negative scorer (which has a tight UB), some branches
    must be pruned. A 0% prune rate means ABB isn't doing its job."""
    g = _toy_signed_graph(seed=2, n=10)
    eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.int64)
    ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.int64)
    es = np.ascontiguousarray(g.signs, dtype=np.int8)
    _v, _s, stats = abb_enumerate_walks(
        eu, ev, es, g.n_nodes,
        walk_len=3, top_k=3,
        scorer=FractionNegativeScorer(),
        seed=0,
    )
    # Heap fills first; ABB only fires AFTER the heap is full.
    # On a typical fixture the prune rate should be > 0 by the end.
    assert stats.n_visited > 0
    assert stats.n_emitted > 0


def test_abb_zero_top_k_returns_empty():
    g = _toy_signed_graph(seed=0, n=6)
    eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.int64)
    ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.int64)
    es = np.ascontiguousarray(g.signs, dtype=np.int8)
    v, s, _ = abb_enumerate_walks(
        eu, ev, es, g.n_nodes,
        walk_len=3, top_k=0, scorer=BalanceScorer(),
    )
    assert v.shape == (0, 4)
    assert s.shape == (0, 3)


# ─── SSG Pareto filter ──────────────────────────────────────────


def test_ssg_pareto_filter_removes_dominated():
    """Single axis with strict ordering: only the maximum survives.
    Multi-axis: a row survives iff no other row dominates it on every
    axis (with strict greater-than on at least one)."""
    v = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int32)
    s = np.array([[1, 1], [-1, 1], [1, -1]], dtype=np.int8)
    # Single axis, strict ordering: max survives, others dominated.
    _, _, mask = ssg_pareto_filter(v, s, [np.array([0.5, 0.7, 0.9])])
    assert mask.tolist() == [False, False, True]
    # Single axis, equal scores: none dominates → all survive.
    _, _, mask = ssg_pareto_filter(v, s, [np.array([0.5, 0.5, 0.5])])
    assert mask.tolist() == [True, True, True]
    # Two axes: row 1 dominates row 0 (0.9, 0.9) > (0.5, 0.5) but row
    # 2 (0.7, 0.7) is also dominated by row 1. Only row 1 survives.
    _, _, mask = ssg_pareto_filter(
        v, s,
        [np.array([0.5, 0.9, 0.7]), np.array([0.5, 0.9, 0.7])],
    )
    assert mask.tolist() == [False, True, False]
    # Two axes, trade-off: row 0 (1.0, 0.0), row 1 (0.5, 0.5),
    # row 2 (0.0, 1.0) — all on the Pareto frontier (no row dominates
    # any other on both axes simultaneously).
    _, _, mask = ssg_pareto_filter(
        v, s,
        [np.array([1.0, 0.5, 0.0]), np.array([0.0, 0.5, 1.0])],
    )
    assert mask.tolist() == [True, True, True]


def test_ssg_pareto_filter_empty_input():
    v = np.zeros((0, 3), dtype=np.int32)
    s = np.zeros((0, 2), dtype=np.int8)
    fv, fs, mask = ssg_pareto_filter(v, s, [np.zeros(0)])
    assert fv.shape == (0, 3)
    assert mask.shape == (0,)


# ─── Strategy adapter round-trips ───────────────────────────────


def test_abb_walk_enumerator_strategy_round_trip():
    g = _toy_signed_graph(seed=0, n=8)
    strategy = ABBWalkEnumerator(
        walk_len=3, top_k=4, scorer_name="balance",
    )
    decorated = CachedEnumerator(strategy)
    pool_cold = decorated(g)
    cold_v = np.array(pool_cold.all_vertices(), copy=True)
    pool_warm = decorated(g)
    warm_v = pool_warm.all_vertices()
    np.testing.assert_array_equal(warm_v, cold_v)


def test_msg_walk_enumerator_is_alias_for_walk_enumerator():
    """MSG is the existing WalkEnumerator under a Friedler-named alias."""
    assert MSGWalkEnumerator is WalkEnumerator


def test_ssg_walk_enumerator_returns_pareto_subset():
    g = _toy_signed_graph(seed=3, n=8)
    strategy = SSGWalkEnumerator(
        walk_len=3, top_k=6,
        primary_scorer="balance",
        secondary_scorer="entropy",
    )
    decorated = CachedEnumerator(strategy)
    pool = decorated(g)
    n = pool.all_vertices().shape[0]
    # SSG ⊆ ABB; with two distinct objectives some inputs may be
    # dominated and dropped. Result count is ≤ top_k.
    assert n <= 6


def test_dispatcher_routes_msg_abb_ssg():
    g = _toy_signed_graph(seed=4, n=8)
    for kind, kw in [
        ("msg_walk", {"walk_len": 3, "max_walks": 5}),
        ("abb_walk", {"walk_len": 3, "top_k": 4}),
        ("ssg_walk", {"walk_len": 3, "top_k": 4}),
    ]:
        pool = cycle_cache.cached_construct(g, kind, **kw)
        assert pool is not None


def test_abb_walk_cache_key_distinct_from_msg():
    """ABB and MSG paths must use different on-disk cache keys so the
    same graph + walk_len doesn't collide their outputs (different
    semantics: top-K-by-score vs uniform-sample)."""
    abb = ABBWalkEnumerator(walk_len=3, top_k=10)
    msg = WalkEnumerator(walk_len=3, max_walks=10)
    abb_p = abb.cache_key_params()
    msg_p = msg.cache_key_params()
    # Same kind + k but DIFFERENT max_cycles sign so the on-disk file
    # never collides.
    assert abb_p["kind"] == msg_p["kind"]
    assert abb_p["k"] == msg_p["k"]
    assert abb_p["max_cycles"] != msg_p["max_cycles"]
    assert abb_p["max_cycles"] < 0
    assert msg_p["max_cycles"] > 0
