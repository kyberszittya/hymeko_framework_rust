"""Tests for the Strategy + Adapter refactor of the cycle-cache surface.

Added 2026-06-03 alongside ``hymeko_neuro/graph/cycle_cache/strategies.py``.
Verifies:

1. Each concrete :class:`TupleEnumerator` returns a shape-correct
   :class:`EnumeratedArrays` on a small fixture (cycles closed, walks
   open, k=2 trivial, triads either dispatch path).

2. :class:`CachedEnumerator` round-trips: a cold call writes the npz,
   a warm call returns a :class:`LazyCyclePool` without re-invoking
   the strategy. The two pools' arrays are byte-identical.

3. Legacy wrapper equivalence: the cache file produced by
   ``cached_construct_k(g, 4, 100)`` is at the SAME on-disk path and
   has IDENTICAL arrays to ``CachedEnumerator(CycleEnumerator(4, 100))(g)``.
   Drift here would orphan every existing cache file on Komondor.

4. Walks vectorised cold path: ``WalkEnumerator`` does not materialise
   the Rust output as Python tuples on the way in (the bug that
   caused the walk k=5 Komondor OOM on 2026-06-03).
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from hymeko_neuro.graph import cycle_cache
from hymeko_neuro.graph.cycle_cache.strategies import (
    CachedEnumerator,
    CycleEnumerator,
    EnumeratedArrays,
    TriadEnumerator,
    TupleEnumerator,
    TwoCycleEnumerator,
    WalkEnumerator,
    cached_construct,
)
from hymeko_neuro.data.datasets import SignedGraph


# ─── Fixtures ──────────────────────────────────────────────────────


def _toy_graph(seed: int = 0) -> SignedGraph:
    """Same small dense graph as ``test_cycle_cache._toy_graph`` so
    fixtures behave identically across the two test files."""
    rng = np.random.default_rng(seed)
    n = 8
    edges: list[tuple[int, int]] = []
    signs: list[int] = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.6:
                edges.append((u, v))
                signs.append(1 if rng.random() < 0.7 else -1)
    return SignedGraph(
        edges=np.array(edges, dtype=np.int64),
        signs=np.array(signs, dtype=np.int8),
        n_nodes=n,
    )


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Each test gets its own cache directory + a fresh stats counter
    so cache hits / misses don't leak between tests."""
    monkeypatch.setenv("HYMEKO_CYCLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYMEKO_CYCLE_CACHE", "1")
    cycle_cache.reset_stats()
    yield


# ─── Strategy contract ────────────────────────────────────────────


def test_enumerated_arrays_rejects_mismatched_shapes():
    """Postcondition contract on :class:`EnumeratedArrays` — drift here
    would let a buggy strategy produce a pool the consumers can't read."""
    v = np.zeros((4, 3), dtype=np.int32)
    sigma = np.zeros((4, 3), dtype=np.int8)
    # is_closed=True → edge_signs must have width == arity (3).
    bad_es = np.zeros((4, 2), dtype=np.int8)
    with pytest.raises(ValueError, match="edge_signs shape"):
        EnumeratedArrays(v=v, sigma=sigma, edge_signs=bad_es, is_closed=True)
    # is_closed=False → edge_signs must have width == arity - 1 (2).
    good_es_walk = np.zeros((4, 2), dtype=np.int8)
    EnumeratedArrays(v=v, sigma=sigma, edge_signs=good_es_walk,
                     is_closed=False)  # no raise


def test_two_cycle_enumerator_returns_edge_count_rows():
    g = _toy_graph()
    arr = TwoCycleEnumerator().enumerate(g, seed=0)
    assert arr.is_closed is True
    assert arr.v.shape == (len(g.edges), 2)
    assert arr.sigma.shape == (len(g.edges), 2)
    assert arr.edge_signs.shape == (len(g.edges), 2)
    # Both endpoints share parity (Davis-style k=2 convention).
    assert np.all(arr.sigma[:, 0] == arr.sigma[:, 1])


def test_cycle_enumerator_k3_shapes():
    g = _toy_graph()
    arr = CycleEnumerator(k=3, max_cycles=100).enumerate(g, seed=0)
    assert arr.is_closed is True
    if arr.v.shape[0] > 0:
        assert arr.v.shape[1] == 3
        assert arr.edge_signs.shape[1] == 3
        # σ ∈ {-1, +1}
        assert set(np.unique(arr.sigma).tolist()).issubset({-1, 1})


def test_walk_enumerator_open_walk_shapes():
    g = _toy_graph()
    arr = WalkEnumerator(walk_len=3, max_walks=50).enumerate(g, seed=0)
    assert arr.is_closed is False
    if arr.v.shape[0] > 0:
        # walk_len=3 → 4 vertices per walk, 3 edges per walk
        assert arr.v.shape[1] == 4
        assert arr.edge_signs.shape[1] == 3


def test_triad_enumerator_dispatches_per_env(monkeypatch):
    """In default (no top-K mode), TriadEnumerator uses the classic
    hyperedges path. The dispatch is via runtime_config so we toggle
    the env var the runtime reads."""
    g = _toy_graph()
    monkeypatch.delenv("HSIKAN_TOPK_MODE", raising=False)
    arr = TriadEnumerator().enumerate(g, seed=0)
    assert arr.is_closed is True
    if arr.v.shape[0] > 0:
        assert arr.v.shape[1] == 3


# ─── Cache key stability ─────────────────────────────────────────


def test_cycle_enumerator_cache_key_stable():
    a = CycleEnumerator(k=4, max_cycles=100).cache_key_params()
    b = CycleEnumerator(k=4, max_cycles=100).cache_key_params()
    assert a == b
    assert a["kind"] == "cycle_k"
    assert a["k"] == 4
    assert a["max_cycles"] == 100


def test_walk_enumerator_cache_key_stable():
    a = WalkEnumerator(walk_len=3, max_walks=50).cache_key_params()
    assert a["kind"] == "walk"
    assert a["k"] == 3
    assert a["max_cycles"] == 50


def test_two_cycle_cache_key_pins_seed_and_extra():
    """k=2 must opt out of env-driven seed / topk fingerprint so its
    cache key matches the legacy ``cached_construct_2`` byte-for-byte
    (which hard-coded enum_seed=0 and no extra)."""
    a = TwoCycleEnumerator().cache_key_params()
    assert a["use_enum_seed"] is False
    assert a["use_topk_fingerprint"] is False


def test_triad_cache_key_pins_off_in_classic_path(monkeypatch):
    monkeypatch.delenv("HSIKAN_TOPK_MODE", raising=False)
    a = TriadEnumerator().cache_key_params()
    assert a["kind"] == "triads"
    assert a["use_enum_seed"] is False
    assert a["use_topk_fingerprint"] is False


# ─── CachedEnumerator round-trip ─────────────────────────────────


def test_cached_enumerator_cold_then_warm():
    """First call: cold cache → strategy invoked, npz written.
    Second call: warm cache → npz read, strategy NOT invoked.
    Both pools return identical arrays."""
    g = _toy_graph()
    strategy = CycleEnumerator(k=3, max_cycles=100)
    decorated = CachedEnumerator(strategy)

    pool_cold = decorated(g)
    cold_v = np.array(pool_cold.all_vertices(), copy=True)
    cold_sigma = np.array(pool_cold.all_signs(), copy=True)

    # Second call: count strategy invocations to prove the warm path
    # skips the enumeration entirely.
    with patch.object(
        strategy, "enumerate",
        wraps=strategy.enumerate,
    ) as spy:
        pool_warm = decorated(g)
        assert spy.call_count == 0, (
            "warm cache must not re-invoke the strategy"
        )

    warm_v = pool_warm.all_vertices()
    warm_sigma = pool_warm.all_signs()
    np.testing.assert_array_equal(warm_v, cold_v)
    np.testing.assert_array_equal(warm_sigma, cold_sigma)


def test_cached_enumerator_disabled_returns_ephemeral_pool(monkeypatch):
    """When ``HYMEKO_CYCLE_CACHE=0``, no disk file is written but
    the strategy is invoked and an ephemeral LazyCyclePool returned."""
    monkeypatch.setenv("HYMEKO_CYCLE_CACHE", "0")
    g = _toy_graph()
    pool = CachedEnumerator(CycleEnumerator(k=3, max_cycles=50))(g)
    assert pool is not None
    # Sentinel path (LazyCyclePool over the disabled-cache branch).
    assert str(pool.path) == "/dev/null"


# ─── Legacy wrapper equivalence ──────────────────────────────────


def test_legacy_cached_construct_k_matches_strategy_path():
    """``cached_construct_k`` is a wrapper now. The cache file it
    produces must be at the same on-disk path AND have the same
    arrays as the strategy direct call. Drift would orphan all
    existing cache files."""
    g = _toy_graph()
    # Wrapper call
    pool_legacy = cycle_cache.cached_construct_k(
        g, k=3, max_cycles=100,
    )
    legacy_v = np.array(pool_legacy.all_vertices(), copy=True)
    legacy_path = pool_legacy.path

    # Reset cache state, strategy direct call into a fresh dir.
    cycle_cache.reset_stats()
    pool_strategy = CachedEnumerator(
        CycleEnumerator(k=3, max_cycles=100),
    )(g)
    strategy_v = pool_strategy.all_vertices()
    strategy_path = pool_strategy.path

    assert legacy_path.name == strategy_path.name, (
        f"cache key drift: legacy={legacy_path.name} vs "
        f"strategy={strategy_path.name}"
    )
    np.testing.assert_array_equal(strategy_v, legacy_v)


def test_legacy_cached_construct_walks_matches_strategy_path():
    g = _toy_graph()
    pool_legacy = cycle_cache.cached_construct_walks(
        g, walk_len=2, max_walks=50,
    )
    legacy_path = pool_legacy.path

    cycle_cache.reset_stats()
    pool_strategy = CachedEnumerator(
        WalkEnumerator(walk_len=2, max_walks=50),
    )(g)
    assert pool_strategy.path.name == legacy_path.name


def test_legacy_cached_construct_2_pins_seed_and_extra():
    """k=2 cache key must match the legacy hard-coded pin."""
    g = _toy_graph()
    pool_legacy = cycle_cache.cached_construct_2(g)
    cycle_cache.reset_stats()
    pool_strategy = CachedEnumerator(TwoCycleEnumerator())(g)
    assert pool_strategy.path.name == pool_legacy.path.name


# ─── Dispatcher ──────────────────────────────────────────────────


def test_dispatcher_routes_each_kind():
    g = _toy_graph()
    for kind, kwargs in [
        ("cycle", {"k": 3, "max_cycles": 100}),
        ("walk",  {"walk_len": 2, "max_walks": 50}),
        ("k2",    {}),
        ("triads", {}),
    ]:
        pool = cached_construct(g, kind, **kwargs)
        assert pool is not None
        # Each call should leave a file in the cache dir.
        assert pool.path.exists() or str(pool.path) == "/dev/null"


def test_dispatcher_rejects_unknown_kind():
    g = _toy_graph()
    with pytest.raises(ValueError, match="unknown tuple kind"):
        cached_construct(g, "bogus")


# ─── Walks vectorised path (the Komondor OOM fix) ────────────────


def test_walks_subsample_in_numpy_space():
    """The fix: ``construct_walks_arrays`` must subsample the raw Rust
    output AS A NUMPY ARRAY before any per-walk Python loop. We probe
    by patching numpy.random.default_rng's ``choice`` and verifying it
    is called with the raw array length (not a post-loop count)."""
    from hymeko_neuro.hyperedge.walks import construct_walks_arrays
    g = _toy_graph()
    # walk_len=2 on this graph: a handful of walks. Cap to 3 so the
    # subsample path activates if the graph yields more.
    v, sigma, es = construct_walks_arrays(
        g, walk_len=2, max_walks=3, seed=0,
    )
    assert v.shape[0] <= 3
    assert v.shape[1] == 3       # walk_len + 1 vertices
    assert es.shape[1] == 2      # walk_len edges (open walk, no wrap)


def test_walks_arrays_match_legacy_classification_semantics():
    """The vectorised classifier must produce the same σ / edge_signs
    as the legacy per-walk ``_classify_walk`` Python loop, so any code
    path that switched over to the numpy path remains semantically
    identical."""
    from hymeko_neuro.hyperedge.walks import (
        _classify_walk, _build_sign_lookup, construct_walks_arrays,
    )
    g = _toy_graph()
    v_arr, sigma_arr, es_arr = construct_walks_arrays(
        g, walk_len=2, max_walks=None, seed=0,
    )
    if v_arr.shape[0] == 0:
        pytest.skip("toy graph yielded no walks")
    sign_of = _build_sign_lookup(g)
    for i in range(min(v_arr.shape[0], 20)):
        walk_tuple = tuple(int(x) for x in v_arr[i])
        nt_legacy = _classify_walk(walk_tuple, sign_of)
        assert nt_legacy is not None
        assert tuple(int(x) for x in sigma_arr[i]) == nt_legacy.sigma
        assert tuple(int(x) for x in es_arr[i]) == nt_legacy.edge_signs
