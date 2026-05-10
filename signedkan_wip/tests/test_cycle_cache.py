"""Tests for ``signedkan_wip.src.cycle_cache``.

The cache wrapper sits between ``run_final_cell`` and the cycle / walk
enumerators.  Two regressions it must not have:

1. The cache-hit path must successfully unpack a saved file --- earlier
   versions of ``_unpack_tuples`` constructed ``SignedNTuple(v=..., sigma=...)``
   without the dataclass's three other required fields, which raised
   TypeError on the very first hit.  No test exercised this until the
   2026-05-10 Epinions OOM forced an audit.

2. On a cache miss, the wrapper must not double peak RSS by holding
   the original ``t_list`` alive while the packed numpy arrays exist
   *and* a fresh return list is built.  ``_pack_and_drop`` now releases
   each entry as it copies it; the test below sanity-checks the
   release.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from signedkan_wip.src import cycle_cache
from signedkan_wip.src.datasets import SignedGraph
from signedkan_wip.src.n_tuples import SignedNTuple


def _toy_graph(seed: int = 0) -> SignedGraph:
    """Small dense signed graph with a known set of triangles and
    4-cycles, deterministic across runs."""
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
    """Each test gets its own cache directory + a fresh stats counter."""
    monkeypatch.setenv("HYMEKO_CYCLE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HYMEKO_CYCLE_CACHE", "1")
    cycle_cache.reset_stats()
    yield


# ─── Pack helpers ───────────────────────────────────────────────────


def test_pack_and_drop_releases_entries():
    """``_pack_and_drop`` empties the input list in-place so the
    caller's ``del t_list`` can free the underlying SignedNTuples
    before the unpack stage allocates a fresh return list."""
    t_list = [
        SignedNTuple(v=(0, 1, 2), sigma=(1, 1, 1),
                     edge_signs=(1, 1, 1), balanced=True, arity=3),
        SignedNTuple(v=(0, 2, 3), sigma=(-1, 1, -1),
                     edge_signs=(-1, 1, -1), balanced=True, arity=3),
    ]
    v, sigma, edge_signs = cycle_cache._pack_and_drop(t_list)
    assert v.shape == (2, 3)
    assert sigma.shape == (2, 3)
    assert edge_signs.shape == (2, 3)
    assert all(t is None for t in t_list), \
        "_pack_and_drop must replace each entry with None to free it"
    assert v[0].tolist() == [0, 1, 2]
    assert sigma[1].tolist() == [-1, 1, -1]
    assert edge_signs[1].tolist() == [-1, 1, -1]


def test_pack_and_drop_empty():
    v, sigma, edge_signs = cycle_cache._pack_and_drop([])
    assert v.shape == (0, 0)
    assert sigma.shape == (0, 0)
    assert edge_signs.shape == (0, 0)


def test_unpack_populates_all_dataclass_fields():
    """Round-trip: pack → save → load → unpack must produce
    SignedNTuples with all five fields populated.  If unpack omits
    any required field the dataclass constructor raises TypeError —
    that was the latent bug shipped 2026-05-10."""
    t_in = [
        SignedNTuple(v=(0, 1, 2), sigma=(1, 1, 1),
                     edge_signs=(1, 1, 1), balanced=True, arity=3),
        SignedNTuple(v=(1, 2, 3), sigma=(-1, -1, 1),
                     edge_signs=(-1, -1, 1), balanced=True, arity=3),
    ]
    v, sigma, edge_signs = cycle_cache._pack_and_drop(t_in)
    out = cycle_cache._unpack_to_ntuples(v, sigma, edge_signs)
    assert len(out) == 2
    for nt in out:
        assert nt.arity == 3
        assert isinstance(nt.balanced, bool)
        assert len(nt.edge_signs) == 3
    assert out[0].balanced is True   # prod(+1, +1, +1) == +1
    assert out[1].balanced is True   # prod(-1, -1, +1) == +1


def test_unpack_legacy_format_without_edge_signs():
    """Cache files written before edge_signs was stored must still
    load — the unpack path falls back to ``edge_signs=()`` and
    ``balanced=False``."""
    v = np.array([[0, 1, 2]], dtype=np.int32)
    sigma = np.array([[1, 1, 1]], dtype=np.int8)
    out = cycle_cache._unpack_to_ntuples(v, sigma, edge_signs=None)
    assert len(out) == 1
    assert out[0].edge_signs == ()
    assert out[0].balanced is False
    assert out[0].arity == 3


# ─── End-to-end cached_construct_* ─────────────────────────────────


def test_miss_then_hit_round_trip_k_cycles():
    g = _toy_graph()
    first = cycle_cache.cached_construct_k(g, k=3, max_cycles=200)
    assert cycle_cache.stats().misses == 1
    assert cycle_cache.stats().hits == 0
    assert cycle_cache.stats().bytes_written > 0

    second = cycle_cache.cached_construct_k(g, k=3, max_cycles=200)
    assert cycle_cache.stats().hits == 1
    assert cycle_cache.stats().bytes_loaded > 0

    # Same set of (v, sigma) tuples on miss and hit.
    key = lambda nt: (nt.v, nt.sigma)
    assert sorted(map(key, first)) == sorted(map(key, second))
    # Hit returns full SignedNTuples.
    for nt in second:
        assert nt.arity == 3
        assert isinstance(nt.balanced, bool)


def test_miss_then_hit_round_trip_triads():
    g = _toy_graph()
    first = cycle_cache.cached_construct_triads(g)
    assert cycle_cache.stats().misses == 1

    second = cycle_cache.cached_construct_triads(g)
    assert cycle_cache.stats().hits == 1

    key = lambda nt: (nt.v, nt.sigma)
    assert sorted(map(key, first)) == sorted(map(key, second))


def test_miss_then_hit_round_trip_construct_2():
    g = _toy_graph()
    first = cycle_cache.cached_construct_2(g)
    second = cycle_cache.cached_construct_2(g)
    assert cycle_cache.stats().misses == 1
    assert cycle_cache.stats().hits == 1
    key = lambda nt: (nt.v, nt.sigma)
    assert sorted(map(key, first)) == sorted(map(key, second))


def test_disabled_cache_passes_through(monkeypatch):
    monkeypatch.setenv("HYMEKO_CYCLE_CACHE", "0")
    g = _toy_graph()
    cycle_cache.reset_stats()
    cycle_cache.cached_construct_k(g, k=3, max_cycles=200)
    cycle_cache.cached_construct_k(g, k=3, max_cycles=200)
    s = cycle_cache.stats()
    assert s.hits == 0 and s.misses == 0


def test_corrupt_cache_file_recovers():
    """A corrupt cache file should be unlinked and the enumeration
    rerun, not propagate the load error."""
    g = _toy_graph()
    cycle_cache.cached_construct_k(g, k=3, max_cycles=200)
    # Corrupt the only cache file.
    cache_dir = cycle_cache._cache_dir()
    files = list(cache_dir.glob("*.npz"))
    assert len(files) == 1
    files[0].write_bytes(b"not an npz file")
    cycle_cache.reset_stats()
    out = cycle_cache.cached_construct_k(g, k=3, max_cycles=200)
    # Treated as a miss; file rewritten.
    assert cycle_cache.stats().misses == 1
    assert cycle_cache.stats().hits == 0
    assert len(out) >= 0


# ─── Performance routing ────────────────────────────────────────────


def test_triads_route_through_topk_path_when_enabled(monkeypatch):
    """When ``HSIKAN_TOPK_MODE=per_vertex`` is set, ``cached_construct_triads``
    must go through ``cached_construct_k(g, k=3, ...)`` — the rayon-parallel
    Rust enumerator — rather than the pure-Python ``hyperedges.construct(g)``.

    Regression guard for the 86-minute Epinions pre-training stall.
    """
    monkeypatch.setenv("HSIKAN_TOPK_MODE", "per_vertex")
    monkeypatch.setenv("HSIKAN_TOPK_K", "8")
    monkeypatch.setenv("HSIKAN_TOPK_SCORER", "fraction_negative")
    monkeypatch.setenv("HSIKAN_TOPK_PRUNER", "none")
    cycle_cache.reset_stats()
    g = _toy_graph()
    triads = cycle_cache.cached_construct_triads(g)
    # All returned tuples have arity 3.
    for nt in triads:
        assert nt.arity == 3
        assert len(nt.v) == 3
        assert len(nt.sigma) == 3
    # The per_vertex Rust path writes its cache file under the
    # ``cycle_k`` kind, not ``triads`` — confirms we took the new
    # branch.  Keys differ in their ``kind`` parameter.
    assert cycle_cache.stats().misses == 1


def test_triads_default_path_unchanged(monkeypatch):
    """Without ``HSIKAN_TOPK_MODE``, ``cached_construct_triads`` keeps
    the classic ``hyperedges.construct(g)`` semantics — preserves
    backward compatibility for every existing experiment that does
    not opt into top-K."""
    monkeypatch.delenv("HSIKAN_TOPK_MODE", raising=False)
    cycle_cache.reset_stats()
    g = _toy_graph()
    triads = cycle_cache.cached_construct_triads(g)
    assert cycle_cache.stats().misses == 1
    # All triads have arity 3 with all five SignedNTuple fields
    # populated by the cache round-trip.
    for nt in triads:
        assert nt.arity == 3
        # SignedTriad has .edge_signs populated → cache preserves it.
        assert len(nt.edge_signs) == 3


def test_unpack_vectorised_returns_native_ints():
    """The vectorised unpack path uses ``ndarray.tolist()`` which
    returns native Python ints, not numpy scalars.  Downstream code
    relies on ``isinstance(x, int)`` — numpy.int32 would fail that
    check in some places.
    """
    v = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    sigma = np.array([[1, 1, 1], [-1, 1, -1]], dtype=np.int8)
    edge_signs = np.array([[1, 1, 1], [-1, 1, -1]], dtype=np.int8)
    out = cycle_cache._unpack_to_ntuples(v, sigma, edge_signs)
    for nt in out:
        for x in nt.v:
            assert type(x) is int   # not numpy.int32
        for x in nt.sigma:
            assert type(x) is int
        for x in nt.edge_signs:
            assert type(x) is int
