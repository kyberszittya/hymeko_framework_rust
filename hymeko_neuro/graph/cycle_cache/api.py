"""Public ``cached_construct_*`` + ``lazy_load_*`` entries.

Refactored 2026-06-03: the four ``cached_construct_{k,walks,2,triads}``
functions are now thin wrappers over the Strategy + Adapter pair in
:mod:`hymeko_neuro.graph.cycle_cache.strategies`. Cache keys, on-disk
file format, and the LazyCyclePool return type are preserved
byte-for-byte so existing cache files remain valid and external
callers see no surface change.

The refactor closes CLAUDE.md §6.5 anti-patterns #1 (Cartesian
product), #3 (per-experiment scaffold duplication), #5 (new-axis-as-
new-function-name) and — as the first natural consequence — closes
the walk k=5 cold-cache leak that OOM-killed Komondor job 13883876
(MaxRSS 31.97 GB on bitcoin_alpha HSIKAN_MIXED_TUPLES). The legacy
walks cold-cache path materialised every Rust-enumerated walk as a
Python tuple before subsampling; :class:`WalkEnumerator` subsamples
in numpy space.
"""
from __future__ import annotations
import pathlib

from .config import _cache_dir, cache_enabled, _enum_seed, _topk_fingerprint
from .key import _hash_graph, _cache_key
from .stats import LazyCyclePool
from .strategies import (
    CachedEnumerator,
    CycleEnumerator,
    WalkEnumerator,
    TwoCycleEnumerator,
    TriadEnumerator,
)


def cached_construct_k(
    g, k: int, max_cycles: int | None,
    model_seed: int = 0, **kwargs,
):
    """Drop-in replacement for ``n_tuples.construct_k`` with disk cache.

    ``model_seed`` is recorded but does NOT affect the enumeration
    when caching is on (uses ``HYMEKO_CYCLE_ENUM_SEED``).
    """
    strategy = CycleEnumerator(k, max_cycles, **kwargs)
    return CachedEnumerator(strategy, model_seed=model_seed)(g)


def cached_construct_walks(
    g, walk_len: int, max_walks: int | None,
    model_seed: int = 0,
):
    """Drop-in replacement for ``walks.construct_walks`` with disk
    cache. Cold-cache path is now numpy-direct via :class:`WalkEnumerator`
    — closes the Komondor walk k=5 OOM (job 13883876, 2026-06-03)."""
    strategy = WalkEnumerator(walk_len, max_walks)
    return CachedEnumerator(strategy, model_seed=model_seed)(g)


def cached_construct_2(g):
    """k=2 cycles are deterministic from the graph; cache them too."""
    return CachedEnumerator(TwoCycleEnumerator())(g)


def cached_construct_triads(g):
    """k=3 triads — fast Rust per_vertex path when ``HSIKAN_TOPK_MODE``
    is set, else the classic ``hyperedges.construct(g)`` Python path.

    Dispatch is inside :class:`TriadEnumerator` so callers don't
    branch on the env var."""
    return CachedEnumerator(TriadEnumerator())(g)


# ─── Lazy public surface ────────────────────────────────────────────


def lazy_load_construct_k(
    g, k: int, max_cycles: int | None,
    model_seed: int, enum_seed: int = 0,
    directed: bool = False, early_stop: bool = True,
) -> LazyCyclePool | None:
    """Lazy variant of `cached_construct_k`.

    Returns a `LazyCyclePool` handle when the cache is hit (no
    SignedNTuple materialisation), or `None` when the cache is cold.
    Cold-cache enumeration still goes through the eager path; this is
    primarily a memory-saving wrapper for the warm-cache case where
    we want to thread the pool reference without paying the unpack
    cost up front.
    """
    if not cache_enabled():
        return None
    graph_hash = _hash_graph(g)
    key = _cache_key(graph_hash, "cycle_k", k, max_cycles, enum_seed,
                      extra=_topk_fingerprint())
    path = _cache_dir() / f"{key}.npz"
    return LazyCyclePool.from_path(path)


def lazy_load_construct_walks(
    g, walk_len: int, max_walks: int | None,
    model_seed: int, enum_seed: int = 0,
) -> LazyCyclePool | None:
    """Lazy variant of `cached_construct_walks`.  Returns None on
    cold cache; the caller must fall back to the eager path then."""
    if not cache_enabled():
        return None
    graph_hash = _hash_graph(g)
    key = _cache_key(graph_hash, "walk", walk_len, max_walks, enum_seed,
                      extra=_topk_fingerprint())
    path = _cache_dir() / f"{key}.npz"
    return LazyCyclePool.from_path(path)


def cache_path_for_k(
    g, k: int, max_cycles: int | None, enum_seed: int = 0,
) -> "pathlib.Path":
    """Return the cache file path that `cached_construct_k` would
    write to / read from, given the current env-var fingerprint.
    Useful for diagnostics and for tests that need to exercise the
    cache lifecycle deterministically."""
    graph_hash = _hash_graph(g)
    key = _cache_key(graph_hash, "cycle_k", k, max_cycles, enum_seed,
                      extra=_topk_fingerprint())
    return _cache_dir() / f"{key}.npz"
