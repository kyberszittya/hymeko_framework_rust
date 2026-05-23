"""Auto-split from cycle_cache.py 2026-05-11 (CLAUDE.md §6.5 #4)."""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np

from ..runtime_config import get_runtime
from .config import (_import_n_tuples, _import_walks, _cache_dir,
                       cache_enabled, _enum_seed, _topk_fingerprint)
from .key import _hash_graph, _cache_key
from .pack import _pack_and_drop, _unpack_to_ntuples
from .format import _cache_format, _save_packed, _load_packed
from .stats import (CacheStats, LazyCyclePool, stats, reset_stats,
                      _cache_hit, _cache_miss)

def cached_construct_k(g, k: int, max_cycles: int | None,
                       model_seed: int = 0, **kwargs):
    """Drop-in replacement for ``n_tuples.construct_k`` with disk cache.

    ``model_seed`` is recorded but does NOT affect the enumeration
    when caching is on (uses ``HYMEKO_CYCLE_ENUM_SEED``).
    """
    nt = _import_n_tuples()
    if not cache_enabled():
        return nt.construct_k(g, k=k, max_cycles=max_cycles,
                               seed=model_seed, **kwargs)
    enum_seed = _enum_seed()
    graph_hash = _hash_graph(g)
    key = _cache_key(graph_hash, "cycle_k", k, max_cycles, enum_seed,
                      extra=_topk_fingerprint())
    path = _cache_dir() / f"{key}.npz"
    hit = _cache_hit(path)
    if hit is not None:
        return hit
    t_list = nt.construct_k(g, k=k, max_cycles=max_cycles,
                              seed=enum_seed, **kwargs)
    return _cache_miss(path, t_list)


def cached_construct_walks(g, walk_len: int, max_walks: int | None,
                            model_seed: int = 0):
    """Drop-in replacement for ``walks.construct_walks`` with disk
    cache."""
    wk = _import_walks()
    if not cache_enabled():
        return wk.construct_walks(g, walk_len=walk_len,
                                   max_walks=max_walks, seed=model_seed)
    enum_seed = _enum_seed()
    graph_hash = _hash_graph(g)
    key = _cache_key(graph_hash, "walk", walk_len, max_walks, enum_seed,
                      extra=_topk_fingerprint())
    path = _cache_dir() / f"{key}.npz"
    hit = _cache_hit(path)
    if hit is not None:
        return hit
    t_list = wk.construct_walks(g, walk_len=walk_len,
                                 max_walks=max_walks, seed=enum_seed)
    return _cache_miss(path, t_list)


def cached_construct_2(g):
    """k=2 cycles are deterministic from the graph; cache them too."""
    nt = _import_n_tuples()
    if not cache_enabled():
        return nt.construct_2(g)
    graph_hash = _hash_graph(g)
    key = _cache_key(graph_hash, "cycle_k", 2, None, 0)
    path = _cache_dir() / f"{key}.npz"
    hit = _cache_hit(path)
    if hit is not None:
        return hit
    t_list = nt.construct_2(g)
    return _cache_miss(path, t_list)


def cached_construct_triads(g):
    """k=3 triads — fast Rust per_vertex path when ``HSIKAN_TOPK_MODE``
    is set, else the classic ``hyperedges.construct(g)`` Python path.

    The Python triad path enumerates every vertex × every neighbour ×
    set-intersection in pure Python — at Epinions scale (131k
    vertices, 841k edges) this is the dominant pre-training cost
    (tens of minutes).  When the caller has opted into top-K cycle
    pruning via ``HSIKAN_TOPK_MODE=per_vertex`` (or ``global``), the
    same env var should also redirect c3 enumeration through the
    rayon-parallel Rust enumerator that c4/c5 already use, keeping
    the cycle space treatment uniform across arities and getting
    triad enumeration off the GIL.

    Returns a list[SignedNTuple] in either branch — all five fields
    populated, so downstream consumers that read ``.balanced`` /
    ``.arity`` (e.g. ``hyperedges.stats``) work transparently.
    """
    if get_runtime().topk.mode in ("global", "per_vertex"):
        # Same path as c4/c5: Rust per_vertex enumerator with
        # top-K pruning.  ``max_cycles=None`` because the per_vertex
        # path enforces ``n_vertices × m_per_vertex`` directly.
        return cached_construct_k(g, k=3, max_cycles=None, model_seed=0)

    from ..core.hyperedges import construct as _construct_triads
    if not cache_enabled():
        return _construct_triads(g)
    graph_hash = _hash_graph(g)
    key = _cache_key(graph_hash, "triads", 3, None, 0)
    path = _cache_dir() / f"{key}.npz"
    hit = _cache_hit(path)
    if hit is not None:
        return hit
    t_list = _construct_triads(g)
    return _cache_miss(path, t_list)


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
