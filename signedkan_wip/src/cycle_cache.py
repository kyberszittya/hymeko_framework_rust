"""On-disk cache for cycle / walk enumerations across seeds.

The cycle (or walk) space of a signed graph at fixed
``(k, max_cycles)`` is independent of the model seed.  In a 5-seed
ablation we currently re-enumerate this space 5 times, paying the
full DFS + classification cost on every seed.  This module hashes
the inputs that actually matter and serves the result from a disk
cache on hits.

Behavioural change vs the un-cached path
----------------------------------------
With caching ON, all model seeds in a run see the *same*
``SignedNTuple`` list at a given ``(graph, k, cap)`` --- in the
un-cached path, when ``len(cycles) > cap``, each seed got a
seed-specific reservoir sample of the full cycle space.  Caching
fixes the sample to a sentinel ``HYMEKO_CYCLE_ENUM_SEED``
(default ``0``) shared across model seeds.  Model-seed variance is
preserved through SGD/init randomness --- the cycle subsample is
no longer a source of seed-to-seed variance.

Opt-in via ``HYMEKO_CYCLE_CACHE=1``.  Cache directory defaults to
``~/.cache/hymeko/cycles_v1``; override with
``HYMEKO_CYCLE_CACHE_DIR``.

Usage
-----
::
    from .cycle_cache import cached_construct_k, cached_construct_walks

    t_k = cached_construct_k(g, k=4, max_cycles=200_000, model_seed=0)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

# Lazy import (avoid loading torch/triton when only the cache is needed).
def _import_n_tuples():
    from . import n_tuples
    return n_tuples


def _import_walks():
    from . import walks
    return walks


# ─── Cache directory ────────────────────────────────────────────────


def _cache_dir() -> pathlib.Path:
    base = os.environ.get(
        "HYMEKO_CYCLE_CACHE_DIR",
        str(pathlib.Path.home() / ".cache" / "hymeko" / "cycles_v1"),
    )
    p = pathlib.Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_enabled() -> bool:
    return int(os.environ.get("HYMEKO_CYCLE_CACHE", "0")) != 0


def _enum_seed() -> int:
    """Sentinel seed used for enumeration sampling.  Decoupled from the
    model seed so caching can amortise across all seeds in an
    ablation."""
    return int(os.environ.get("HYMEKO_CYCLE_ENUM_SEED", "0"))


def _topk_fingerprint() -> dict[str, str]:
    """Capture env vars that change which cycles are enumerated, so the
    cache key separates default / top-K / balance-pruned / etc.
    runs.  Without this, switching ``HSIKAN_TOPK_MODE`` mid-session
    would serve stale results.

    Also fingerprints the degree-adaptive m_v knobs
    (``HSIKAN_TOPK_M_V_C / _MIN / _MAX``) and the entropy /
    inverse-degree heuristic knobs.  Adding a knob that affects
    enumeration without including it here is a silent
    correctness bug — different params return cached cycles from
    a stale config.
    """
    keys = (
        "HSIKAN_TOPK_MODE",
        "HSIKAN_TOPK_K",
        "HSIKAN_TOPK_SCORER",
        "HSIKAN_TOPK_PRUNER",
        # Degree-adaptive m_v (per_vertex_adaptive mode).
        "HSIKAN_TOPK_M_V_C",
        "HSIKAN_TOPK_M_V_MIN",
        "HSIKAN_TOPK_M_V_MAX",
        # Entropy / hybrid heuristic mode.
        "HSIKAN_TOPK_HEURISTIC",
        "HSIKAN_TOPK_HYBRID_ALPHA",
        "HSIKAN_TOPK_SIGNAL",
    )
    return {k: os.environ.get(k, "") for k in keys}


# ─── Cache key ──────────────────────────────────────────────────────


def _hash_graph(g) -> str:
    """Stable hash of a SignedGraph's structural identity.  Sensitive
    to edge set, signs, and n_nodes; insensitive to the in-memory
    Python dtype."""
    edges = np.ascontiguousarray(g.edges, dtype=np.int64)
    signs = np.ascontiguousarray(g.signs, dtype=np.int8)
    h = hashlib.sha256()
    h.update(b"hymeko_cycle_cache_v1\n")
    h.update(f"n_nodes={g.n_nodes}\n".encode())
    h.update(f"n_edges={edges.shape[0]}\n".encode())
    h.update(edges.tobytes())
    h.update(signs.tobytes())
    return h.hexdigest()[:16]


def _cache_key(graph_hash: str, kind: str, k: int,
                max_cycles: int | None, enum_seed: int,
                extra: dict[str, Any] | None = None) -> str:
    parts = {
        "graph": graph_hash,
        "kind": kind,
        "k": int(k),
        "max_cycles": -1 if max_cycles is None else int(max_cycles),
        "enum_seed": int(enum_seed),
    }
    if extra:
        parts.update(extra)
    payload = json.dumps(parts, sort_keys=True).encode()
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()[:24]


# ─── Pack / unpack SignedNTuple lists ───────────────────────────────


def _pack_and_drop(t_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serialize a list[SignedNTuple] (or SignedTriad) to numpy arrays,
    releasing each entry as it is copied so peak RSS during the pack is
    bounded by ``len(t_list) + arrays`` instead of ``2 * len(t_list)``.

    On return, ``t_list`` contains ``None`` placeholders — the original
    SignedNTuple objects become eligible for garbage collection. The
    caller should ``del t_list`` to drop the outer list shell as well.

    Output shapes:
        v          : (N, arity)  int32
        sigma      : (N, arity)  int8
        edge_signs : (N, arity)  int8   (zeros if not provided)
    """
    n = len(t_list)
    if n == 0:
        empty_i = np.zeros((0, 0), dtype=np.int32)
        empty_s = np.zeros((0, 0), dtype=np.int8)
        return empty_i, empty_s, empty_s.copy()
    arity = len(t_list[0].v)
    v = np.empty((n, arity), dtype=np.int32)
    sigma = np.empty((n, arity), dtype=np.int8)
    edge_signs = np.zeros((n, arity), dtype=np.int8)
    for i in range(n):
        t = t_list[i]
        v[i] = t.v
        sigma[i] = t.sigma
        es = getattr(t, "edge_signs", None)
        if es is not None and len(es) == arity:
            edge_signs[i] = es
        t_list[i] = None
    return v, sigma, edge_signs


def _unpack_to_ntuples(v: np.ndarray, sigma: np.ndarray,
                        edge_signs: np.ndarray | None):
    """Rebuild list[SignedNTuple] from packed arrays. All five
    SignedNTuple dataclass fields are populated:

    - ``arity`` is derived from ``v.shape[1]``
    - ``edge_signs`` is read from the packed array (or zero-filled if
      missing — backward-compat for cache files written before
      edge_signs was stored)
    - ``balanced`` is derived from ``prod(edge_signs) == +1``; if
      ``edge_signs`` is unavailable, ``balanced`` defaults to False

    Performance: the per-row int casts go through ``ndarray.tolist()``
    (C-level conversion to native Python ints) rather than a
    ``tuple(int(x) for x in row)`` generator — ~10× faster on
    multi-million-cycle Epinions enumerations where the unpack stage
    dominates seed-1..N cache-hit latency.
    """
    from .n_tuples import SignedNTuple
    n = v.shape[0]
    arity = v.shape[1] if v.ndim == 2 else 0
    if n == 0:
        return []
    has_edge_signs = (
        edge_signs is not None
        and edge_signs.ndim == 2
        and edge_signs.shape == v.shape
        and bool(np.any(edge_signs))
    )
    v_lists = v.tolist()
    sigma_lists = sigma.tolist()
    if has_edge_signs:
        es_lists = edge_signs.tolist()
        balanced_arr = (edge_signs.astype(np.int32).prod(axis=1) == 1)
    else:
        es_lists = None
        balanced_arr = None
    out = [None] * n
    for i in range(n):
        if has_edge_signs:
            es = tuple(es_lists[i])
            bal = bool(balanced_arr[i])
        else:
            es = ()
            bal = False
        out[i] = SignedNTuple(
            v=tuple(v_lists[i]),
            sigma=tuple(sigma_lists[i]),
            edge_signs=es,
            balanced=bal,
            arity=arity,
        )
    return out


def _save_packed(path: pathlib.Path, v: np.ndarray, sigma: np.ndarray,
                  edge_signs: np.ndarray) -> None:
    np.savez(path, v=v, sigma=sigma, edge_signs=edge_signs)


def _load_packed(path: pathlib.Path
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    arr = np.load(path)
    v = arr["v"]
    sigma = arr["sigma"]
    edge_signs = arr["edge_signs"] if "edge_signs" in arr.files else None
    return v, sigma, edge_signs


# ─── Public API ─────────────────────────────────────────────────────


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bytes_loaded: int = 0
    bytes_written: int = 0


_STATS = CacheStats()


def stats() -> CacheStats:
    return _STATS


def reset_stats() -> None:
    global _STATS
    _STATS = CacheStats()


def _cache_hit(path: pathlib.Path):
    """Try to load a cache file. Returns the unpacked SignedNTuple
    list on success, or None if the file is corrupt (also unlinks it)
    or missing."""
    if not path.exists():
        return None
    try:
        v, sigma, edge_signs = _load_packed(path)
    except Exception:
        path.unlink(missing_ok=True)
        return None
    _STATS.hits += 1
    _STATS.bytes_loaded += path.stat().st_size
    return _unpack_to_ntuples(v, sigma, edge_signs)


def _cache_miss(path: pathlib.Path, t_list):
    """Pack ``t_list`` (releasing its entries as we go), save to disk,
    drop the original list, and return a fresh SignedNTuple list
    rebuilt from the packed arrays.

    Peak memory cost vs the no-cache path is one set of (v, sigma,
    edge_signs) numpy arrays — never two simultaneous Python lists.
    """
    _STATS.misses += 1
    v, sigma, edge_signs = _pack_and_drop(t_list)
    del t_list
    _save_packed(path, v, sigma, edge_signs)
    _STATS.bytes_written += path.stat().st_size
    return _unpack_to_ntuples(v, sigma, edge_signs)


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
    if os.environ.get("HSIKAN_TOPK_MODE", "").strip() in ("global", "per_vertex"):
        # Same path as c4/c5: Rust per_vertex enumerator with
        # top-K pruning.  ``max_cycles=None`` because the per_vertex
        # path enforces ``n_vertices × m_per_vertex`` directly.
        return cached_construct_k(g, k=3, max_cycles=None, model_seed=0)

    from .hyperedges import construct as _construct_triads
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
