"""Auto-split from cycle_cache.py 2026-05-11 (CLAUDE.md §6.5 #4)."""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np

from .config import _cache_dir
from .key import _cache_key
from .pack import _pack_and_drop, _unpack_to_ntuples
from .format import _cache_format, _save_packed, _load_packed

# ─── Public API ─────────────────────────────────────────────────────


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bytes_loaded: int = 0
    bytes_written: int = 0


class LazyCyclePool:
    """Lazy handle to a cached cycle pool.

    Wraps the on-disk `.npz` cache: holds the packed (v, sigma,
    edge_signs) numpy arrays in memory but does NOT materialise the
    Python `SignedNTuple` list until `.materialize()` or `.iter()` is
    called.

    For Epinions-scale pools (~500K cycles) the materialised
    SignedNTuple list is the dominant memory cost — ~5x the raw
    numpy arrays.  LazyCyclePool lets callers thread the pool
    reference through the pipeline (model construction, batching
    setup, etc.) without paying that cost.  Materialisation happens
    once at the actual consumption point.

    Public surface:
        len(pool)                — number of cycles, O(1) (no materialise)
        pool.arity()             — k (cycle length), O(1)
        pool.materialize()       — return the SignedNTuple list (eager)
        pool.iter()              — yield SignedNTuple one at a time
        pool.iter_chunks(n)      — yield lists of n SignedNTuples each
        pool.cycle_vertices(c)   — return (k,) array of vertex indices
                                    for cycle c, no materialise
        pool.cycle_signs(c)      — return (k,) array of edge signs
        pool.path                — pathlib.Path of the cache file
    """

    def __init__(
        self, path: "pathlib.Path",
        v: np.ndarray, sigma: np.ndarray,
        edge_signs: np.ndarray | None,
    ):
        self.path = path
        self._v = v
        self._sigma = sigma
        self._edge_signs = edge_signs
        self._materialised: list | None = None

    @classmethod
    def from_path(cls, path: "pathlib.Path") -> "LazyCyclePool | None":
        """Load the packed arrays from disk without materialising the
        SignedNTuple list.  Returns None if the file is missing or
        corrupt (and unlinks the corrupt file)."""
        if not path.exists():
            return None
        try:
            v, sigma, edge_signs = _load_packed(path)
        except Exception:
            path.unlink(missing_ok=True)
            return None
        _STATS.hits += 1
        _STATS.bytes_loaded += path.stat().st_size
        return cls(path, v, sigma, edge_signs)

    def __len__(self) -> int:
        return int(self._v.shape[0])

    def arity(self) -> int:
        """Cycle length k.  O(1)."""
        return int(self._v.shape[1])

    def cycle_vertices(self, idx: int) -> np.ndarray:
        return self._v[idx]

    def cycle_signs(self, idx: int) -> np.ndarray:
        if self._edge_signs is None:
            # Older cache files without edge_signs; reconstruct as
            # sign-of-sigma (which is correct for the way the legacy
            # path packed cycles).
            return self._sigma[idx].astype(np.int8)
        return self._edge_signs[idx]

    def materialize(self) -> list:
        """Build (and cache) the SignedNTuple list.  Subsequent calls
        return the cached list without rebuilding."""
        if self._materialised is None:
            self._materialised = _unpack_to_ntuples(
                self._v, self._sigma, self._edge_signs,
            )
        return self._materialised

    def all_vertices(self) -> np.ndarray:
        """Return the full (N, k) vertex-index array WITHOUT materialising
        SignedNTuple objects. Equivalent to
        ``np.array([t.v for t in pool.materialize()])`` but ~5× less RAM
        because no Python-object list is built. Caller may safely cast
        to int64 if the downstream consumer (e.g. ``torch.from_numpy``)
        needs it."""
        return self._v

    def all_signs(self) -> np.ndarray:
        """Return the full (N, k) per-vertex sign array (int8). For
        downstream consumers that need the sigma channel without
        materialising the SignedNTuple list."""
        return self._sigma

    def all_edge_signs(self) -> np.ndarray | None:
        """Return the full (N, k) edge-sign array or None if the cache
        file pre-dates edge_signs storage."""
        return self._edge_signs

    def __iter__(self):
        """Backward compatibility: ``for t in pool`` yields SignedNTuples
        one at a time (no full materialisation). Allows code paths that
        previously consumed a list to keep working after the cache
        layer transitioned to returning LazyCyclePool (2026-06-03)."""
        return self.iter()

    def __bool__(self) -> bool:
        return len(self) > 0

    def iter(self):
        """Yield SignedNTuples one at a time without ever materialising
        the full list.  For one-pass streaming consumers."""
        n = len(self)
        # Re-use the same builder logic as _unpack_to_ntuples but
        # yield instead of accumulating.
        from ..core.n_tuples import SignedNTuple
        v = self._v
        sigma = self._sigma
        es = self._edge_signs
        k = self.arity()
        # Group consecutive cycles by arity (they're all the same
        # arity in a single pool, so just iterate flat).
        for i in range(n):
            vs = tuple(int(x) for x in v[i])
            sig = tuple(int(x) for x in sigma[i])
            edge_s = (
                tuple(int(x) for x in es[i])
                if es is not None else None
            )
            bal = 1
            for s in sig:
                bal *= s
            yield SignedNTuple(
                v=vs, sigma=sig,
                edge_signs=edge_s, balanced=bool(bal == 1),
                arity=k,
            )

    def iter_chunks(self, chunk_size: int = 1024):
        """Yield lists of up to `chunk_size` SignedNTuples each, then
        STOP.  Caller's responsibility to consume each chunk before
        the next is yielded; chunks are NOT retained across yields."""
        buf: list = []
        for nt in self.iter():
            buf.append(nt)
            if len(buf) == chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def __repr__(self) -> str:
        return (
            f"LazyCyclePool(n={len(self)}, k={self.arity()}, "
            f"path={self.path.name}, materialised={self._materialised is not None})"
        )


_STATS = CacheStats()


def stats() -> CacheStats:
    return _STATS


def reset_stats() -> None:
    global _STATS
    _STATS = CacheStats()


def _cache_hit(path: pathlib.Path):
    """Try to load a cache file. Returns a ``LazyCyclePool`` on
    success, or None if the file is corrupt (also unlinks it) or
    missing.

    Memory fix 2026-06-03: previously returned ``_unpack_to_ntuples(...)``
    which materialised a SignedNTuple list (~5× the raw numpy size,
    ~40 GB peak for 500K-cycle pools). The LazyCyclePool defers
    materialisation until ``.materialize()`` is called explicitly,
    keeping warm-cache RSS bounded to the on-disk file size.
    """
    if not path.exists():
        return None
    try:
        v, sigma, edge_signs = _load_packed(path)
    except Exception:
        path.unlink(missing_ok=True)
        return None
    _STATS.hits += 1
    _STATS.bytes_loaded += path.stat().st_size
    return LazyCyclePool(path, v, sigma, edge_signs)


def _cache_miss(path: pathlib.Path, t_list):
    """Pack ``t_list`` (releasing its entries as we go), save to disk,
    drop the original list, and return a ``LazyCyclePool`` handle
    holding only the packed numpy arrays.

    Memory fix 2026-06-03 (Komondor/A100 OOM): previously this returned
    ``_unpack_to_ntuples(...)`` — a freshly rebuilt SignedNTuple list,
    which doubles peak RSS for large pools (~500K cycles ⇒ ~40 GB
    Python objects). Returning a ``LazyCyclePool`` instead lets cold-
    cache callers benefit from the same bulk-numpy access path that
    warm-cache callers use via ``lazy_load_construct_k`` — without
    paying the materialisation cost twice.

    Callers expecting an eager list:
      - get a LazyCyclePool now, which supports ``len()`` and
        ``iter()`` (yields SignedNTuples one at a time, no full
        materialisation).
      - can call ``.materialize()`` explicitly if they need the
        list.
    """
    _STATS.misses += 1
    v, sigma, edge_signs = _pack_and_drop(t_list)
    del t_list
    _save_packed(path, v, sigma, edge_signs)
    _STATS.bytes_written += path.stat().st_size
    return LazyCyclePool(path, v, sigma, edge_signs)

