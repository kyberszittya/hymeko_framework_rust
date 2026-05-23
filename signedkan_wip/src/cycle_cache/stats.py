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

