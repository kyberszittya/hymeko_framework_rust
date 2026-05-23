"""Cycle / walk enumeration cache — split out of the 714-LOC monolith
2026-05-11 per CLAUDE.md §6.5 #4. Sub-modules (all ≤ 300 LOC):

  config.py    runtime config + cache-dir + fingerprint helpers
  key.py       graph hash + cache key derivation
  pack.py      SignedNTuple ↔ flat-array serialization
  format.py    on-disk format dispatch (npz / cbor)
  stats.py     CacheStats + LazyCyclePool + hit/miss accounting
  api.py       public `cached_construct_*` + `lazy_load_*` entries
"""
from .config import cache_enabled, _cache_dir
# Private helpers re-exported for tests / introspection.
from .pack import _pack_and_drop, _unpack_to_ntuples
from .format import _save_packed, _load_packed

from .stats import CacheStats, LazyCyclePool, stats, reset_stats
from .api import (
    cached_construct_k, cached_construct_walks,
    cached_construct_2, cached_construct_triads,
    lazy_load_construct_k, lazy_load_construct_walks,
    cache_path_for_k,
)

__all__ = [
    "cache_enabled", "CacheStats", "LazyCyclePool",
    "stats", "reset_stats",
    "cached_construct_k", "cached_construct_walks",
    "cached_construct_2", "cached_construct_triads",
    "lazy_load_construct_k", "lazy_load_construct_walks",
    "cache_path_for_k",
    "_save_packed", "_load_packed",
]
