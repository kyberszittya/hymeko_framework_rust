"""Auto-split from cycle_cache.py 2026-05-11 (CLAUDE.md §6.5 #4)."""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np



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


