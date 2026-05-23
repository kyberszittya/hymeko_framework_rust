"""Auto-split from mixed_arity_signedkan.py 2026-05-11 (CLAUDE.md §6.5 #4).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def subsample_tuples(tuples, max_count: int, seed: int):
    """Deterministic random subsample. Returns the same list when
    max_count >= len(tuples)."""
    if len(tuples) <= max_count:
        return list(tuples)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(tuples), size=max_count, replace=False)
    return [tuples[int(i)] for i in idx]


def build_edge_to_tuples(tuples,
                          directed: bool = False) -> dict[tuple[int, int], list[int]]:
    """For each edge appearing as a cycle edge of some n-tuple, list
    the tuple indices it belongs to.

    ``directed=False`` (default): keys are unordered ``(min, max)``
    pairs. Each cycle edge contributes one key per cycle position.
    ``directed=True``: keys are directional ``(u, v)`` in cycle order
    — query edge ``(src, dst)`` only matches a cycle if that exact
    direction appears as one of the cycle's directed edges.
    """
    out: dict[tuple[int, int], list[int]] = {}
    for ti, t in enumerate(tuples):
        v = t.v
        k = len(v)
        if k == 2:
            # k=2 hyperedge IS a single edge — record it once.
            u, w = int(v[0]), int(v[1])
            key = (u, w) if directed else (min(u, w), max(u, w))
            out.setdefault(key, []).append(ti)
            continue
        for i in range(k):
            u, w = int(v[i]), int(v[(i + 1) % k])
            key = (u, w) if directed else (min(u, w), max(u, w))
            out.setdefault(key, []).append(ti)
    return out


def build_vertex_to_tuples(tuples) -> dict[int, list[int]]:
    """For each vertex, list of tuple indices having that vertex as
    one of its endpoints. Used for k=2 line-graph-style incidence."""
    out: dict[int, list[int]] = {}
    for ti, t in enumerate(tuples):
        for vid in t.v:
            out.setdefault(int(vid), []).append(ti)
    return out
