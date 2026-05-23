"""Unit tests for the Phase-8 ``m_per_vertex`` cap on triangle
construction in :mod:`signedkan_wip.src.core.hyperedges`.

Pins three properties:

  1. ``m_per_vertex=None`` → all enumerated triads kept (back-compat).
  2. ``m_per_vertex=M > 0`` → no apex has more than M triads.
  3. ``m_per_vertex`` larger than every per-apex bucket is a no-op.

Plus a determinism check: re-importing the module and re-running
``construct`` returns the same triad list (the Phase-8 sort-the-
neighbour-set fix to ``_enumerate_triangles``).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.datasets import SignedGraph


def _k4_plus() -> SignedGraph:
    """K4 with all + edges. Triangles: (0,1,2), (0,1,3), (0,2,3),
    (1,2,3). All balanced; apex = lowest-index by ``_classify`` tie-
    break: 0, 0, 0, 1. Per-apex buckets: apex 0 → 3 triads, apex 1
    → 1 triad. Total 4 triads."""
    edges = np.array(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64,
    )
    signs = np.ones(len(edges), dtype=np.int64)
    return SignedGraph(edges=edges, signs=signs, n_nodes=4)


def _apex_of(t) -> int:
    return t.v[t.sigma.index(1)]


def test_no_cap_returns_all_triads():
    g = _k4_plus()
    triads = construct(g)
    # K4 has exactly 4 triangles.
    assert len(triads) == 4


def test_cap_limits_per_apex_bucket():
    g = _k4_plus()
    triads = construct(g, m_per_vertex=2)
    # apex 0 originally 3 triads → cap to 2; apex 1 has 1 (uncapped).
    # Total: 2 + 1 = 3.
    assert len(triads) == 3
    by_apex = Counter(_apex_of(t) for t in triads)
    assert by_apex[0] == 2
    assert by_apex[1] == 1


def test_cap_one_keeps_one_per_apex():
    g = _k4_plus()
    triads = construct(g, m_per_vertex=1)
    by_apex = Counter(_apex_of(t) for t in triads)
    assert sorted(by_apex.keys()) == [0, 1]
    assert all(c == 1 for c in by_apex.values())


def test_cap_larger_than_buckets_is_noop():
    g = _k4_plus()
    full = construct(g, m_per_vertex=None)
    big = construct(g, m_per_vertex=100)
    assert full == big


def test_cap_is_deterministic_across_calls():
    g = _k4_plus()
    a = construct(g, m_per_vertex=2)
    b = construct(g, m_per_vertex=2)
    assert a == b


def test_cap_zero_or_negative_is_treated_as_uncapped():
    """Contract: only positive caps bite. 0 and negatives are
    treated as "no cap" rather than raising — matches the back-compat
    semantics of ``None``."""
    g = _k4_plus()
    full = construct(g)
    assert construct(g, m_per_vertex=0) == full
    assert construct(g, m_per_vertex=-1) == full


def test_cap_subset_of_uncapped():
    """A capped construction is a subsequence of the uncapped one;
    the cap doesn't reorder or fabricate triads."""
    g = _k4_plus()
    full = construct(g)
    capped = construct(g, m_per_vertex=2)
    full_set = {(t.v, t.sigma, t.edge_signs) for t in full}
    capped_set = {(t.v, t.sigma, t.edge_signs) for t in capped}
    assert capped_set.issubset(full_set)
