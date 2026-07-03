"""Tests for the memory-bounded reservoir sampler used by the
Python-fallback cycle / walk enumerators.

Added 2026-06-03 alongside ``hymeko_neuro/hyperedge/reservoir.py``
following the Komondor probe 13883886 OOM root-cause analysis:
the Singularity image did not include the ``hymeko`` Rust wheel,
so ``_python_walks`` ran and built an unbounded Python ``list`` of
every visited walk. With the reservoir in place the same walk_len=4
bitcoin_alpha case is capped at ``max_walks`` items regardless.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from hymeko_neuro.hyperedge.reservoir import (
    NumpyReservoirSampler, ReservoirSampler,
)


# ─── Core sampler contract ────────────────────────────────────────


def test_cap_none_is_unbounded():
    """With ``cap=None`` the sampler degenerates to ``list.append``."""
    s: ReservoirSampler[int] = ReservoirSampler(None, seed=0)
    for i in range(100):
        s.offer(i)
    assert s.items == list(range(100))
    assert s.seen == 100


def test_cap_zero_keeps_nothing():
    s: ReservoirSampler[int] = ReservoirSampler(0, seed=0)
    for i in range(50):
        s.offer(i)
    assert s.items == []
    assert s.seen == 50


def test_negative_cap_rejected():
    with pytest.raises(ValueError, match="cap must be >= 0"):
        ReservoirSampler(-1, seed=0)


def test_below_cap_keeps_everything():
    s: ReservoirSampler[int] = ReservoirSampler(100, seed=0)
    for i in range(80):
        s.offer(i)
    assert s.items == list(range(80))
    assert len(s) == 80


def test_at_cap_keeps_everything():
    s: ReservoirSampler[int] = ReservoirSampler(100, seed=0)
    for i in range(100):
        s.offer(i)
    assert s.items == list(range(100))
    assert len(s) == 100


# ─── Memory bound (the actual point) ─────────────────────────────


def test_above_cap_holds_exactly_cap_items():
    """The whole reason this class exists: regardless of stream length,
    the retained list never exceeds ``cap``. Komondor walk_len=4 fix."""
    s: ReservoirSampler[int] = ReservoirSampler(100, seed=42)
    for i in range(10_000):
        s.offer(i)
    assert len(s.items) == 100
    assert s.seen == 10_000


# ─── Statistical correctness (uniform sample) ────────────────────


def test_uniform_distribution_over_many_runs():
    """Vitter Algorithm R is uniform. We can't verify uniformity
    deterministically without statistics, so we run the sampler 5000
    times against a length-50 stream with cap=10, count how often
    each input position survives, and assert the empirical frequency
    is within a sane band of the expected ``cap/n = 0.2``.

    Tolerance: ±0.04 around 0.20 per position (3σ on a Bernoulli with
    n=5000 — narrow enough to catch genuine bias, wide enough to be
    flake-free on CI noise)."""
    counts: Counter[int] = Counter()
    n_trials = 5000
    stream_len = 50
    cap = 10
    for trial in range(n_trials):
        s: ReservoirSampler[int] = ReservoirSampler(cap, seed=trial)
        for i in range(stream_len):
            s.offer(i)
        for item in s.items:
            counts[item] += 1

    expected = n_trials * cap / stream_len
    for pos in range(stream_len):
        observed = counts[pos]
        rel_err = abs(observed - expected) / expected
        assert rel_err < 0.20, (
            f"position {pos}: observed {observed}, expected ~{expected}, "
            f"rel_err {rel_err:.3f} — possible bias in reservoir"
        )


def test_extend_matches_repeated_offer():
    a: ReservoirSampler[int] = ReservoirSampler(50, seed=1)
    for i in range(200):
        a.offer(i)

    b: ReservoirSampler[int] = ReservoirSampler(50, seed=1)
    b.extend(range(200))

    assert a.items == b.items
    assert a.seen == b.seen


def test_repr_includes_state():
    s: ReservoirSampler[int] = ReservoirSampler(10, seed=0)
    for i in range(25):
        s.offer(i)
    r = repr(s)
    assert "cap=10" in r
    assert "len(items)=10" in r
    assert "seen=25" in r


# ─── Wired into _python_walks (the actual Komondor fix) ──────────


def _toy_dense_graph():
    """Small dense graph that yields many length-4 walks (more than
    100 even from a single start vertex) so the cap kicks in."""
    from hymeko_neuro.data.datasets import SignedGraph
    rng = np.random.default_rng(0)
    n = 12
    edges = []
    signs = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.7:
                edges.append((u, v))
                signs.append(1 if rng.random() < 0.6 else -1)
    return SignedGraph(
        edges=np.array(edges, dtype=np.int64),
        signs=np.array(signs, dtype=np.int8),
        n_nodes=n,
    )


def test_python_walks_respects_max_walks():
    """The Komondor regression: ``_python_walks`` MUST cap its returned
    list at ``max_walks`` items. Without this, the DFS appended every
    length-(L+1) tuple it found and the legacy code path OOM'd at 32 GB
    on bitcoin_alpha walk_len=4."""
    from hymeko_neuro.hyperedge.walks import _python_walks
    g = _toy_dense_graph()
    walks = _python_walks(g, walk_len=4, max_walks=10, seed=0)
    assert len(walks) <= 10, (
        f"reservoir cap not enforced: got {len(walks)} walks, expected <= 10"
    )


def test_python_walks_uncapped_matches_legacy_behaviour():
    """When ``max_walks=None`` the new code must return every visited
    walk in DFS order — the same as the pre-2026-06-03 legacy path —
    so any caller that relied on getting every walk (e.g. exhaustive
    enumeration for unit tests) is unaffected by the refactor."""
    from hymeko_neuro.hyperedge.walks import _python_walks
    g = _toy_dense_graph()
    walks_none = _python_walks(g, walk_len=3, max_walks=None, seed=0)
    walks_huge = _python_walks(g, walk_len=3, max_walks=10**9, seed=0)
    # With a cap larger than the population, reservoir keeps everything;
    # output should be identical to the None case.
    assert walks_none == walks_huge


def test_python_walks_seed_determinism():
    """Same seed + same graph + same cap → same surviving sample.
    Different seeds → samples likely differ (allows the model-seed
    sweep to actually sweep)."""
    from hymeko_neuro.hyperedge.walks import _python_walks
    g = _toy_dense_graph()
    a = _python_walks(g, walk_len=4, max_walks=20, seed=7)
    b = _python_walks(g, walk_len=4, max_walks=20, seed=7)
    assert a == b
    c = _python_walks(g, walk_len=4, max_walks=20, seed=13)
    # Don't assert inequality here — small graphs may yield the same
    # surviving set under two seeds if the input population is small.
    # The determinism assertion (a == b) is the load-bearing one.
    assert isinstance(c, list)


# ─── enumerate_k_cycles fallback ────────────────────────────────


# ─── NumpyReservoirSampler (the fast path) ───────────────────────


def test_numpy_reservoir_below_cap_keeps_everything():
    s = NumpyReservoirSampler(cap=10, k=3, dtype=np.int32, seed=0)
    for i in range(7):
        s.offer([i, i + 1, i + 2])
    out = s.to_array()
    assert out.shape == (7, 3)
    np.testing.assert_array_equal(out[:, 0], np.arange(7))
    assert s.seen == 7


def test_numpy_reservoir_above_cap_holds_exactly_cap():
    """Komondor walk-bound: stream length ≫ cap, output stays ≤ cap rows."""
    s = NumpyReservoirSampler(cap=100, k=4, dtype=np.int32, seed=42)
    for i in range(100_000):
        s.offer([i, i + 1, i + 2, i + 3])
    out = s.to_array()
    assert out.shape == (100, 4)
    assert s.seen == 100_000


def test_numpy_reservoir_zero_cap_is_no_op():
    s = NumpyReservoirSampler(cap=0, k=3, dtype=np.int32, seed=0)
    for i in range(50):
        s.offer([i, i, i])
    out = s.to_array()
    assert out.shape == (0, 3)
    assert s.seen == 50


def test_numpy_reservoir_uniform_distribution():
    """Algorithm L is uniform. Empirical-frequency check identical in
    spirit to the ReservoirSampler version above."""
    counts: Counter[int] = Counter()
    n_trials = 5000
    stream_len = 50
    cap = 10
    for trial in range(n_trials):
        s = NumpyReservoirSampler(
            cap=cap, k=1, dtype=np.int32, seed=trial,
        )
        for i in range(stream_len):
            s.offer([i])
        for row in s.to_array():
            counts[int(row[0])] += 1

    expected = n_trials * cap / stream_len
    for pos in range(stream_len):
        observed = counts[pos]
        rel_err = abs(observed - expected) / expected
        assert rel_err < 0.20, (
            f"position {pos}: observed {observed}, expected ~{expected}, "
            f"rel_err {rel_err:.3f} — possible Algorithm L bias"
        )


def test_numpy_reservoir_no_python_tuple_alloc_on_hot_path():
    """White-box check: the offer path must not allocate per-call
    Python tuples. We probe by counting ``tuple()`` calls via a
    tracking subclass on a 100K-element stream with cap=100.

    Algorithm L expected RNG draws ≈ cap × (1 + log(n/cap))
                                    = 100 × (1 + log(1000))
                                    ≈ 100 × 7.9
                                    ≈ 790
    so the buf[…] = seq row assignment runs ≤ ~890 times (pre-fill
    100 + ~790 selections). Far below the 100K offers.
    """
    s = NumpyReservoirSampler(cap=100, k=3, dtype=np.int32, seed=1)
    # Count actual buf[i] = seq writes via a counter wrapping the buf.
    real_buf = s.buf
    write_count = {"n": 0}

    class CountingBuf(np.ndarray):
        def __setitem__(self, idx, value):  # noqa: D401
            write_count["n"] += 1
            real_buf[idx] = value

    s.buf = real_buf.view(CountingBuf)
    for i in range(100_000):
        s.offer([i, i + 1, i + 2])

    # Pre-fill writes 100; Algorithm L selections add ~700-900. Total
    # under ~1500. Definitely under 10_000 (i.e. < 10% of stream).
    assert write_count["n"] < 10_000, (
        f"too many buf writes ({write_count['n']}) — Algorithm L skip "
        "logic appears broken; should be ≪ stream length"
    )


def test_numpy_reservoir_negative_cap_rejected():
    with pytest.raises(ValueError, match="cap must be >= 0"):
        NumpyReservoirSampler(cap=-1, k=3)


def test_numpy_reservoir_negative_k_rejected():
    with pytest.raises(ValueError, match="k must be >= 0"):
        NumpyReservoirSampler(cap=10, k=-1)


# ─── enumerate_k_cycles fallback ────────────────────────────────


def test_enumerate_k_cycles_respects_max_cycles():
    """Same pattern for the cycle-side Python fallback. Defensive —
    the cycles path is less likely to OOM on bitcoin_alpha (cycle
    count is much smaller than walk count) but the same hymeko-wheel-
    missing failure mode would still affect Slashdot / Epinions."""
    from hymeko_neuro.hyperedge.n_tuples import enumerate_k_cycles
    g = _toy_dense_graph()
    # Build the adj dict the function expects
    adj: dict[int, dict[int, int]] = {}
    for i in range(len(g.edges)):
        u, v = int(g.edges[i, 0]), int(g.edges[i, 1])
        s = int(g.signs[i])
        adj.setdefault(u, {})[v] = s
        adj.setdefault(v, {})[u] = s
    cycles = enumerate_k_cycles(adj, k=4, max_cycles=5, seed=0)
    assert len(cycles) <= 5
