"""Strategy + Adapter pattern for tuple enumeration (2026-06-03).

Replaces the Cartesian-product API surface of four ``cached_construct_*``
functions with one :class:`TupleEnumerator` strategy hierarchy and one
:class:`CachedEnumerator` decorator. Addresses CLAUDE.md §6.5 anti-
patterns #1 (Cartesian product), #3 (per-experiment scaffold dup),
#5 (new-axis-as-new-function-name).

The walks cold-cache vectorisation that fixes the Komondor walk-k=5 OOM
(job 13883876, MaxRSS 31.97 GB on bitcoin_alpha mixed-tuples) is a
natural consequence of putting WalkEnumerator next to CycleEnumerator
under the same Strategy contract — they now share the numpy-direct
return shape, no Python-object materialisation peak in either.

Public surface:

    EnumeratedArrays                — dataclass: v / sigma / edge_signs + is_closed
    TupleEnumerator (ABC)           — enumerate(g, seed) + cache_key_params()
    CycleEnumerator, WalkEnumerator,
    TwoCycleEnumerator, TriadEnumerator   — concrete strategies
    CachedEnumerator                — disk-cache decorator
    cached_construct(g, kind, **kw) — one-line dispatch

Legacy ``cached_construct_{k,walks,2,triads}`` are kept in ``api.py``
as 2-line wrappers over CachedEnumerator + Strategy; the public
signatures and on-disk cache keys are preserved byte-for-byte so
existing cache files remain valid.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import pathlib
from typing import Any, Mapping

import numpy as np


# ─── Result contract ──────────────────────────────────────────────


@dataclass(frozen=True)
class EnumeratedArrays:
    """Packed numpy-array result of every :class:`TupleEnumerator`.

    The on-disk cache format and the in-memory :class:`LazyCyclePool`
    consume these three arrays directly — no Python-object intermediate.
    For 500K-cycle pools this caps peak RSS at the array sum (~30 MB
    for k=5 / cap=100K) vs. ~5× more for the legacy SignedNTuple list
    path (the Komondor OOM root cause for walk k=5 on 2026-06-03).

    Attributes
    ----------
    v
        ``(N, arity)`` int32 vertex indices. For cycles ``arity`` is
        the cycle length and the implied edge set is the modular
        ``(v[i], v[(i+1) % arity])`` chain. For walks ``arity`` is
        ``walk_len + 1`` and the edges are open ``(v[i], v[i+1])``
        for ``i < arity - 1``.
    sigma
        ``(N, arity)`` int8 per-vertex sign (+1 / -1). Davis-style
        parity: ``σ_i = (-1)^(neg-edge count incident to vertex i
        within the tuple)``.
    edge_signs
        ``(N, arity_edges)`` int8 per-edge sign. For cycles
        ``arity_edges == arity``; for walks ``arity_edges == arity-1``.
    is_closed
        True iff the tuple is a cycle (edge set wraps modulo arity).
        Drives ``arity_edges`` and the σ-recurrence at the seam.
    """

    v: np.ndarray
    sigma: np.ndarray
    edge_signs: np.ndarray
    is_closed: bool

    def __post_init__(self) -> None:
        # Defensive shape checks — preconditions / postconditions
        # (CLAUDE.md §8). Cheap, deterministic, fires once per
        # strategy invocation.
        if self.v.ndim != 2 or self.sigma.ndim != 2:
            raise ValueError(
                f"v / sigma must be 2-D; got shapes {self.v.shape} / "
                f"{self.sigma.shape}"
            )
        if self.v.shape != self.sigma.shape:
            raise ValueError(
                f"v and sigma must have identical shape; got "
                f"{self.v.shape} vs {self.sigma.shape}"
            )
        expected_e = self.v.shape[1] if self.is_closed else self.v.shape[1] - 1
        if (self.edge_signs.shape[0] != self.v.shape[0]
                or self.edge_signs.shape[1] != expected_e):
            raise ValueError(
                f"edge_signs shape {self.edge_signs.shape} mismatches "
                f"v {self.v.shape} for is_closed={self.is_closed} "
                f"(expected (N, {expected_e}))"
            )


# ─── Strategy ABC ─────────────────────────────────────────────────


class TupleEnumerator(ABC):
    """How to enumerate tuples for a given graph.

    Each concrete subclass:

    - Takes its enumeration-defining parameters in ``__init__`` (no
      env-var reads in the hot path — CLAUDE.md §6.5 #11);
    - Returns :class:`EnumeratedArrays` directly (no Python-object
      list materialisation);
    - Reports the cache-key parameters via :meth:`cache_key_params`
      so :class:`CachedEnumerator` can produce a stable on-disk key
      byte-identical to the legacy path.

    The split between :meth:`enumerate` (computes the arrays) and
    :meth:`cache_key_params` (identifies the input) lets
    :class:`CachedEnumerator` skip the enumeration entirely on warm
    cache.
    """

    @abstractmethod
    def enumerate(
        self, g, *, seed: int,
    ) -> EnumeratedArrays:
        """Enumerate tuples on ``g``. ``seed`` is the deterministic
        seed for any random subsampling inside the strategy.

        Precondition: ``g`` is a :class:`SignedGraph`.
        Postcondition: returned ``EnumeratedArrays`` is shape-consistent
        per its ``is_closed`` flag (asserted by the dataclass).
        """

    @abstractmethod
    def cache_key_params(self) -> Mapping[str, Any]:
        """Return the keyword args passed to :func:`_cache_key`.

        Required keys: ``kind`` (str), ``k`` (int), ``max_cycles``
        (int | None). Optional: ``use_enum_seed`` (bool, default True),
        ``use_topk_fingerprint`` (bool, default True). These two
        booleans gate whether the env-driven enum seed and topk
        fingerprint participate in the on-disk key; the legacy paths
        for k=2 and pre-topk triads pin them to 0 / empty, so the
        strategies for those tuple kinds must opt out.
        """


# ─── Concrete strategies ─────────────────────────────────────────


class CycleEnumerator(TupleEnumerator):
    """k-cycles via the Rust enumerator.

    All HSIKAN_TOPK_MODE dispatch is handled inside
    :func:`construct_k_arrays` — the strategy just passes the call
    through. The numpy-direct return is the path the post-2026-06-03
    cold-cache uses already.
    """

    def __init__(
        self, k: int, max_cycles: int | None,
        *, directed: bool = False, early_stop: bool = False,
    ):
        if k < 3:
            raise ValueError(f"k must be >= 3 for CycleEnumerator, got {k}")
        self.k = int(k)
        self.max_cycles = max_cycles
        self.directed = bool(directed)
        self.early_stop = bool(early_stop)

    def enumerate(self, g, *, seed: int) -> EnumeratedArrays:
        from hymeko_neuro.hyperedge.n_tuples import construct_k_arrays
        v, sigma, es = construct_k_arrays(
            g, self.k, max_cycles=self.max_cycles, seed=seed,
            directed=self.directed, early_stop=self.early_stop,
        )
        return EnumeratedArrays(
            v=v, sigma=sigma, edge_signs=es, is_closed=True,
        )

    def cache_key_params(self) -> Mapping[str, Any]:
        return {
            "kind": "cycle_k", "k": self.k,
            "max_cycles": self.max_cycles,
            "use_enum_seed": True,
            "use_topk_fingerprint": True,
        }


class WalkEnumerator(TupleEnumerator):
    """L-walks via the Rust enumerator. Returns open-walk arrays.

    Cold-cache path is vectorised: subsamples to ``max_walks`` in numpy
    space *before* any Python tuple/SignedNTuple materialisation. This
    is the fix for the walk k=5 Komondor OOM (job 13883876) — the
    legacy path materialised every Rust-enumerated walk as a Python
    tuple before subsampling, peaking at multi-million tuples on
    bitcoin_alpha walks-of-length-5.
    """

    def __init__(self, walk_len: int, max_walks: int | None):
        if walk_len < 1:
            raise ValueError(
                f"walk_len must be >= 1 for WalkEnumerator, got {walk_len}"
            )
        self.walk_len = int(walk_len)
        self.max_walks = max_walks

    def enumerate(self, g, *, seed: int) -> EnumeratedArrays:
        from hymeko_neuro.hyperedge.walks import construct_walks_arrays
        v, sigma, es = construct_walks_arrays(
            g, self.walk_len, self.max_walks, seed=seed,
        )
        return EnumeratedArrays(
            v=v, sigma=sigma, edge_signs=es, is_closed=False,
        )

    def cache_key_params(self) -> Mapping[str, Any]:
        return {
            "kind": "walk", "k": self.walk_len,
            "max_cycles": self.max_walks,
            "use_enum_seed": True,
            "use_topk_fingerprint": True,
        }


class ABBWalkEnumerator(TupleEnumerator):
    """Top-K walks under a :class:`PathScorer` with Accelerated
    Branch-and-Bound pruning. Friedler P-graph tradition.

    The DFS maintains a min-heap of ``top_k`` ``(score, walk)`` pairs.
    At each extension the scorer's admissible upper bound is computed
    on the current prefix; if the bound falls below the worst-in-heap
    score, the entire subtree is pruned. Composes the scorer adapter
    so the same DFS engine serves every score family (balance,
    fraction_negative, entropy, ...) without code duplication
    (CLAUDE.md §6.5 #1 / #9).

    Cache-key parity with :class:`WalkEnumerator`: writes to the same
    ``kind="walk"`` namespace but pinning ``max_cycles`` to
    ``-(top_k)`` so ABB results never collide with the MSG path's
    on-disk cache file.
    """

    def __init__(
        self, walk_len: int, top_k: int,
        scorer_name: str = "balance",
    ):
        if walk_len < 1:
            raise ValueError(
                f"walk_len must be >= 1, got {walk_len}"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.walk_len = int(walk_len)
        self.top_k = int(top_k)
        self.scorer_name = str(scorer_name)

    def enumerate(self, g, *, seed: int) -> EnumeratedArrays:
        from hymeko_neuro.hyperedge.abb_walks import abb_enumerate_walks
        from hymeko_neuro.hyperedge.path_scorers import pick_scorer
        scorer = pick_scorer(self.scorer_name)
        eu = np.ascontiguousarray(g.edges[:, 0], dtype=np.int64)
        ev = np.ascontiguousarray(g.edges[:, 1], dtype=np.int64)
        es = np.ascontiguousarray(g.signs, dtype=np.int8)
        v, edge_signs, _stats = abb_enumerate_walks(
            eu, ev, es, int(g.n_nodes),
            walk_len=self.walk_len,
            top_k=self.top_k,
            scorer=scorer,
            seed=seed,
        )
        # Vectorised σ from edge_signs (mirrors _classify_walks_arrays).
        sigma = _walks_sigma_from_edge_signs(edge_signs, self.walk_len)
        return EnumeratedArrays(
            v=v.astype(np.int32),
            sigma=sigma,
            edge_signs=edge_signs,
            is_closed=False,
        )

    def cache_key_params(self):
        # Negate top_k so the on-disk file never collides with the
        # MSG (max_walks=positive) cache; same kind="walk" namespace
        # keeps the cache directory simple.
        return {
            "kind": "walk", "k": self.walk_len,
            "max_cycles": -self.top_k,
            "use_enum_seed": True,
            "use_topk_fingerprint": True,
        }


class SSGWalkEnumerator(TupleEnumerator):
    """Subset Structure Generation — Pareto-filter over
    (primary_score, secondary_score) axes after an inner ABB / MSG
    enumeration. Multi-objective extension of :class:`ABBWalkEnumerator`.

    With a single secondary axis (e.g. entropy on top of balance), SSG
    returns the walks on the Pareto frontier of both — never a walk
    that another dominates on both axes simultaneously.

    Composition: instantiate an inner :class:`ABBWalkEnumerator` for
    the primary axis, then post-filter against the secondary scorer's
    per-walk score. Both scorers must be :class:`PathScorer` instances
    so the same code path drives any pair.
    """

    def __init__(
        self, walk_len: int, top_k: int,
        primary_scorer: str = "balance",
        secondary_scorer: str = "entropy",
    ):
        if walk_len < 1:
            raise ValueError(f"walk_len must be >= 1, got {walk_len}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.walk_len = int(walk_len)
        self.top_k = int(top_k)
        self.primary_scorer = str(primary_scorer)
        self.secondary_scorer = str(secondary_scorer)

    def enumerate(self, g, *, seed: int) -> EnumeratedArrays:
        from hymeko_neuro.hyperedge.abb_walks import ssg_pareto_filter
        from hymeko_neuro.hyperedge.path_scorers import pick_scorer
        # First emit a top-K pool via ABB on the primary scorer.
        inner = ABBWalkEnumerator(
            self.walk_len, self.top_k,
            scorer_name=self.primary_scorer,
        )
        primary_arr = inner.enumerate(g, seed=seed)
        if primary_arr.v.shape[0] == 0:
            return primary_arr
        # Score the surviving pool on the secondary axis.
        scorer_p = pick_scorer(self.primary_scorer)
        scorer_s = pick_scorer(self.secondary_scorer)
        primary_scores = np.array([
            scorer_p.score(primary_arr.v[i], primary_arr.edge_signs[i])
            for i in range(primary_arr.v.shape[0])
        ], dtype=np.float64)
        secondary_scores = np.array([
            scorer_s.score(primary_arr.v[i], primary_arr.edge_signs[i])
            for i in range(primary_arr.v.shape[0])
        ], dtype=np.float64)
        v, edge_signs, _mask = ssg_pareto_filter(
            primary_arr.v, primary_arr.edge_signs,
            [primary_scores, secondary_scores],
        )
        sigma = _walks_sigma_from_edge_signs(edge_signs, self.walk_len)
        return EnumeratedArrays(
            v=v, sigma=sigma, edge_signs=edge_signs, is_closed=False,
        )

    def cache_key_params(self):
        # Distinct cache namespace so SSG doesn't collide with ABB / MSG.
        return {
            "kind": "walk_ssg", "k": self.walk_len,
            "max_cycles": -self.top_k,
            "use_enum_seed": True,
            "use_topk_fingerprint": True,
        }


# MSG is the existing :class:`WalkEnumerator` — alias kept here so the
# Friedler MSG / ABB / SSG triad reads uniformly at call sites.
MSGWalkEnumerator = WalkEnumerator


def _walks_sigma_from_edge_signs(
    edge_signs: np.ndarray, walk_len: int,
) -> np.ndarray:
    """Vertex σ from per-edge signs (open walk, no modular wrap).

    Vertex 0 sees edge 0 only; vertex L sees edge L-1 only; interior
    vertices see edges i-1 and i. σ_i = +1 iff incident negative
    count is even. Mirrors the recurrence in
    :func:`hymeko_neuro.hyperedge.walks._classify_walks_arrays`.
    """
    if edge_signs.size == 0:
        return np.zeros(
            (edge_signs.shape[0], walk_len + 1), dtype=np.int8,
        )
    N = edge_signs.shape[0]
    L = walk_len
    is_neg = (edge_signs == -1).astype(np.int8)
    neg_counts = np.zeros((N, L + 1), dtype=np.int8)
    neg_counts[:, 0] = is_neg[:, 0]
    neg_counts[:, L] = is_neg[:, L - 1]
    if L > 1:
        neg_counts[:, 1:L] = is_neg[:, :L - 1] + is_neg[:, 1:L]
    return np.where((neg_counts % 2) == 0, 1, -1).astype(np.int8)


class TwoCycleEnumerator(TupleEnumerator):
    """k=2 cycles ≡ the edges themselves.

    No randomness, no top-K family, deterministic from ``g.edges`` /
    ``g.signs``. Cache key matches the legacy ``cached_construct_2``
    which pinned ``enum_seed=0`` and no topk fingerprint.
    """

    def enumerate(self, g, *, seed: int) -> EnumeratedArrays:
        from hymeko_neuro.hyperedge.n_tuples import construct_2_arrays
        v, sigma, es = construct_2_arrays(g)
        return EnumeratedArrays(
            v=v, sigma=sigma, edge_signs=es, is_closed=True,
        )

    def cache_key_params(self) -> Mapping[str, Any]:
        return {
            "kind": "cycle_k", "k": 2, "max_cycles": None,
            "use_enum_seed": False,
            "use_topk_fingerprint": False,
        }


class TriadEnumerator(TupleEnumerator):
    """k=3 triads. Dispatches per ``HSIKAN_TOPK_MODE``:

    - In ``global`` / ``per_vertex`` (etc.) mode → defers to
      :class:`CycleEnumerator` with ``k=3``, which uses the rayon-
      parallel Rust path with top-K pruning.
    - Otherwise → uses the classic ``hyperedges.construct(g)`` Python
      path, packed to arrays via ``_pack_and_drop``.

    Cache key matches the legacy ``cached_construct_triads``: pre-topk
    triad path uses kind=``triads``, pinned seed=0, no topk fingerprint.
    Topk-mode path delegates to CycleEnumerator's key (different kind),
    so on-disk cache files for the two regimes never collide.
    """

    def _is_topk_mode(self) -> bool:
        from hymeko_neuro.runtime.runtime_config import get_runtime
        return get_runtime().topk.mode in (
            "global", "global_bb", "entropy",
            "per_vertex", "per_vertex_adaptive", "per_vertex_tiered",
        )

    def enumerate(self, g, *, seed: int) -> EnumeratedArrays:
        if self._is_topk_mode():
            return CycleEnumerator(k=3, max_cycles=None).enumerate(
                g, seed=seed,
            )
        from hymeko_neuro.hyperedge.hyperedges import construct as _construct_triads
        from .pack import _pack_and_drop
        t_list = _construct_triads(g)
        v, sigma, es = _pack_and_drop(t_list)
        del t_list
        return EnumeratedArrays(
            v=v, sigma=sigma, edge_signs=es, is_closed=True,
        )

    def cache_key_params(self) -> Mapping[str, Any]:
        # When in topk mode, delegate to CycleEnumerator(k=3, None)'s key
        # so cached_construct_triads and cached_construct_k(k=3) share
        # the on-disk cache file (matches legacy behaviour).
        if self._is_topk_mode():
            return CycleEnumerator(k=3, max_cycles=None).cache_key_params()
        return {
            "kind": "triads", "k": 3, "max_cycles": None,
            "use_enum_seed": False,
            "use_topk_fingerprint": False,
        }


# ─── Cache decorator ─────────────────────────────────────────────


class CachedEnumerator:
    """Adapter wrapping a :class:`TupleEnumerator` with disk-cache +
    :class:`LazyCyclePool`.

    On a cache hit, returns the LazyCyclePool without invoking the
    strategy at all — warm-cache latency is dominated by the .npz
    read (~10s of ms for 500K-cycle pools).

    On a cache miss, invokes the strategy, packs the result to disk,
    returns a LazyCyclePool over the freshly-built arrays. The arrays
    do NOT round-trip through disk on the cold call — they're handed
    directly to the LazyCyclePool, so cold-cache latency adds one
    ``np.savez`` to the bare enumeration cost.

    When ``cache_enabled()`` is False, both paths skip the disk layer
    entirely: the strategy is invoked and an ephemeral LazyCyclePool
    backed by a sentinel path is returned.
    """

    # Sentinel for the no-cache branch — never written to / read from.
    _NULL_PATH = pathlib.Path("/dev/null")

    def __init__(
        self,
        inner: TupleEnumerator,
        model_seed: int = 0,
    ):
        self.inner = inner
        self.model_seed = int(model_seed)

    def __call__(self, g):
        from .config import cache_enabled, _enum_seed, _cache_dir, _topk_fingerprint
        from .key import _hash_graph, _cache_key
        from .format import _save_packed
        from .stats import LazyCyclePool, _STATS

        params = self.inner.cache_key_params()
        if not cache_enabled():
            # No cache: invoke strategy, return an ephemeral pool.
            arr = self.inner.enumerate(g, seed=self.model_seed)
            return LazyCyclePool(
                self._NULL_PATH, arr.v, arr.sigma, arr.edge_signs,
            )

        # Compose the on-disk cache key. The two booleans
        # (use_enum_seed / use_topk_fingerprint) gate whether the
        # env-driven seed and topk fingerprint participate in the
        # key — pinned False for k=2 / pre-topk triads to match the
        # legacy ``cached_construct_2`` / ``cached_construct_triads``
        # cache-key shapes byte-for-byte.
        use_enum_seed = bool(params.get("use_enum_seed", True))
        use_topk = bool(params.get("use_topk_fingerprint", True))
        enum_seed_val = _enum_seed() if use_enum_seed else 0
        extra = _topk_fingerprint() if use_topk else None
        graph_hash = _hash_graph(g)
        key = _cache_key(
            graph_hash, params["kind"], params["k"],
            params["max_cycles"], enum_seed_val,
            extra=extra,
        )
        path = _cache_dir() / f"{key}.npz"

        hit = LazyCyclePool.from_path(path)
        if hit is not None:
            return hit

        # Cold cache: enumerate, persist, wrap.
        arr = self.inner.enumerate(g, seed=enum_seed_val)
        _STATS.misses += 1
        _save_packed(path, arr.v, arr.sigma, arr.edge_signs)
        try:
            _STATS.bytes_written += path.stat().st_size
        except FileNotFoundError:
            # _save_packed may have written .cbor under HYMEKO_CACHE_FORMAT=cbor;
            # the bytes-written stat is informational, swallow the miss.
            pass
        return LazyCyclePool(
            path, arr.v, arr.sigma, arr.edge_signs,
        )


# ─── Dispatcher ───────────────────────────────────────────────────


_STRATEGY_FACTORIES: dict[str, Any] = {
    "cycle":   CycleEnumerator,
    "walk":    WalkEnumerator,
    "k2":      TwoCycleEnumerator,
    "triads":  TriadEnumerator,
    # MSG / ABB / SSG family — Friedler P-graph tradition for top-K
    # acceleration. See :mod:`hymeko_neuro.hyperedge.path_scorers` and
    # :mod:`hymeko_neuro.hyperedge.abb_walks`.
    "msg_walk": WalkEnumerator,           # alias
    "abb_walk": ABBWalkEnumerator,
    "ssg_walk": SSGWalkEnumerator,
}


def cached_construct(g, kind: str, *, model_seed: int = 0, **strategy_kwargs):
    """One-line dispatcher over the four strategy kinds.

    Parameters
    ----------
    g
        :class:`SignedGraph`.
    kind
        Strategy name: ``"cycle"`` | ``"walk"`` | ``"k2"`` | ``"triads"``.
    model_seed
        Seed forwarded to the strategy when cache is disabled. With
        cache enabled, the env-driven enum seed wins.
    **strategy_kwargs
        Strategy-specific constructor args. ``cycle`` needs ``k`` +
        ``max_cycles`` (+ optional ``directed`` / ``early_stop``);
        ``walk`` needs ``walk_len`` + ``max_walks``; ``k2`` and
        ``triads`` take no args.

    Returns
    -------
    :class:`LazyCyclePool` — same lazy handle the legacy
    ``cached_construct_*`` functions return.
    """
    if kind not in _STRATEGY_FACTORIES:
        raise ValueError(
            f"unknown tuple kind {kind!r}; valid: "
            f"{sorted(_STRATEGY_FACTORIES)}"
        )
    strategy = _STRATEGY_FACTORIES[kind](**strategy_kwargs)
    return CachedEnumerator(strategy, model_seed=model_seed)(g)
