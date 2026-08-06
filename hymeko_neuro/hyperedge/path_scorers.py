"""Path scorers for top-K walk / cycle enumeration with Accelerated
Branch and Bound (ABB) pruning, in the Friedler P-graph tradition.

Every score function comes with an admissible upper-bound function so
the DFS enumerator can prune subtrees that cannot possibly contribute
to the top-K. The two-function contract:

    score(closed_path) ≤ upper_bound(prefix_state, steps_remaining)

Admissibility postcondition: a violation silently produces a wrong
top-K result — the integration tests
``test_path_scorers_admissibility`` enforce it on random fixtures.

Mirrors the Rust ``hymeko_graph::topk_cycles::BoundedScorer`` trait so
that the Python ABB fallback (when the hymeko wheel is unavailable,
e.g. Komondor Singularity container 2026-06-03) produces semantically
identical top-K results. Python is slower but the answers match.

Three concrete scorers covering the HSIKAN regimes:

- :class:`FractionNegativeScorer` — fraction of negative edges in the
  walk. Tight UB: every remaining edge could be negative.
- :class:`BalanceScorer` — Cartwright-Harary sign product. Trivial UB
  of +1 (parity can flip via the next edge).
- :class:`ShannonEntropyScorer` — per-vertex frequency entropy across
  the walk. UB = ``log(walk_len + 1)`` (max when all vertices distinct).

Designed to compose under the Strategy + Adapter pattern in
:mod:`hymeko_neuro.graph.cycle_cache.strategies`:

    enum = ABBWalkEnumerator(
        walk_len=4, top_k=100,
        scorer=BalanceScorer(),
    )

Refactoring 2026-06-03: extracted from the user's "MSG / ABB / SSG"
top-K acceleration ask. The architecture parallel to Friedler P-graph
search:

- **MSG** (Maximum Structure Generation) — enumerate every feasible
  walk. The vanilla :class:`WalkEnumerator` is MSG with reservoir
  sampling for memory bound.
- **ABB** (Accelerated Branch and Bound) — DFS with upper-bound
  pruning by a :class:`PathScorer`. Returns top-K walks by score.
- **SSG** (Subset Structure Generation) — Pareto-filter over
  (score, cost) axes. Multi-objective extension of ABB; degenerates
  to ABB when there is a single objective.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Sequence


class PathScorer(ABC):
    """A score function paired with an admissible upper-bound for ABB.

    Admissibility contract:

        for every closed path ``p`` extending the prefix ``q``:
        score(p) <= upper_bound(<prefix state of q>, steps_remaining)

    where the prefix state captures everything the bound function needs
    to know about ``q`` — typically the negative-edge count so far.

    Implementations MUST be deterministic (no internal state) so that
    the same prefix produces the same bound across DFS visits.
    """

    @abstractmethod
    def score(
        self, vs: Sequence[int], signs: Sequence[int],
    ) -> float:
        """Score a CLOSED path with the given vertex sequence and
        edge-sign sequence.

        Preconditions:
            - ``len(signs) == arity_edges`` (i.e. ``walk_len`` for an
              open walk, ``arity`` for a closed cycle).
            - Each entry of ``signs`` is in ``{-1, +1}``.
            - ``vs`` is the walk's vertex sequence (length
              ``arity_edges + 1`` for walks, ``arity`` for cycles).

        Postcondition:
            - Returns a finite ``float`` (NaN / Inf is a bug).
        """

    @abstractmethod
    def upper_bound(
        self, n_neg_so_far: int, steps_remaining: int, k_len: int,
    ) -> float:
        """Maximum possible score given the prefix state.

        Parameters
        ----------
        n_neg_so_far
            Number of negative edges in the prefix walked so far.
        steps_remaining
            Number of edges yet to be walked.
        k_len
            Total edge length of the completed walk (= arity_edges).

        Returns
        -------
        Upper bound for any completion of the current prefix; the ABB
        DFS prunes branches whose bound falls below the current top-K
        threshold.

        Postcondition (admissibility):
            For every concrete completion that the DFS could produce,
            ``score(...) <= upper_bound(...)``.
        """

    @abstractmethod
    def name(self) -> str:
        """Stable identifier for cache keys and reports."""


class FractionNegativeScorer(PathScorer):
    """Score = fraction of negative edges in the walk.

    Tight admissible UB: every remaining edge could in principle be
    negative, so ``UB = (n_neg_so_far + steps_remaining) / k_len``.
    Used as the default ABB scorer for the
    ``HSIKAN_TOPK_MODE=global`` cycle path (see
    ``reports/2026-05-10-abb-global-topk.md``).
    """

    def score(self, vs, signs) -> float:
        if not len(signs):
            return 0.0
        n_neg = sum(1 for s in signs if s < 0)
        return n_neg / len(signs)

    def upper_bound(
        self, n_neg_so_far: int, steps_remaining: int, k_len: int,
    ) -> float:
        if k_len <= 0:
            return 0.0
        return (n_neg_so_far + steps_remaining) / k_len

    def name(self) -> str:
        return "fraction_negative"


class BalanceScorer(PathScorer):
    """Cartwright–Harary balance: signed product of edge signs.

    Trivial admissible UB of +1: the sign product is always in
    ``{-1, +1}`` and the next edge can flip parity either way, so
    every prefix can in principle complete to a +1 product.
    """

    def score(self, vs, signs) -> float:
        if not len(signs):
            return 1.0
        out = 1
        for s in signs:
            out *= s
        return float(out)

    def upper_bound(
        self, n_neg_so_far: int, steps_remaining: int, k_len: int,
    ) -> float:
        return 1.0

    def name(self) -> str:
        return "balance"


class SignProductAbsScorer(PathScorer):
    """``|product of signs|``: 1.0 on every signed walk (signs ∈ {±1}).

    Trivial UB = 1.0. Kept for parity with the Rust trio so callers
    can switch implementations without changing call sites.
    """

    def score(self, vs, signs) -> float:
        return 1.0 if len(signs) > 0 else 0.0

    def upper_bound(
        self, n_neg_so_far: int, steps_remaining: int, k_len: int,
    ) -> float:
        return 1.0

    def name(self) -> str:
        return "sign_product_abs"


class ShannonEntropyScorer(PathScorer):
    """Vertex-frequency Shannon entropy of the walk.

    For a walk with ``n_v`` distinct vertices and frequency
    distribution ``p_v = count_v / total``, the score is
    ``-sum(p_v · log p_v)``. Higher = better vertex coverage.

    Admissible UB: ``log(k_len + 1)`` because the maximum entropy
    over a walk of length ``k_len`` (with ``k_len + 1`` vertices) is
    achieved when every vertex is distinct — uniform distribution
    over ``k_len + 1`` outcomes.
    """

    def score(self, vs, signs) -> float:
        # ``vs`` may be a list / tuple / numpy 1-D array; ``len`` is
        # the portable empty-check (``not vs`` raises on numpy arrays
        # of length > 1, "truth value is ambiguous").
        if len(vs) == 0:
            return 0.0
        counts = Counter(int(v) for v in vs)
        total = sum(counts.values())
        if total == 0:
            return 0.0
        return -sum(
            (c / total) * math.log(c / total)
            for c in counts.values()
        )

    def upper_bound(
        self, n_neg_so_far: int, steps_remaining: int, k_len: int,
    ) -> float:
        # k_len walk-edges → k_len + 1 vertices when distinct.
        # Max entropy over k_len + 1 outcomes is log(k_len + 1).
        if k_len <= 0:
            return 0.0
        return math.log(k_len + 1)

    def name(self) -> str:
        return "entropy"


# ─── Registry for string-keyed dispatch ──────────────────────────


_SCORER_FACTORIES: dict[str, type[PathScorer]] = {
    "fraction_negative": FractionNegativeScorer,
    "balance":           BalanceScorer,
    "sign_product_abs":  SignProductAbsScorer,
    "entropy":           ShannonEntropyScorer,
}


def pick_scorer(name: str) -> PathScorer:
    """String-to-scorer dispatcher; mirrors
    :func:`hymeko_py::cycles::io::pick_scorer` for the Python fallback
    path. Raises ``ValueError`` for unknown names so a misspelled env
    var fails at strategy construction rather than silently using the
    wrong scorer in the inner DFS (CLAUDE.md §6.5 #7).
    """
    if name not in _SCORER_FACTORIES:
        raise ValueError(
            f"unknown PathScorer name {name!r}; valid: "
            f"{sorted(_SCORER_FACTORIES)}"
        )
    return _SCORER_FACTORIES[name]()
