"""ARCHITECTURAL_ASSIMILATION_V1 — task-independent tests for the multimodal proposal + policy search.

Proves the O2-motivated property on a synthetic (non-coin) objective: when the winning options live in TWO distant modes,
a multimodal proposal + `MultimodalBudgetSearch` recovers the global-best mode, while a single-centre `FixedBudgetSearch`
anchored at the wrong mode stays local and misses it. Also covers the budget allocator and the single-mode adapter.
"""
import numpy as np

from hymeko_rl.option_rl import (
    FixedBudgetSearch,
    MultimodalBudgetSearch,
    ProposalMode,
    SingleModeProposal,
    allocate_budget,
)


class GaussGen:
    """Local candidate generator: Gaussian jitter around a centre."""

    def __init__(self, std=0.08):
        self.std = std

    def sample(self, center, n, rng):
        return np.asarray(center, np.float64) + rng.normal(0, self.std, (n, len(center)))


class BimodalScorer:
    """Two distant reward peaks (at −1 and +1); the +1 peak is the slightly-higher GLOBAL best. A search anchored at −1
    cannot reach +1 by local jitter — exactly the O2 multimodal structure in miniature."""

    def score(self, cand, rng):
        x = float(np.asarray(cand)[0])                          # wide peaks (0.5) so the mode WEIGHT decides, not sampling noise
        s = max(np.exp(-((x - 1.0) ** 2) / 0.5) * 1.00, np.exp(-((x + 1.0) ** 2) / 0.5) * 0.90)
        return float(s), {"k6": int(s > 0.5)}


class TwoModeProposal:
    def modes(self, obs):
        return [ProposalMode(0.5, np.array([-1.0]), 0.08, 0), ProposalMode(0.5, np.array([1.0]), 0.08, 1)]


class WrongCenter:
    """A single deterministic centre at the WORSE (−1) mode — the shape-blind-averaged-guess stand-in."""

    def center(self, obs):
        return np.array([-1.0])


def test_multimodal_recovers_global_best_mode():
    rng = np.random.default_rng(0)
    prov = MultimodalBudgetSearch(GaussGen(), BimodalScorer(), budget=8).select(TwoModeProposal(), np.zeros(1), rng)
    assert prov.selected_mode == 1                       # the +1 (global-best) mode won
    assert prov.score > 0.95                             # near the top of the +1 peak
    assert abs(float(prov.selected[0]) - 1.0) < 0.2      # the selected action is at the +1 mode
    assert prov.as_dict()["n_modes"] == 2 and prov.as_dict()["k6"] == 1


def test_single_center_search_misses_the_other_mode():
    rng = np.random.default_rng(0)
    prov = FixedBudgetSearch(GaussGen(), BimodalScorer(), budget=8).select(np.array([-1.0]), rng)
    assert float(prov.selected[0]) < 0.0                 # stuck near −1
    assert prov.score < 0.95                             # never reaches the higher +1 peak — the failure the mixture fixes


def test_single_mode_adapter_is_k1():
    modes = SingleModeProposal(WrongCenter()).modes(np.zeros(1))
    assert len(modes) == 1 and modes[0].prob == 1.0 and float(modes[0].center[0]) == -1.0


class PeakScorer:
    """Deterministic (no mutable state); a single reward peak at ``target`` — one mode is the clear winner."""

    def __init__(self, target=2.0):
        self.target = target

    def score(self, cand, rng):
        return -float(abs(float(np.asarray(cand)[0]) - self.target)), {"k6": 0}


class TieScorer:
    """Every candidate scores EXACTLY the same — forces the canonical tie-break to decide the winner."""

    def score(self, cand, rng):
        return 1.0, {"k6": 0}


class OrderProposal:
    """Four modes with distinct centres + ids, emitted in a caller-chosen ORDER (to test order-invariance)."""

    def __init__(self, order):
        self._m = [ProposalMode(0.25, np.array([float(i)]), 0.05, i) for i in range(4)]
        self.order = order

    def modes(self, obs):
        return [self._m[i] for i in self.order]


def _run(order, scorer):
    gen = GaussGen(0.05)
    return MultimodalBudgetSearch(gen, scorer, budget=12).select(OrderProposal(order), np.zeros(1), np.random.default_rng(3))


def test_search_mode_order_invariant_winner_theta_score():
    a = _run([0, 1, 2, 3], PeakScorer(2.0))
    for order in ([3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]):     # reverse + two permutations
        b = _run(order, PeakScorer(2.0))
        assert np.allclose(a.selected, b.selected)               # identical winning θ
        assert abs(a.score - b.score) < 1e-12                    # identical score
        assert np.allclose(a.mode_centers[a.selected_mode], b.mode_centers[b.selected_mode])  # same winning MODE (by centre)
        assert a.outcome == b.outcome                            # same certificate/outcome


def test_tie_break_is_stable_and_order_invariant():
    a = _run([0, 1, 2, 3], TieScorer())
    for order in ([3, 2, 1, 0], [2, 0, 3, 1]):                   # all-ties: the canonical key must pick the SAME candidate
        b = _run(order, TieScorer())
        assert np.allclose(a.selected, b.selected) and a.score == b.score


def test_allocate_budget_ge1_each_and_sums():
    assert allocate_budget([0.5, 0.5], 8) == [4, 4]      # even split, ≥1 each
    a = allocate_budget([0.9, 0.1], 10)
    assert sum(a) == 10 and min(a) >= 1 and a[0] > a[1]  # ∝ prob, but no mode starved
    assert allocate_budget([0.5, 0.5, 0.5], 2) == [1, 1, 0] or sum(allocate_budget([0.5, 0.5, 0.5], 2)) == 2  # budget<K funds top modes
    assert allocate_budget([1.0, 1.0], 0) == [0, 0]
    assert allocate_budget([], 5) == []
