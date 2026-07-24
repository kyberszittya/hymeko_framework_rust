"""EQUAL_BUDGET_KMODE_ABLATION_V1 — unit tests for the coin↔runtime adapter pieces (pure; no MuJoCo/pi_0).

The physical rollout (CoinCarryScorer/run_arm/main) is covered by the smoke integration run; here we lock the pure
contracts: the lexicographic-score adapter, the structured jitter generator's bounds+determinism, and the K-mode
template proposal (K=1 ≡ classifier-argmax template; K>1 ≡ top-K by prob, keyed by template index).
"""
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "experiments/2026_07_22_coin_v3_learning/rl_entry")
from coin_kmode_budget_ablation import (  # noqa: E402
    CoinJitterGenerator, LexScore, TemplateKModeProposal)
from hymeko_rl.coin_delivery.coin_carry_structured import A_BOUND, T_MAX, T_MIN  # noqa: E402


# ── LexScore: lexicographic order + beats the -inf search seed ──
def test_lexscore_k6_dominates():
    assert LexScore((1, 0, 0)) > LexScore((0, 1, 999))          # k6 first, even against high dwell
    assert LexScore((1, 1, 5)) > LexScore((1, 1, 4))            # then dwell breaks the tie
    assert not (LexScore((1, 1, 5)) > LexScore((1, 1, 5)))      # equal ⇒ not greater
    assert LexScore((1, 1, 5)) == LexScore((1, 1, 5))


def test_lexscore_beats_neg_inf_sentinel():
    assert LexScore((0, 0, 0)) > float("-inf")                  # any real outcome beats the search's -inf seed
    assert not (LexScore((0, 0, 0)) == float("-inf"))
    assert float(LexScore((1, 1, 3))) > float(LexScore((0, 1, 300)))   # summary is k6-monotone for logging


# ── CoinJitterGenerator: bounds + determinism (matches structured_random_around) ──
def test_jitter_bounds_and_shape():
    center = np.concatenate([np.full(12, A_BOUND, np.float32), np.full(3, T_MAX, np.float32)])
    cands = CoinJitterGenerator().sample(center, 32, np.random.default_rng(0))
    assert cands.shape == (32, 15)
    assert cands[:, :12].max() <= A_BOUND + 1e-6 and cands[:, :12].min() >= -A_BOUND - 1e-6
    assert cands[:, 12:].max() <= T_MAX + 1e-6 and cands[:, 12:].min() >= T_MIN - 1e-6


def test_jitter_is_deterministic_per_seed():
    c = np.zeros(15, np.float32)
    a = CoinJitterGenerator().sample(c, 8, np.random.default_rng(3))
    b = CoinJitterGenerator().sample(c, 8, np.random.default_rng(3))
    assert np.array_equal(a, b)


# ── TemplateKModeProposal: K=1 argmax, top-K ordering, index keying ──
class _FakeProposal:
    """clf makes template 2 the argmax (then 3, 1, 0); residual is zero ⇒ centre = denorm(template_norm)."""

    K = 4

    def __init__(self):
        self.templates_norm = (np.arange(4, dtype=np.float32)[:, None] * 0.1 * np.ones((4, 15), np.float32))

    def clf(self, x):
        return torch.tensor([[0.1, 0.2, 5.0, 0.3]], dtype=torch.float32).repeat(x.shape[0], 1)

    def residual(self, x, onehot):
        return torch.zeros((x.shape[0], 15), dtype=torch.float32)


def test_k1_is_classifier_argmax_template():
    modes = TemplateKModeProposal(_FakeProposal(), np.zeros(48, np.float32), 1).modes(None)
    assert len(modes) == 1 and modes[0].mode_id == 2       # template 2 is the argmax
    assert abs(modes[0].prob - float(torch.softmax(torch.tensor([0.1, 0.2, 5.0, 0.3]), -1)[2])) < 1e-5


def test_topk_modes_ordered_by_prob_and_keyed_by_index():
    modes = TemplateKModeProposal(_FakeProposal(), np.zeros(48, np.float32), 4).modes(None)
    assert [m.mode_id for m in modes] == [2, 3, 1, 0]       # descending classifier prob
    probs = [m.prob for m in modes]
    assert probs == sorted(probs, reverse=True)
    assert all(m.center.shape == (15,) for m in modes)      # legal θ centres


def test_k2_returns_top_two_modes():
    modes = TemplateKModeProposal(_FakeProposal(), np.zeros(48, np.float32), 2).modes(None)
    assert len(modes) == 2 and [m.mode_id for m in modes] == [2, 3]


def test_modes_centers_distinct_for_distinct_templates():
    modes = TemplateKModeProposal(_FakeProposal(), np.zeros(48, np.float32), 4).modes(None)
    centers = np.stack([m.center for m in modes])
    assert len({c.tobytes() for c in centers}) == 4         # distinct templates ⇒ distinct centres


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
