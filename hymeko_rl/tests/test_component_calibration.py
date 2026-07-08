"""Component calibration: perfect/constant/degenerate critics behave as specified, within-state agreement is exact."""
from __future__ import annotations

import numpy as np

from hymeko_rl.eval.component_calibration import _within_state_agreement, calibrate_component


def test_perfect_critic_is_calibrated():
    """Q == mc → spearman 1, within-state agreement 1, calibrated True."""
    rng = np.random.default_rng(0)
    mc = rng.normal(size=200).astype(np.float32)
    state_id = np.repeat(np.arange(40), 5)
    is_ood = np.zeros(200, np.float32)
    cal = calibrate_component("progress", q=mc.copy(), mc=mc, is_ood=is_ood, state_id=state_id)
    assert cal.spearman_q_mc > 0.99
    assert cal.within_state_rank > 0.99
    assert cal.calibrated


def test_anticorrelated_critic_not_calibrated():
    mc = np.linspace(0, 1, 100, dtype=np.float32)
    state_id = np.repeat(np.arange(20), 5)
    cal = calibrate_component("progress", q=-mc, mc=mc, is_ood=np.zeros(100, np.float32), state_id=state_id)
    assert cal.spearman_q_mc < 0
    assert cal.within_state_rank < 0.5
    assert not cal.calibrated


def test_degenerate_target_flagged():
    mc = np.zeros(50, np.float32)  # constant target (e.g. body_progress when the policy never body-shoves)
    cal = calibrate_component("body_progress", q=np.random.default_rng(0).normal(size=50).astype(np.float32),
                              mc=mc, is_ood=np.zeros(50, np.float32), state_id=np.repeat(np.arange(10), 5))
    assert cal.degenerate
    assert not cal.calibrated


def test_ood_gap_sign():
    """In-distribution Q above OOD Q → positive ood_gap."""
    q = np.concatenate([np.full(50, 2.0), np.full(50, -1.0)]).astype(np.float32)
    mc = q.copy()
    is_ood = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.float32)
    cal = calibrate_component("contact", q=q, mc=mc, is_ood=is_ood, state_id=np.arange(100))
    assert cal.ood_gap > 0


def test_within_state_agreement_exact():
    # two states, each with two actions; Q order matches mc order in state 0, opposes in state 1
    q = np.array([1.0, 0.0, 0.0, 1.0])
    mc = np.array([1.0, 0.0, 1.0, 0.0])
    sid = np.array([0, 0, 1, 1])
    frac, n = _within_state_agreement(q, mc, sid)
    assert n == 2 and abs(frac - 0.5) < 1e-9
