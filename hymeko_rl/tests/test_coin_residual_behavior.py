"""§3 gated residual behavior contract (pure-function unit tests): gate=0 ⇒ composite==base bit-identically for any
residual; residual clipped to ±0.25; composite clipped to ±4; gate=1 applies the bounded residual."""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.coin_residual_behavior import gated_composite_action


def test_gate_zero_is_base_for_any_residual():
    base = np.array([1.0, -2.0, 0.5, 3.9], np.float32)
    for delta in (np.zeros(4, np.float32), np.full(4, 0.25, np.float32), np.full(4, 100.0, np.float32),
                  np.random.default_rng(0).standard_normal(4).astype(np.float32)):
        a = gated_composite_action(base, 0.0, delta)
        assert np.array_equal(a, np.clip(base, -4, 4))          # bit-identical to clip(base)


def test_gate_one_applies_bounded_residual():
    base = np.array([0.0, 0.0, 0.0, 0.0], np.float32)
    a = gated_composite_action(base, 1.0, np.array([0.1, -0.1, 0.5, -0.5], np.float32))
    assert np.allclose(a, np.array([0.1, -0.1, 0.25, -0.25]))   # residual clipped to ±0.25


def test_composite_clipped_to_action_bounds():
    base = np.array([3.95, -3.95, 0.0, 0.0], np.float32)
    a = gated_composite_action(base, 1.0, np.array([0.25, -0.25, 0.0, 0.0], np.float32))
    assert a.max() <= 4.0 + 1e-6 and a.min() >= -4.0 - 1e-6


def test_residual_bound_enforced_independent_of_request():
    base = np.zeros(4, np.float32)
    a = gated_composite_action(base, 1.0, np.array([10.0, -10.0, 10.0, -10.0], np.float32))
    assert np.allclose(np.abs(a), 0.25)                          # request clamped to the residual bound
