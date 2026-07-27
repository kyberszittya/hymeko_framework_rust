"""R9 causal residual — fast pure-logic tests (no physics; the full-trace update-zero identity is the STAGE 2 harness).

Covers: the causal-increment bounds, the zero-Δ actor, the slew clamp, and the a_exec = clip(a_R8 + Δa) formula whose
Δa≡0 case must return the frozen base a_R8 exactly (the identity the STAGE 2 physics harness then verifies bit-for-bit).
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import (
    CAUSAL_DIM, DELTA_ROLES, ConstantDeltaActor, DeltaBounds, ZeroDeltaActor)
from hymeko_rl.coin_delivery.theta_option.residual_adapter import ResidualBounds


def test_r9_delta_bounds_and_actors():
    assert len(DELTA_ROLES) == 3 and CAUSAL_DIM == 11 + 3 + 3 + 4 * 6
    b = DeltaBounds()
    assert np.all(b.vec() > 0) and b.slew > 0
    # the increment bounds are SMALLER than the R8 residual bounds (Δa nudges; a_R8 carries the base)
    assert np.all(b.vec() <= ResidualBounds().vec() + 1e-12)
    assert np.array_equal(ZeroDeltaActor()(np.zeros(CAUSAL_DIM)), np.zeros(3))
    d = ConstantDeltaActor(np.array([0.5, -0.3, 0.9]))(np.zeros(CAUSAL_DIM))
    assert d.shape == (3,) and np.allclose(d, [0.5, -0.3, 0.9])


def test_r9_zero_delta_a_exec_is_base():
    """a_exec = clip(a_R8 + Δa·dbounds/bounds); at Δa=0 it must equal the (already-clipped) base a_R8 exactly."""
    dbounds, bounds = DeltaBounds(), ResidualBounds()
    for a_r8 in (np.array([0.68, -0.99, 0.97]), np.array([-0.995, -0.90, 0.95]), np.zeros(3)):
        a_clip = np.clip(a_r8, -1.0, 1.0)
        a_exec0 = np.clip(a_clip + np.zeros(3) * dbounds.vec() / bounds.vec(), -1.0, 1.0)
        assert np.array_equal(a_exec0, a_clip)          # Δa=0 ⇒ a_exec == a_R8 (the update-zero identity, in algebra)


def test_r9_slew_clamp():
    """A per-decision increment is slew-limited to |Δa_t − Δa_{t-1}| ≤ slew per role (anti-chatter)."""
    b = DeltaBounds()
    prev = np.array([0.0, 0.2, -0.1])
    raw = np.array([1.0, -1.0, 0.5])                    # a large jump
    slewed = np.clip(raw, prev - b.slew, prev + b.slew)
    assert np.all(np.abs(slewed - prev) <= b.slew + 1e-12)
    assert np.allclose(slewed, [b.slew, 0.2 - b.slew, -0.1 + b.slew])
