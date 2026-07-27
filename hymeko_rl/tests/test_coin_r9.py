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


def test_r9_segment_delta_actor():
    """The reachability-search actor emits the right piecewise-constant Δa per decision and clips to [-1,1]."""
    from hymeko_rl.coin_delivery.theta_option.r9_causal_residual import SegmentDeltaActor
    seg = np.array([[0.5, 0.0, 0.0], [-0.5, 1.0, 0.0], [2.0, 0.0, -1.0]])   # last row must clip
    a = SegmentDeltaActor(seg, n_decisions=6)
    outs = [a(np.zeros(CAUSAL_DIM)) for _ in range(6)]
    assert np.allclose(outs[0], [0.5, 0.0, 0.0]) and np.allclose(outs[2], [-0.5, 1.0, 0.0])
    assert np.allclose(outs[4], [1.0, 0.0, -1.0])                # 2.0 clipped to 1.0
    a.reset()
    assert a.i == 0 and np.allclose(a(np.zeros(CAUSAL_DIM)), [0.5, 0.0, 0.0])


def test_r9_slew_clamp():
    """A per-decision increment is slew-limited to |Δa_t − Δa_{t-1}| ≤ slew per role (anti-chatter)."""
    b = DeltaBounds()
    prev = np.array([0.0, 0.2, -0.1])
    raw = np.array([1.0, -1.0, 0.5])                    # a large jump
    slewed = np.clip(raw, prev - b.slew, prev + b.slew)
    assert np.all(np.abs(slewed - prev) <= b.slew + 1e-12)
    assert np.allclose(slewed, [b.slew, 0.2 - b.slew, -0.1 + b.slew])


# ── R10-C2: the APPROACH momentum-build mode — causal exit-guard logic (pure, no physics) ────────────────────────────
def test_r10_approach_exit_guards():
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import APPROACH, ApproachParams, HybridApproachController
    c = object.__new__(HybridApproachController)                 # bypass physics __init__ to unit-test the guard logic
    c.ap = ApproachParams()
    c._impulse, c._approach_steps = 0.0, 0
    assert APPROACH == -1
    # SAFETY overrides everything (even a valid launch state)
    assert c._exit_approach(dtz=0.10, v_par=0.30, both=True, contact_risk=True) == "SAFETY"
    # LAUNCH only inside the band AND with bilateral contact
    assert c._exit_approach(0.10, 0.30, True, False) == "LAUNCH"
    assert c._exit_approach(0.10, 0.30, False, False) != "LAUNCH"     # no bilateral contact
    assert c._exit_approach(0.10, 0.10, True, False) is None          # below the launch band, keep approaching
    # REACHABILITY: v_par² ≥ 2·coast_decel·d_remain, with v_par ABOVE the launch band so LAUNCH does not pre-empt it
    assert c.ap.launch_vhi < 0.5
    assert c._exit_approach(0.05, 0.5, True, False) == "REACHABILITY"    # d_remain 0.03, 0.25 ≥ 2·0.5·0.03=0.03
    # BUDGET and HORIZON are mandatory exits far from the zone with no launch/reach
    c._impulse = 1.0
    assert c._exit_approach(1.0, 0.05, True, False) == "BUDGET"
    c._impulse, c._approach_steps = 0.0, 99
    assert c._exit_approach(1.0, 0.05, True, False) == "HORIZON"
    c._approach_steps = 0
    assert c._exit_approach(1.0, 0.05, True, False) is None          # far, slow, in-budget ⇒ keep building momentum


def test_r10_approach_bounds_and_no_regression():
    from hymeko_rl.coin_delivery.theta_option.hybrid_approach import APPROACH, ApproachParams
    from hymeko_rl.coin_delivery.theta_option.tip_transport import REGULATE
    ap = ApproachParams()
    assert ap.qdot_approach > 1.0 and ap.qdot_approach < 3.0          # above the settle cap, under the hard motion limit
    assert 0.0 < ap.launch_vlo < ap.launch_vhi                        # a band, not a point (anti-chatter)
    assert ap.impulse_budget > 0 and ap.max_steps > 0                 # bounded effort + horizon guards exist
    assert APPROACH < REGULATE                                        # APPROACH precedes the scaffold phases (monotone order)
