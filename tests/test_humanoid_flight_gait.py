"""The dynamic FLIGHT-PHASE gait — a genuine (contact-verified) flight phase the quasi-static stack can't do."""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.flight_gait import PDIM, FlightGaitConfig, FlightGaitEnv, _P


def _seed_theta() -> np.ndarray:
    d = dict(freq=2.0, hip_amp=0.5, hip_off=0.0, knee_amp=0.5, knee_off=0.0, knee_crouch=0.3,
             ankle_amp=0.3, ankle_off=0.0, push_amp=0.8, lean=0.0, arm_amp=0.4)
    return np.array([d[p] for p in _P], np.float64)


def test_rollout_is_finite_and_deterministic() -> None:
    env = FlightGaitEnv(FlightGaitConfig(steps=200), seed=0)
    r0 = env.rollout(_seed_theta(), seed=0)
    r1 = env.rollout(_seed_theta(), seed=0)
    assert all(np.isfinite(x) for x in r0)
    assert np.allclose(r0, r1)                                # same theta + seed → same rollout


def test_stronger_pushoff_lifts_off_more_than_none() -> None:
    """Push-off EARNS flight: a strong push-off spends more time fully airborne than a no-push gait (relative,
    honest — from a planted start the absolute flight fraction is small; the earlier ~20% was a drop artifact)."""
    env = FlightGaitEnv(FlightGaitConfig(steps=900), seed=0)
    strong = _seed_theta()
    strong[_P.index("push_amp")] = 1.5
    strong[_P.index("knee_amp")] = 0.8
    no_push = _seed_theta()
    no_push[_P.index("push_amp")] = 0.0
    _r, _f, fl_strong, up = env.rollout(strong, seed=0)
    _r, _f, fl_none, _u = env.rollout(no_push, seed=0)
    assert fl_strong > fl_none                                # the push-off buys airborne time the no-push gait lacks
    assert up > 0.8                                           # and it stays upright


def test_reset_plants_the_feet_and_is_deterministic() -> None:
    """A planted grounded start (not the earlier feet-in-the-air drop) + deterministic rollout (warmstart cleared)."""
    env = FlightGaitEnv(FlightGaitConfig(steps=200), seed=0)
    env.reset(seed=0)
    assert not env._both_feet_airborne()                     # settled onto the floor (a floor contact exists)
    r0 = env.rollout(_seed_theta(), seed=0)
    r1 = env.rollout(_seed_theta(), seed=0)
    assert np.allclose(r0, r1)                                # mj_resetData clears warmstart → reproducible


def test_flight_detection_is_contact_based() -> None:
    """Flight must use floor contact, not the foot-body-z (the anatomical foot origin sits ~0.22 m up = first bug)."""
    env = FlightGaitEnv(FlightGaitConfig(steps=50), seed=0)
    env.reset(seed=0)
    assert env._floor >= 0                                    # floor geom resolved
    _r, _f, flight, _u = env.rollout(np.array([1.0, *([0.0] * (PDIM - 1))]), seed=0)  # a clock, no leg motion
    assert flight < 0.1                                       # stays grounded without a push-off (not 100% = the bug)
