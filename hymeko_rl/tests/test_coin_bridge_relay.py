"""Tests for the bridge-relay mechanism: feature sanitization, readiness detector + hysteresis, threshold calibration,
relay mode logic, and the bridge reward (the frozen-transport-policy integration is exercised by the driver smoke)."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.experiments.coin_bridge_relay import (
    BridgeReward, ReadinessDetector, calibrate_thresholds, ready_features,
)
from hymeko_rl.experiments.coin_bridge_relay_run import _classify


def test_ready_features_sanitizes_inf() -> None:
    obs = np.zeros(41, np.float32)
    obs[2] = np.inf                                                  # coin_vx can go inf on a dt=0 step
    obs[3] = -np.inf
    f = ready_features(obs)
    assert np.isfinite(f).all()                                      # clipped, never poisons the kNN


def test_detector_distance_zero_on_bank_member() -> None:
    bank = np.random.default_rng(0).standard_normal((6, 15)).astype(np.float32)
    det = ReadinessDetector(bank, enter_thresh=0.5, exit_thresh=1.0)
    # a bank member's standardized distance to itself is 0 → below any positive threshold
    obs = np.zeros(41, np.float32)
    from hymeko_rl.experiments.coin_bridge_relay import _FEAT_IDX
    obs[_FEAT_IDX] = bank[0]
    assert det.distance(obs) == pytest.approx(0.0, abs=1e-4)


def test_detector_hysteresis_enter_vs_exit() -> None:
    bank = np.zeros((3, 15), np.float32)
    det = ReadinessDetector(bank, enter_thresh=0.5, exit_thresh=2.0)
    obs = np.zeros(41, np.float32)
    from hymeko_rl.experiments.coin_bridge_relay import _FEAT_IDX
    obs[_FEAT_IDX[0]] = 1.0                                          # push one std out (sd is tiny → large z)
    # a mid-distance state can be "not ready to enter" but "still ready to stay" (hysteresis)
    d = det.distance(obs)
    assert det.is_ready(obs, currently_transport=True) == (d <= 2.0)
    assert det.is_ready(obs, currently_transport=False) == (d <= 0.5)


def test_detector_rejects_empty_bank() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ReadinessDetector(np.zeros((0, 15), np.float32), enter_thresh=1.0, exit_thresh=2.0)


def test_calibrate_excludes_self() -> None:
    """A READY state's nearest neighbour is itself (distance 0); calibration must use the nearest OTHER ready state,
    so the threshold reflects basin spread, not 0."""
    feats = np.array([[0.0] * 15, [1.0] * 15, [2.0] * 15], np.float32)
    labels = ["TRANSPORT_READY", "TRANSPORT_READY", "TRANSPORT_READY"]
    enter, exit_ = calibrate_thresholds(feats, labels, feats)
    assert enter > 0.0 and exit_ == pytest.approx(2 * enter)


def test_bridge_reward_dwell_terminal_dominates() -> None:
    rw = BridgeReward()
    assert rw.dwell_target == 3
    assert rw.r_dwell > rw.r_first_ready                            # the 3-step dwell bonus dominates momentary entry
    assert rw.r_dwell > rw.w_streak * rw.dwell_target              # …and the accumulated streak bonus


def test_classify_bridge_positive() -> None:
    c1 = dict(transport_alone=1, trained_relay=5, zero_action=0)
    c2 = dict(transport_alone=0, trained_relay=4, zero_action=0)
    bands = [dict(held1=dict(ready_entry=6), held2=dict(ready_entry=3))]
    assert _classify(c1, c2, bands) == "BRIDGE_POSITIVE"            # relay 9 >= 8, > transport 1, zero 0


def test_classify_contact_positive_and_no_effect() -> None:
    c1 = dict(transport_alone=1, trained_relay=1, zero_action=0)
    c2 = dict(transport_alone=0, trained_relay=0, zero_action=0)
    bands = [dict(held1=dict(ready_entry=3), held2=dict(ready_entry=1))]
    assert _classify(c1, c2, bands) == "BRIDGE_CONTACT_POSITIVE"    # ready entry > 0, relay >= transport
    bands0 = [dict(held1=dict(ready_entry=0), held2=dict(ready_entry=0))]
    assert _classify(c1, c2, bands0) == "NO_EFFECT"
