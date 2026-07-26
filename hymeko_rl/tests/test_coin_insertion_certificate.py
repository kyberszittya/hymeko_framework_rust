"""TASK CONTACT-LEGALITY (behavioural) tests — separate from the physical collision tests.

Physics decides what is POSSIBLE (every arm geom collides with the coin); the task certificate decides what SOLUTION we
accept. Link contact is allowed (morphology-assisted guiding); the forbidden shortcut is a BALLISTIC KNOCK. These tests
prove: controlled insertion is accepted, a ballistic knock is rejected, and the current teacher grades E0
(WHOLE_ARM_ASSISTED_INSERTION — not fingertip-dominant).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option.insertion_certificate import (
    controlled_insertion_pass, is_ballistic_knock)


def _m(*, forward=0.05, peak_coin_speed=0.5, peak_qdot=1.5, terminal_coin_speed=0.0, k6=True):
    return {"forward": forward, "peak_coin_speed": peak_coin_speed, "peak_qdot": peak_qdot,
            "terminal_coin_speed": terminal_coin_speed, "k6_delivered": k6}


# ───────────────────────────── certificate logic (fast, no physics) ─────────────────────────────
def test_controlled_insertion_accepts_a_controlled_delivery():
    m = _m(forward=0.05, peak_coin_speed=0.45, terminal_coin_speed=0.0, k6=True)   # target-directed, braked, K6
    assert controlled_insertion_pass(m) is True and is_ballistic_knock(m) is False


def test_ballistic_knock_is_rejected_by_overspeed():
    m = _m(peak_coin_speed=2.0, k6=True)                                           # exceeds ee_speed_hard 1.5
    assert is_ballistic_knock(m) is True and controlled_insertion_pass(m) is False


def test_ballistic_knock_is_rejected_by_flythrough():
    m = _m(peak_coin_speed=0.5, k6=False)                                          # fast (>3*SETTLE) but never settles
    assert is_ballistic_knock(m) is True and controlled_insertion_pass(m) is False


def test_controlled_insertion_rejects_no_k6_or_high_terminal_speed():
    assert controlled_insertion_pass(_m(k6=False, peak_coin_speed=0.1)) is False   # never dwelled
    assert controlled_insertion_pass(_m(terminal_coin_speed=0.2)) is False         # not braked (flies on)
    assert controlled_insertion_pass(_m(forward=-0.01)) is False                   # not target-directed


# ───────────────────────────── live physics (slow) ─────────────────────────────
@pytest.mark.slow
def test_teacher_delivery_is_controlled_insertion_e0_whole_arm():
    """The frozen teacher θ on a dev cradle passes CONTROLLED_INSERTION and grades E0 (whole-arm assisted): link contact
    is present, it is not a ballistic knock, and the fingertip impulse share is well below fingertip-dominant."""
    bank_path = "reports/2026-07-27-coin-teacher-to-rl/teacher_bank.json"
    if not os.path.exists(bank_path):
        pytest.skip("teacher_bank.json not built")
    from hymeko_rl.coin_delivery.theta_option.insertion_certificate import grade_delivery
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    e = [s for s in json.load(open(bank_path))["states"] if s["tag"] == "s1"][0]
    snap, _ = acquire_snapshot(load_harness(), e["seed"])
    g = grade_delivery(snap, e["canonical_theta_vec"])
    assert g.controlled_insertion is True and g.ballistic_knock is False and g.k6_delivered is True
    assert g.level == "E0_WHOLE_ARM_ASSISTED_INSERTION"
    assert g.fingertip_impulse_share is not None and g.fingertip_impulse_share < 0.5   # not fingertip-dominant


@pytest.mark.slow
def test_no_brake_max_forward_theta_is_not_controlled_insertion():
    """A no-braking, maximum-forward θ (a shove) does NOT pass CONTROLLED_INSERTION — it either exceeds the speed bound,
    flies through without the K6 dwell, or leaves a high terminal speed (a knock, not a controlled insertion)."""
    bank_path = "reports/2026-07-27-coin-teacher-to-rl/teacher_bank.json"
    if not os.path.exists(bank_path):
        pytest.skip("teacher_bank.json not built")
    from hymeko_rl.coin_delivery.theta_option.insertion_certificate import grade_delivery
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    e = [s for s in json.load(open(bank_path))["states"] if s["tag"] == "s1"][0]
    snap, _ = acquire_snapshot(load_harness(), e["seed"])
    knock = np.array([0.02, 0.30, 0.0, 4.0, 40.0, 0.0])       # min grip, MAX forward, no brake, late release
    g = grade_delivery(snap, knock)
    assert g.controlled_insertion is False                    # a shove is not a controlled insertion
