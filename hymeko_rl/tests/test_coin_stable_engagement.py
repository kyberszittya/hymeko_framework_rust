"""Contract tests for the hybrid STABLE_OBJECT_ENGAGEMENT_V1 gate (§1-§6): bilateral fast path, unilateral co-motion
slow path (duration + coin/tip co-motion + slip bound), hysteresis, settling-no-disarm, and V2 controller-state resume.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.coin_stable_engagement import (
    ArmMechanism,
    EngagementMode,
    StableEngagementConfig,
    StableEngagementGate,
)

FAR = [10.0, 10.0]        # non-contacting tip parked far away (irrelevant)


def feed(g, seq):
    return [g.update(*step)[0] for step in seq]


def comoving(n, *, start=(0.10, 0.15), step=(0.002, 0.0), tip_lag=(-0.005, 0.0), side="L"):
    """n steps of one-sided contact with the coin co-moving with the contacting tip."""
    out = []
    for t in range(n):
        coin = [start[0] + step[0] * t, start[1] + step[1] * t]
        tip = [coin[0] + tip_lag[0], coin[1] + tip_lag[1]]
        lc, rc = (side == "L"), (side == "R")
        ltip = tip if side == "L" else FAR
        rtip = tip if side == "R" else FAR
        out.append((lc, rc, coin, ltip, rtip))
    return out


def test_bilateral_fast_arms_at_3():
    g = StableEngagementGate()
    outs = [g.update(True, True, [0.1, 0.15], [0.09, 0.15], [0.11, 0.15]) for _ in range(4)]
    gs = [o[0] for o in outs]
    assert gs == [0.0, 0.0, 1.0, 1.0]
    assert outs[2][1] == ArmMechanism.BILATERAL_FAST.value
    assert g.s.mode == EngagementMode.LATE_CONTROL_ARMED.value


def test_unilateral_short_never_arms():
    g = StableEngagementGate(StableEngagementConfig(uni_arm_after=6))
    gs = feed(g, comoving(5))                      # only 5 same-side steps < 6
    assert set(gs) == {0.0}
    assert g.s.mode == EngagementMode.EARLY_CONTROL.value


def test_unilateral_comotion_arms_at_6():
    g = StableEngagementGate()
    outs = [g.update(*s) for s in comoving(7)]
    assert outs[-1][0] == 1.0
    assert g.s.last_arm_mechanism == ArmMechanism.UNILATERAL_COMOTION.value


def test_unilateral_stationary_coin_no_arm():
    # coin does not move -> below motion floor -> co-motion fails even at 7 same-side steps
    g = StableEngagementGate()
    gs = feed(g, comoving(8, step=(0.0, 0.0)))
    assert set(gs) == {0.0}


def test_unilateral_opposite_motion_no_arm():
    # coin moves opposite to the tip -> directional agreement fails
    g = StableEngagementGate()
    seq = []
    for t in range(8):
        coin = [0.10 - 0.002 * t, 0.15]           # coin -x
        tip = [0.10 + 0.002 * t, 0.15]            # tip +x
        seq.append((True, False, coin, tip, FAR))
    assert set(feed(g, seq)) == {0.0}


def test_unilateral_high_slip_no_arm():
    # coin moves WITH tip direction but tip races far ahead -> slip bound exceeded
    g = StableEngagementGate()
    seq = []
    for t in range(8):
        coin = [0.10 + 0.002 * t, 0.15]
        tip = [0.10 + 0.02 * t, 0.15]             # 10x faster -> slip >> bound
        seq.append((True, False, coin, tip, FAR))
    assert set(feed(g, seq)) == {0.0}


def test_alternating_LR_no_accumulation():
    g = StableEngagementGate()
    seq = []
    for t in range(12):
        side = "L" if t % 2 == 0 else "R"
        coin = [0.10 + 0.002 * t, 0.15]; tip = [coin[0] - 0.005, 0.15]
        seq.append(((side == "L"), (side == "R"), coin, tip if side == "L" else FAR, tip if side == "R" else FAR))
    assert set(feed(g, seq)) == {0.0}
    assert g.arm_count == 0


def test_disarm_on_two_loss_then_reacquire():
    g = StableEngagementGate()
    feed(g, comoving(7))                            # armed via co-motion
    assert g.gate == 1.0
    # complete contact loss for 2 steps -> REACQUIRE
    g.update(False, False, [0.114, 0.15], FAR, FAR)
    g.update(False, False, [0.114, 0.15], FAR, FAR)
    assert g.s.mode == EngagementMode.REACQUIRE.value and g.gate == 0.0
    # bilateral re-arm
    for _ in range(3):
        g.update(True, True, [0.114, 0.15], [0.10, 0.15], [0.13, 0.15])
    assert g.gate == 1.0 and g.arm_count == 2


def test_settling_does_not_disarm():
    # once armed, the coin stops (no motion) but contact holds -> must STAY armed (co-motion only required to arm)
    g = StableEngagementGate()
    feed(g, comoving(7))
    assert g.gate == 1.0
    for _ in range(20):                            # coin frozen, contact maintained (settling/dwell)
        g.update(True, False, [0.114, 0.15], [0.109, 0.15], FAR)
    assert g.gate == 1.0 and g.s.mode == EngagementMode.LATE_CONTROL_ARMED.value


def test_terminal_absorbing():
    g = StableEngagementGate()
    feed(g, comoving(7))
    assert g.update(True, False, [0.11, 0.15], [0.10, 0.15], FAR, terminated=True)[0] == 0.0
    assert g.s.mode == EngagementMode.TERMINAL.value
    assert g.update(True, True, [0.11, 0.15], [0.10, 0.15], [0.12, 0.15])[0] == 0.0


def test_reset_clears():
    g = StableEngagementGate()
    feed(g, comoving(7))
    g.reset()
    assert g.s.mode == EngagementMode.EARLY_CONTROL.value
    assert g.s.uni_counter == 0 and len(g._coin) == 0


def test_controller_state_v2_resume_reproduces_transition():
    # §6: serialize mid-episode, resume in a fresh gate, and the NEXT transition + gate must match bit-for-bit
    g = StableEngagementGate()
    seq = comoving(9)
    for s in seq[:6]:
        g.update(*s)
    saved = g.state_v2()
    g2 = StableEngagementGate(); g2.load_state_v2(saved)
    a = g.update(*seq[6]); b = g2.update(*seq[6])
    assert a == b
    assert g.state_v2()["mode"] == g2.state_v2()["mode"]
    assert g.state_v2()["uni_counter"] == g2.state_v2()["uni_counter"]


def test_contract_v2_sha_deterministic_and_config_sensitive():
    a = StableEngagementGate().contract_v2_sha256()
    b = StableEngagementGate().contract_v2_sha256()
    c = StableEngagementGate(StableEngagementConfig(uni_arm_after=8)).contract_v2_sha256()
    assert a == b and a != c and len(a) == 64


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        StableEngagementConfig(uni_arm_after=3, bilateral_arm_after=3)   # slow path not stricter
    with pytest.raises(ValueError):
        StableEngagementConfig(kin_window=0)
