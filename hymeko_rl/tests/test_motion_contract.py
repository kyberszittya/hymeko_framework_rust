"""REALISTIC_MOTION_CONTRACT_V1 — limits, metrics, slew-rate limiter, thresholded penalties, velocity-aware certificate,
and the hard acceptance gate. Pure (no MuJoCo)."""
import numpy as np
import pytest

from hymeko_rl.env.motion_contract import (
    MotionLimits, MotionMetrics, TorqueGovernorConfig, assert_realistic_motion, govern_torque, motion_penalties,
    motion_report, slew_limited_position, terminal_velocity_certified)


def test_slew_limit_caps_command_change():
    q_prev = np.zeros(4)
    raw = np.array([1.0, -1.0, 0.5, 2.0])                     # a big requested jump
    out = slew_limited_position(raw, q_prev, joint_vel_limit=2.0, dt=0.01)
    assert np.all(np.abs(out - q_prev) <= 2.0 * 0.01 + 1e-9)  # move ≤ vel_limit·dt
    small = slew_limited_position(np.array([0.005, 0, 0, 0]), q_prev, joint_vel_limit=2.0, dt=0.01)
    assert np.isclose(small[0], 0.005)                        # a within-budget move passes through unchanged


def test_metrics_peaks_and_terminal():
    mm = MotionMetrics()
    for jv in (0.5, 1.0, 2.5, 0.1):                           # velocity rises then settles
        mm.add([jv, 0.0, 0.0, 0.0], [10.0, 0, 0, 0], jv * 0.1, [0.0, 0, 0, 0])
    r = mm.finalize(dt=0.01)
    assert r["peak_joint_vel"] == 2.5 and r["terminal_joint_vel"] == 0.1
    assert r["peak_ee_speed"] == pytest.approx(0.25) and r["n_steps"] == 4


def test_thresholded_penalties_ignore_normal_speed():
    lim = MotionLimits()
    normal = motion_penalties([1.0, 1.0, 1.0, 1.0], [10, 10, 10, 10], 0.5, [0, 0, 0, 0], [0, 0, 0, 0], lim)
    assert normal["speed"] == 0.0 and normal["ee_speed"] == 0.0   # below the safe band ⇒ no penalty
    fast = motion_penalties([5.0, 5, 5, 5], [10, 10, 10, 10], 2.0, [1, 0, 0, 0], [0, 0, 0, 0], lim)
    assert fast["speed"] > 0.0 and fast["ee_speed"] > 0.0 and fast["smooth"] > 0.0   # excess is penalised


def test_terminal_velocity_certificate():
    lim = MotionLimits()
    assert terminal_velocity_certified([0.1, 0.1, 0.0, 0.0], 0.02, lim)        # settled
    assert not terminal_velocity_certified([2.0, 0, 0, 0], 0.02, lim)          # joint still moving
    assert not terminal_velocity_certified([0.1, 0, 0, 0], 0.5, lim)           # EE flying through


def test_gate_passes_realistic_and_raises_on_fast_motion():
    lim = MotionLimits()
    realistic = {"peak_joint_vel": 1.9, "peak_ee_speed": 0.87, "peak_joint_acc": 1600.0}   # 6D-1-like (accel is diagnostic)
    d = assert_realistic_motion(realistic, lim, label="6d1")
    assert d["ok"] and d["margins"]["joint_vel"] > 0
    coin = {"peak_joint_vel": 27.2, "peak_ee_speed": 1.54, "peak_joint_acc": 1671.0}        # the audited coin arm
    with pytest.raises(AssertionError, match="joint_vel 27"):
        assert_realistic_motion(coin, lim, label="coin")


def test_gate_does_not_fail_on_transient_acceleration():
    """A realistic-velocity arm with a huge instantaneous qacc (servo/contact transient) must PASS — accel is not gated."""
    lim = MotionLimits()
    assert_realistic_motion({"peak_joint_vel": 1.9, "peak_ee_speed": 0.8, "peak_joint_acc": 1600.0}, lim)


def test_torque_governor_suppresses_accelerating_but_keeps_braking():
    cfg = TorqueGovernorConfig(qdot_soft=2.0, qdot_hard=3.0)
    # joint moving fast +; accelerating torque (+) must be suppressed, braking torque (−) untouched
    assert govern_torque([1.0], [3.5], cfg)[0] == 0.0            # at/above hard: accelerating torque zeroed
    assert govern_torque([-1.0], [3.5], cfg)[0] == -1.0          # braking torque preserved fully
    mid = govern_torque([1.0], [2.5], cfg)[0]                    # between soft and hard: partial suppression
    assert 0.0 < mid < 1.0
    assert govern_torque([1.0], [1.0], cfg)[0] == 1.0           # below soft: untouched
    # vector + sign preserved, magnitude never increased
    out = govern_torque([1.0, -1.0, 2.0, -2.0], [3.5, 3.5, -3.5, -3.5], cfg)
    assert out.tolist() == [0.0, -1.0, 2.0, 0.0]                # accel(+/+ and -/-) zeroed; braking kept


def test_torque_governor_never_increases_magnitude():
    cfg = TorqueGovernorConfig(1.5, 2.5)                     # over_hard_brake off (default)
    for tau, v in [(3.0, 2.0), (-2.0, 3.0), (1.0, -2.2), (0.5, 0.1)]:
        g = govern_torque([tau], [v], cfg)[0]
        assert abs(g) <= abs(tau) + 1e-9 and (g == 0.0 or np.sign(g) == np.sign(tau))


def test_over_hard_brake_adds_active_braking_above_hard():
    cfg = TorqueGovernorConfig(1.5, 2.5, over_hard_brake=2.0)
    # below hard: no active braking (only the accel-suppression path)
    assert govern_torque([0.0], [2.0], cfg)[0] == 0.0
    # above hard, moving +: an ACTIVE braking torque (negative) is added even with zero command
    g = govern_torque([0.0], [4.0], cfg)[0]
    assert g < 0.0 and abs(g - (-2.0 * (4.0 - 2.5))) < 1e-9   # -over_hard_brake·overspeed
    # above hard, moving −: braking is positive (opposes motion)
    assert govern_torque([0.0], [-4.0], cfg)[0] > 0.0
    # a braking command is reinforced, not blocked
    assert govern_torque([-1.0], [4.0], cfg)[0] < -1.0


def test_motion_report_verdict():
    lim = MotionLimits()
    rep = motion_report({"peak_joint_vel": 1.9, "peak_ee_speed": 0.8, "peak_joint_acc": 1600.0,
                         "n_steps": 20, "terminal_ee_speed": 0.01}, lim, label="t")
    assert rep["within_hard_limits"] and rep["accel_is_diagnostic_only"]
    bad = motion_report({"peak_joint_vel": 27.0, "peak_ee_speed": 1.5, "peak_joint_acc": 1600.0}, lim)
    assert not bad["within_hard_limits"]
