"""Tests for the object-generic Geometric Approach layer (planar 2R analytic IK transit HOME -> READY).

Analytic tests (no physics) cover calibration, closed-form IK round-trip, branch selection/continuity, link geometry,
the object-geometry seam, tip/link validation (accept + reject), path composition, densification, and the home
contracts. Integration tests roll the transit through the frozen governed servo and assert the gate ladder + determinism.
A lightweight performance test bounds the transit wall time.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy, _fingertip_geoms
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.home_states import (
    HOME_STATE_V1_GENERIC, HOME_STATE_V2_READY, HomeState, build_home_snapshot)


@pytest.fixture(scope="module")
def cradle():
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    snap, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    return snap


@pytest.fixture(scope="module")
def coin(cradle):
    return _coin_xy(cradle.branch())


@pytest.fixture(scope="module")
def home(cradle):
    return build_home_snapshot(cradle, HOME_STATE_V1_GENERIC)


@pytest.fixture(scope="module")
def arms(home, coin):
    return pga.build_arms(home, coin)


@pytest.fixture(scope="module")
def fk(home, coin):
    gl, gr = _fingertip_geoms(home.branch().inner.model)
    return pga.make_fk(home, coin, gl, gr)


# ── analytic: calibration + IK ───────────────────────────────────────────────────────────────────────────────────────
def test_calibration_is_clean(arms):
    arm_l, arm_r = arms
    for arm in (arm_l, arm_r):
        assert arm.s0 in (-1.0, 1.0) and arm.s1 in (-1.0, 1.0)
        assert 0.15 < arm.l1 < 0.17 and 0.13 < arm.l2 < 0.15
        assert abs(arm.reach - 0.30) < 1e-6


def test_ik_roundtrip_recovers_tip(arms, fk):
    arm_l, _ = arms
    for a, b in [(-0.5, -0.6), (-0.9, -1.4), (-0.4, -0.3)]:
        _, tip_l, _ = fk(np.array([a, b, 0.0, 2.7]))
        ab = arm_l.ik_cont(tip_l, np.array([a, b]))
        _, ftl, _ = fk(np.array([ab[0], ab[1], 0.0, 2.7]))
        assert np.linalg.norm(ftl - tip_l) < 1e-3


def test_ik_two_branches_have_opposite_elbow_sign(arms, coin):
    arm_l, _ = arms
    tip = pga.CoinStraddleTargets(coin=coin, shell_dist=0.055).precontact().tip_left  # r_base ~0.294, in annulus
    up, down = arm_l.ik(tip, +1.0), arm_l.ik(tip, -1.0)
    assert np.sign(up[1]) != np.sign(down[1]) and abs(up[1]) > 1e-3   # genuine elbow-up / elbow-down split


def test_ik_cont_picks_the_continuous_branch(arms, coin):
    arm_l, _ = arms
    tip = coin + np.array([0.02, 0.05])
    near = arm_l.ik(tip, -1.0)
    got = arm_l.ik_cont(tip, near + np.array([0.01, 0.01]))
    assert np.linalg.norm(pga.wrap(got - near)) < 1e-6      # continuity selects the branch closest to the seed


def test_link_points_matches_forward_kinematics(arms, fk):
    arm_l, arm_r = arms
    for q in (HOME_STATE_V1_GENERIC.q, HOME_STATE_V2_READY.q):
        _, tip_l, tip_r = fk(q)
        assert np.linalg.norm(arm_l.link_points(q[[0, 1]])[2] - tip_l) < 1e-9
        assert np.linalg.norm(arm_r.link_points(q[[2, 3]])[2] - tip_r) < 1e-9


# ── analytic: object-geometry seam + validation ──────────────────────────────────────────────────────────────────────
def test_coin_straddle_targets_opposite_sides(coin):
    pre = pga.CoinStraddleTargets(coin=coin, shell_dist=0.055).precontact()
    assert abs(np.linalg.norm(pre.tip_left - coin) - 0.055) < 1e-9
    assert abs(np.linalg.norm(pre.tip_right - coin) - 0.055) < 1e-9
    assert pre.side_left_rad > 0 > pre.side_right_rad       # left tip above, right tip below the coin


def test_tip_feasible_rejects_far_side_and_accepts_near_base(arms, coin, fk):
    arm_l, _ = arms
    cfg = pga.TransitConfig()
    _, home_tip_l, _ = fk(HOME_STATE_V1_GENERIC.q)
    goal = pga.CoinStraddleTargets(coin=coin).precontact().tip_left
    far = pga._arc_waypoints(home_tip_l, goal, coin, 0.055, +1.0, cfg.n_waypoints)   # over the (unreachable) top
    near = pga._arc_waypoints(home_tip_l, goal, coin, 0.055, -1.0, cfg.n_waypoints)  # near-base
    assert not pga._tip_feasible(far, coin, arm_l, cfg)     # far edge 0.316 > 0.30 reach -> singular
    assert pga._tip_feasible(near, coin, arm_l, cfg)


def test_links_clear_accepts_valid_and_rejects_coin_crossing(arms, coin):
    arm_l, _ = arms
    cfg = pga.TransitConfig()
    clear = np.array([arm_l.ik_cont(coin + 0.055 * np.array([np.cos(t), np.sin(t)]), np.array([-0.5, -0.6]))
                      for t in np.linspace(2.0, 2.6, 20)])
    crossing = np.array([arm_l.ik_cont(coin + 0.001 * np.array([np.cos(t), np.sin(t)]), np.array([-0.5, -0.6]))
                         for t in np.linspace(0.0, 0.3, 20)])   # tip driven onto the coin -> a link crosses it
    assert pga._links_clear(clear, arm_l, coin, cfg)
    assert not pga._links_clear(crossing, arm_l, coin, cfg)


def test_seg_point_dist_basic():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert abs(pga._seg_point_dist(a, b, np.array([0.5, 1.0])) - 1.0) < 1e-9   # perpendicular
    assert abs(pga._seg_point_dist(a, b, np.array([2.0, 0.0])) - 1.0) < 1e-9   # beyond the endpoint


def test_arc_waypoints_direction_controls_sweep(coin):
    start, goal = coin + np.array([0.06, 0.0]), coin + np.array([-0.055, 0.02])
    ccw = pga._arc_waypoints(start, goal, coin, 0.055, +1.0, 60)
    cw = pga._arc_waypoints(start, goal, coin, 0.055, -1.0, 60)
    assert not np.allclose(ccw, cw)                          # opposite sweeps differ
    assert np.linalg.norm(ccw[-1] - goal) < 1e-9 and np.linalg.norm(cw[-1] - goal) < 1e-9


def test_pad_and_branch_consistency():
    traj = np.array([[0.1, -0.2], [0.2, -0.25], [0.3, -0.3]])
    padded = pga._pad(traj, 5)
    assert padded.shape == (5, 2) and np.allclose(padded[-1], traj[-1]) and np.allclose(padded[3:], traj[-1])
    assert pga._branch_consistent(traj)
    assert not pga._branch_consistent(np.array([[0.0, -0.1], [0.0, 0.1]]))      # elbow sign flip


def test_densify_slew_respects_cap():
    qref = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])               # a 1.0-rad jump
    dense = pga._densify_slew(qref, 0.3)
    assert np.abs(np.diff(dense, axis=0)).max() <= 0.3 + 1e-9


# ── home contracts ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_home_states_shapes_and_snapshot(cradle, coin):
    assert HOME_STATE_V1_GENERIC.q.shape == (4,) and HOME_STATE_V2_READY.q.shape == (4,)
    with pytest.raises(AssertionError):
        HomeState("bad", np.zeros(3), "m", "d")
    snap = build_home_snapshot(cradle, HOME_STATE_V1_GENERIC)
    data = snap.branch().inner.data
    assert np.allclose(data.qpos[:4], HOME_STATE_V1_GENERIC.q)
    assert np.allclose(data.qvel[:], 0.0)
    assert np.linalg.norm(data.qpos[4:6] - coin) < 1e-9


def test_v2_is_analytic_ik_of_75mm_targets(arms, coin):
    arm_l, arm_r = arms
    pre = pga.CoinStraddleTargets(coin=coin, shell_dist=0.075).precontact()
    ab_l = arm_l.ik_cont(pre.tip_left, np.array([-0.5, 0.0]))
    ab_r = arm_r.ik_cont(pre.tip_right, np.array([-0.2, 2.2]))
    v2 = np.array([ab_l[0], ab_l[1], ab_r[0], ab_r[1]])
    assert np.max(np.abs(v2 - HOME_STATE_V2_READY.q)) < 3e-3


# ── planning: raises on singular start ───────────────────────────────────────────────────────────────────────────────
def test_plan_rejects_singular_v2_start(home, coin, arms, fk):
    arm_l, arm_r = arms
    with pytest.raises(pga.TransitInfeasible):
        pga.plan_collision_free_transit(home, HOME_STATE_V2_READY.q, pga.CoinStraddleTargets(coin=coin),
                                        arm_l, arm_r, pga.TransitConfig(), fk)


def test_plan_qref_is_slew_feasible_and_single_branch(home, coin, arms, fk):
    arm_l, arm_r = arms
    plan = pga.plan_collision_free_transit(home, HOME_STATE_V1_GENERIC.q, pga.CoinStraddleTargets(coin=coin),
                                           arm_l, arm_r, pga.TransitConfig(), fk)
    slew = float(home.stack.tau_rate * home.stack.control_dt)
    assert plan.max_joint_step <= slew + 1e-9 and plan.single_branch


# ── integration: rolled transit gates + determinism ──────────────────────────────────────────────────────────────────
def test_execute_transit_v1_is_collision_free(home, coin):
    res = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, pga.CoinStraddleTargets(coin=coin), pga.TransitConfig())
    assert res.contacts == 0
    assert res.coin_pert_mm <= 1.0
    assert res.single_branch
    assert res.min_clearance_mm >= 40.0                     # fingertip never penetrates the coin (surface >= 0)


def test_execute_transit_is_deterministic(home, coin):
    tgt, cfg = pga.CoinStraddleTargets(coin=coin), pga.TransitConfig()
    a = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, tgt, cfg)
    b = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, tgt, cfg)
    assert a.coin_pert_mm == b.coin_pert_mm and a.reached_qerr == b.reached_qerr and a.mode == b.mode


def test_ready_handoff_snapshot_is_deterministic(home, coin):
    res = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, pga.CoinStraddleTargets(coin=coin), pga.TransitConfig())
    x, y = res.ready_snapshot.branch().inner.data, res.ready_snapshot.branch().inner.data
    assert np.array_equal(x.qpos, y.qpos) and np.array_equal(x.qvel, y.qvel)


# ── performance ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_transit_wall_time_budget(home, coin):
    tgt, cfg = pga.CoinStraddleTargets(coin=coin), pga.TransitConfig()
    pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, tgt, cfg)     # warm up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, tgt, cfg)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) < 3.0, f"transit median wall {np.median(times):.2f}s exceeds 3 s"
