"""Tests for the exact-zero-HOME reach milestone: the geometric positive control + the robust RRT-Connect reach planner.

Fast: the exact-reset gate (first frame is q=[0,0,0,0]), the segment-segment distance, the inflated-coin collision
checker, the straddle goal SET, and that RRT-Connect returns a collision-free path from the exact zero reset to a goal.
Physics (slow): the honest positive control — from q=[0,0,0,0], the task-space reach + READY-specific CEM capture +
frozen downstream delivers strict K6 with no premature contact and negligible pre-capture coin motion (the no-teleport /
continuous-state gate). CEM is still a teacher/oracle for the capture, not teacher-free deployment.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments import coin_zero_home_rrt as R


@pytest.fixture(scope="module")
def rig():
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
    return _rig()


# ── fast ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_zero_home_is_exact_zero_reset(rig):
    home, _coin = Z._home_with_coin(rig, None)
    assert np.array_equal(home.branch().inner.data.qpos[:4], np.zeros(4))          # EXACT-RESET gate: q=[0,0,0,0]


def test_seg_seg_dist_crossing_and_parallel():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    assert R._seg_seg_dist(a, b, np.array([0.5, -1.0]), np.array([0.5, 1.0])) == 0.0        # crossing -> 0
    assert R._seg_seg_dist(a, b, np.array([0.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)  # parallel, 1 apart


def test_collision_checker_start_clear_and_rejects_tip_on_coin(rig):
    home, coin = Z._home_with_coin(rig, None)
    arm_l, arm_r = pga.build_arms(home, coin)
    cfg = pga.TransitConfig()
    assert R._collision_free(np.zeros(4), coin, arm_l, arm_r, cfg)                 # zero home is valid for the canonical coin
    q_bad = np.concatenate([arm_l.ik(coin, 1.0), arm_r.ik(coin, 1.0)])            # both tips driven onto the coin centre
    assert not R._collision_free(q_bad, coin, arm_l, arm_r, cfg)                   # keep-out violated -> rejected


def _coll_fn(coin, arm_l, arm_r, cfg):
    def coll(q):
        return R._collision_free(q, coin, arm_l, arm_r, cfg)
    return coll


def test_straddle_goal_set_nonempty_and_valid(rig):
    home, coin = Z._home_with_coin(rig, None)
    arm_l, arm_r = pga.build_arms(home, coin)
    coll = _coll_fn(coin, arm_l, arm_r, pga.TransitConfig())
    goals = R._straddle_goal_set(coin, arm_l, arm_r, pga.CoinStraddleTargets(coin=coin), coll)
    assert len(goals) >= 8 and all(coll(g) for g in goals)                        # a real GOAL SET, all collision-free


def test_rrt_connect_finds_collision_free_path_from_exact_zero(rig):
    home, coin = Z._home_with_coin(rig, None)
    arm_l, arm_r = pga.build_arms(home, coin)
    coll = _coll_fn(coin, arm_l, arm_r, pga.TransitConfig())
    goals = R._straddle_goal_set(coin, arm_l, arm_r, pga.CoinStraddleTargets(coin=coin), coll)
    path = R.rrt_connect(np.zeros(4), goals, coll, np.random.default_rng(0))
    assert path is not None
    assert np.array_equal(path[0], np.zeros(4))                                   # starts at the exact zero reset
    assert any(np.allclose(path[-1], g) for g in goals)                          # ends at a goal-set configuration
    assert all(R._edge_free(a, b, coll) for a, b in zip(path[:-1], path[1:]))     # every edge is collision-free


# ── physics (slow: reach + CEM capture) ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_exact_zero_home_positive_control_delivers_strict_k6():
    r = Z.reach_and_deliver()
    assert r["reached"] and r["first_frame_q"] == [0.0, 0.0, 0.0, 0.0]            # EXACT-RESET
    assert r["reach_contacts"] == 0 and r["coin_moved_before_capture_mm"] < 5.0   # no premature contact / no teleport
    assert r["capture_k6"] and r["min_dtz_mm"] < 20.0 and r["safe"]               # strict-K6 positive control
