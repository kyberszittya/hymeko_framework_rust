"""Unit tests for the lifted-horizon authority measurement (pure logic + the rate-limiter integrator identity).

The heavy MuJoCo rollout is exercised by the benchmark (integration, production scale); here we pin the PURE decision
machinery and the mechanism that motivates the whole session: the torque-rate limiter is an INTEGRATOR that absorbs a
one-step ±ε command (Session-1 dead zone) but lets it through once the acquisition debt is spent (Session-2 authority).
"""
from __future__ import annotations

import numpy as np

from hymeko_rl.coin_delivery.horizon_authority import (
    HORIZONS, HorizonConfig, _bv_at, _rank2_before_collapse, _rank2_onset, _repro_gap, _state_route, _sustains,
    collapse_step, cradle_alive, decide_campaign_route, first_crossing, numeric_rank)
from hymeko_rl.env.governed_arm import V3Stack, pd_governed_torque


# --------------------------------------------------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------------------------------------------------
def test_first_crossing():
    assert first_crossing([0.0, 0.0, 1e-3, 5.0], 1e-6) == 3
    assert first_crossing([0.0, 0.0, 0.0], 1e-6) is None
    assert first_crossing([2.0], 1e-6) == 1


def test_numeric_rank_full_deficient_and_nan():
    assert numeric_rank(np.eye(4), 1e-3, 1e-2) == 4
    M = np.zeros((4, 4))
    M[:, 0], M[:, 1] = [1, 0, 0, 0], [0, 1, 0, 0]
    assert numeric_rank(M, 1e-3, 1e-2) == 2
    N = np.full((4, 4), np.nan)
    N[:, 0], N[:, 2] = [1, 0, 0, 0], [0, 1, 0, 0]                                # two finite independent columns
    assert numeric_rank(N, 1e-3, 1e-2) == 2
    assert numeric_rank(np.full((4, 4), np.nan), 1e-3, 1e-2) == 0
    tiny = np.zeros((4, 4))
    tiny[:, 0] = [1e-4, 0, 0, 0]                                                 # below abs_tol ⇒ rank 0
    assert numeric_rank(tiny, 1e-3, 1e-2) == 0


def test_cradle_alive_pass_and_each_failure():
    cfg = HorizonConfig()
    good = {"per": {"left": {"present": True, "same_identity": True, "fn": 2.0, "count": 1},
                    "right": {"present": True, "same_identity": True, "fn": 2.0, "count": 1}}, "straddle_live": -0.9}
    ok, sig = cradle_alive(good, 0.5, cfg)
    assert ok and all(sig.values())
    lost = {"per": {"left": {"present": False, "same_identity": False, "fn": 0.0, "count": 0},
                    "right": {"present": True, "same_identity": True, "fn": 2.0, "count": 1}}, "straddle_live": -0.9}
    assert cradle_alive(lost, 0.5, cfg)[0] is False
    no_straddle = dict(good, straddle_live=0.5)
    assert cradle_alive(no_straddle, 0.5, cfg)[0] is False
    assert cradle_alive(good, 99.0, cfg)[0] is False                             # motion contract breach


def test_collapse_step():
    assert collapse_step([True, True, True]) == 4       # survives → len+1
    assert collapse_step([True, False, True]) == 2
    assert collapse_step([False]) == 1


# --------------------------------------------------------------------------------------------------------------------
# synthetic column authorities → assembly / rank / onset / sustain / reproducibility
# --------------------------------------------------------------------------------------------------------------------
def _synthetic_columns(onset: int, collapse: int, cfg: HorizonConfig, scale: float = 1.0):
    """Two independent columns that light up at `onset` and die at `collapse` (others null) — a rank-2 authority band."""
    H = cfg.h_max
    cols = []
    for j in range(4):
        alive = np.array([(h + 1) < collapse for h in range(H)])
        bv = np.zeros((H, 4))
        if j in (0, 2):                                  # two independent contact-velocity directions
            for h in range(H):
                if (h + 1) >= onset and alive[h]:
                    bv[h, j] = scale * 40.0              # authority magnitude ~ J_tip scale
        cols.append({"j": j, "both_alive": alive, "bv_col": bv, "dvrel_norm": np.linalg.norm(bv, axis=1)})
    return cols


def test_bv_assembly_and_rank_onset_sustain():
    cfg = HorizonConfig()
    cols = _synthetic_columns(onset=4, collapse=20, cfg=cfg)
    assert numeric_rank(_bv_at(cols, 3), cfg.rank_abs_tol, cfg.rank_rel_tol) == 0     # before onset
    assert numeric_rank(_bv_at(cols, 4), cfg.rank_abs_tol, cfg.rank_rel_tol) == 2     # at onset
    assert np.all(np.isnan(_bv_at(cols, 25)))                                          # after collapse (all NaN)
    rs = {e: np.asarray([numeric_rank(_bv_at(cols, h), cfg.rank_abs_tol, cfg.rank_rel_tol)
                         for h in range(1, cfg.h_max + 1)]) for e in cfg.bv.eps_scales}
    eps = list(cfg.bv.eps_scales)
    assert _rank2_onset(rs, eps, cfg) == 4
    assert _sustains(rs, eps, 4, cfg) is True
    assert _rank2_before_collapse(rs, eps, h_collapse=20) is True


def test_repro_gap_matches_and_differs():
    cfg = HorizonConfig()
    a = _synthetic_columns(onset=4, collapse=20, cfg=cfg, scale=1.0)
    b_same = _synthetic_columns(onset=4, collapse=20, cfg=cfg, scale=1.0)
    b_diff = _synthetic_columns(onset=4, collapse=20, cfg=cfg, scale=3.0)
    assert _repro_gap(a, b_same, 6) == 0.0
    assert _repro_gap(a, b_diff, 6) > cfg.repro_rel_gap                              # 3x magnitude → large gap


# --------------------------------------------------------------------------------------------------------------------
# route decision tree (per-state and campaign)
# --------------------------------------------------------------------------------------------------------------------
def test_state_route_tree():
    # usable → C
    assert _state_route(True, 4, 20, True, 4, True, True)[0] == "ROUTE_C_LIFTED_HORIZON"
    # no authority before collapse → B
    assert _state_route(False, None, 8, False, None, False, False)[0] == "ROUTE_B_SLEW_ADMISSIBLE_DTAU"
    assert _state_route(False, 10, 8, False, None, False, False)[0] == "ROUTE_B_SLEW_ADMISSIBLE_DTAU"
    # rank-2 authority exists but unusable (appears within margin of collapse) → A
    assert _state_route(False, 6, 8, True, 6, True, False)[0] == "ROUTE_A_LOWER_DEBT_HANDOFF"
    # authority appears but never rank-2 → B (rank-deficient interface)
    assert _state_route(False, 6, 20, False, None, False, False)[0] == "ROUTE_B_SLEW_ADMISSIBLE_DTAU"


def test_campaign_route_requires_dev_all_and_heldout_any():
    def st(route, usable):
        return {"route": route, "usable_authority": usable}
    # dev both usable, one held-out usable → C
    res = {0: st("ROUTE_C_LIFTED_HORIZON", True), 1: st("ROUTE_C_LIFTED_HORIZON", True),
           2: st("ROUTE_C_LIFTED_HORIZON", True), 3: st("ROUTE_A_LOWER_DEBT_HANDOFF", False)}
    assert decide_campaign_route(res, [0, 1], [2, 3])["route"] == "ROUTE_C_LIFTED_HORIZON"
    # dev not all usable, a B present → B dominates
    res2 = {0: st("ROUTE_C_LIFTED_HORIZON", True), 1: st("ROUTE_B_SLEW_ADMISSIBLE_DTAU", False),
            2: st("ROUTE_C_LIFTED_HORIZON", True), 3: st("ROUTE_C_LIFTED_HORIZON", True)}
    assert decide_campaign_route(res2, [0, 1], [2, 3])["route"] == "ROUTE_B_SLEW_ADMISSIBLE_DTAU"
    # dev usable but NO held-out usable → cannot claim C; falls to A (conservative)
    res3 = {0: st("ROUTE_C_LIFTED_HORIZON", True), 1: st("ROUTE_C_LIFTED_HORIZON", True),
            2: st("ROUTE_A_LOWER_DEBT_HANDOFF", False), 3: st("ROUTE_A_LOWER_DEBT_HANDOFF", False)}
    assert decide_campaign_route(res3, [0, 1], [2, 3])["route"] == "ROUTE_A_LOWER_DEBT_HANDOFF"


# --------------------------------------------------------------------------------------------------------------------
# the mechanism: the torque-rate limiter is an integrator (Session-1 dead zone → Session-2 catch-up authority)
# --------------------------------------------------------------------------------------------------------------------
def _stack():
    return V3Stack(qdot_soft=2.0, qdot_hard=3.0, armature=0.1, damping=0.5, friction=0.0,
                   kp=120.0, kv=8.0, tau_rate=30.0, control_dt=0.01)


def test_rate_limiter_is_integrator_dead_zone_then_authority():
    """Held servo target with a large PD debt (raw demand WITHIN the actuator range): the ±ε applied torque is
    IDENTICAL while the debt exceeds the per-step slew (Session-1 dead zone), then DIVERGES once prev_tau catches up
    to the raw demand, opening a steady-state authority band of 2·kp·ε (Session-2 authority)."""
    s = _stack()
    lo, hi = np.full(4, -4.0), np.full(4, 4.0)
    q, qd = np.zeros(4), np.zeros(4)
    q_des = np.array([0.02, 0.0, 0.0, 0.0])              # joint-0 raw demand kp*0.02 = 2.4 N·m — inside (-4,4)
    eps = 0.002
    raw_p = s.kp * (q_des[0] + eps)                      # 2.64 N·m
    prev_p = np.full(4, -3.0)
    prev_m = prev_p.copy()                              # start far below the target (debt 5.4 ≫ slew 0.3)
    diverged_at, a_p, a_m = None, None, None
    for h in range(1, 60):
        a_p = pd_governed_torque(q, qd, q_des + np.array([eps, 0, 0, 0]), s, prev_p, lo, hi)
        a_m = pd_governed_torque(q, qd, q_des - np.array([eps, 0, 0, 0]), s, prev_m, lo, hi)
        if diverged_at is None and float(np.max(np.abs(a_p - a_m))) > 1e-9:
            diverged_at = h
        prev_p, prev_m = a_p, a_m
    assert diverged_at is not None and diverged_at > 1        # NOT a one-step response — the Session-1 result
    assert abs(float(a_p[0]) - raw_p) < 1e-6                  # +ε branch converged to its raw demand (below ceiling)
    assert abs(float(a_p[0] - a_m[0]) - 2.0 * s.kp * eps) < 1e-6   # steady-state authority band = 2·kp·ε


def test_pd_governed_one_step_dead_zone_matches_session1():
    """The Session-1 primitive fact: at a large debt, one-step ±ε applied torque is bit-identical (B_v ≡ 0 root)."""
    s = _stack()
    lo, hi = np.full(4, -4.0), np.full(4, 4.0)
    q, qd = np.zeros(4), np.zeros(4)
    q_des = np.array([0.30, 0.0, 0.0, 0.0])
    prev = np.full(4, -3.0)
    eps = 0.002
    a_p = pd_governed_torque(q, qd, q_des + np.array([eps, 0, 0, 0]), s, prev, lo, hi)
    a_m = pd_governed_torque(q, qd, q_des - np.array([eps, 0, 0, 0]), s, prev, lo, hi)
    assert float(np.max(np.abs(a_p - a_m))) == 0.0


def test_horizons_frozen():
    assert HORIZONS == (1, 2, 4, 8, 12, 16, 24, 32, 40)
