"""Tests for the opt-in grasp-aware capture-candidate ranking (R11.4A).

The candidate-population audit proved a CAPTURE_CANDIDATE_RANKING_CONTRACT_BUG: a held bilateral grasp (contacts=2,
dwell>=target, min_dtz 52.93) was generated but the grasp-agnostic cost picked an ungrasped nudge (min_dtz 45.08). These
tests pin the corrected class-based lexicographic ranking and prove the default (obj=None) path stays bit-exact.
"""
import math

import numpy as np
import pytest

import hymeko_rl.coin_delivery.theta_option.moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option.moving_precapture import (
    BILATERAL_TRANSIENT,
    CaptureOutcome,
    GRASP_CERTIFIED,
    GraspObjective,
    NO_CONTACT,
    SAFETY_FAILURE,
    SINGLE_CONTACT_ONLY,
    _ContactTrace,
    _NO_DELAY,
    _grasp_class,
    _rank_key,
    _solution_found,
)


def _oc(**kw) -> CaptureOutcome:
    base = dict(snapshot=None, cos_dir=0.0, vel_scale=0.0, dtau=0.0, contacts=0)
    base.update(kw)
    return CaptureOutcome(**base)


def test_new_fields_default_to_no_contact_values() -> None:
    o = _oc()
    assert o.bilateral_dwell == 0
    assert math.isnan(o.first_contact_relvel) and math.isnan(o.second_contact_relvel)
    assert math.isnan(o.coin_disp_capture_mm)
    assert o.left_right_contact_delay == _NO_DELAY
    assert o.terminal_coin_speed == 0.0


def test_default_rank_key_is_bit_exact_old_cost() -> None:
    # obj=None must reproduce the original scalar: 1e3 if unsafe; else 0.0 on K6 else min_dtz.
    assert _rank_key(_oc(safe=False, min_dtz_mm=5.0)) == 1e3
    assert _rank_key(_oc(k6=True, min_dtz_mm=5.0)) == 0.0
    assert _rank_key(_oc(k6=False, min_dtz_mm=5.0)) == 5.0


def test_grasp_class_partition() -> None:
    assert _grasp_class(_oc(safe=False), 4) == SAFETY_FAILURE
    assert _grasp_class(_oc(contacts=2, bilateral_dwell=4), 4) == GRASP_CERTIFIED
    assert _grasp_class(_oc(contacts=2, bilateral_dwell=2, second_contact_relvel=0.1), 4) == BILATERAL_TRANSIENT
    assert _grasp_class(_oc(contacts=1, first_contact_relvel=0.1), 4) == SINGLE_CONTACT_ONLY
    assert _grasp_class(_oc(), 4) == NO_CONTACT


def test_ranking_bug_is_fixed_grasp_beats_nudge_despite_worse_dtz() -> None:
    """The exact seed-0 situation: a held grasp at 52.93 mm vs an ungrasped nudge at 45.08 mm."""
    obj = GraspObjective()
    grasp = _oc(contacts=2, bilateral_dwell=4, min_dtz_mm=52.93, left_right_contact_delay=2, terminal_coin_speed=0.1)
    nudge = _oc(contacts=0, bilateral_dwell=0, min_dtz_mm=45.08, first_contact_relvel=0.3)
    assert _rank_key(grasp, obj) < _rank_key(nudge, obj)      # grasp-aware picks the real grasp
    assert _rank_key(nudge, None) < _rank_key(grasp, None)    # the OLD score preferred the nudge (the bug)


def test_nudge_k6_does_not_outrank_a_real_grasp() -> None:
    obj = GraspObjective()
    grasp_no_k6 = _oc(contacts=2, bilateral_dwell=5, min_dtz_mm=30.0)
    nudge_k6 = _oc(contacts=0, k6=True, min_dtz_mm=15.0, first_contact_relvel=0.3)
    assert _rank_key(grasp_no_k6, obj) < _rank_key(nudge_k6, obj)


def test_within_class_delivery_beats_more_dwell() -> None:
    """A/B lesson: among held grasps, the DELIVERABLE one (lower min_dtz) must win, even with less dwell — ranking by
    dwell instead picked stable-but-undeliverable grasps and regressed valid K6 to zero."""
    obj = GraspObjective()
    delivers = _oc(contacts=2, bilateral_dwell=5, min_dtz_mm=10.0, terminal_coin_speed=0.05)
    just_holds = _oc(contacts=2, bilateral_dwell=8, min_dtz_mm=20.0, terminal_coin_speed=0.05)
    assert _rank_key(delivers, obj) < _rank_key(just_holds, obj)


def test_dwell_breaks_ties_at_equal_delivery() -> None:
    obj = GraspObjective()
    more_dwell = _oc(contacts=2, bilateral_dwell=8, min_dtz_mm=12.0, terminal_coin_speed=0.05)
    less_dwell = _oc(contacts=2, bilateral_dwell=5, min_dtz_mm=12.0, terminal_coin_speed=0.05)
    assert _rank_key(more_dwell, obj) < _rank_key(less_dwell, obj)


def test_within_grasp_class_prefers_k6_via_lower_dtz() -> None:
    obj = GraspObjective()
    grasp_k6 = _oc(contacts=2, bilateral_dwell=6, k6=True, min_dtz_mm=8.0, terminal_coin_speed=0.05)
    grasp_far = _oc(contacts=2, bilateral_dwell=6, k6=False, min_dtz_mm=40.0, terminal_coin_speed=0.05)
    assert _rank_key(grasp_k6, obj) < _rank_key(grasp_far, obj)


def test_solution_found_gate_rejects_nudge_k6_when_grasp_aware() -> None:
    obj = GraspObjective()
    nudge_k6 = (0.0, None, _oc(contacts=0, k6=True, min_dtz_mm=15.0, first_contact_relvel=0.3))
    grasp_k6 = ((0, -5, 2, 0.0, 8.0), None, _oc(contacts=2, bilateral_dwell=5, k6=True, min_dtz_mm=8.0))
    assert _solution_found(nudge_k6, None) is True        # default: any K6 stops the search
    assert _solution_found(nudge_k6, obj) is False         # grasp-aware: a nudge-K6 must NOT stop it
    assert _solution_found(grasp_k6, obj) is True          # grasp-aware: a held grasped K6 stops it
    assert _solution_found(None, obj) is False


def test_contact_delay_property() -> None:
    assert _ContactTrace(coin0=np.zeros(2)).contact_delay == _NO_DELAY
    both = _ContactTrace(coin0=np.zeros(2), first_l_step=11, first_r_step=12)
    assert both.contact_delay == 1
    one = _ContactTrace(coin0=np.zeros(2), first_l_step=5, first_r_step=-1)
    assert one.contact_delay == _NO_DELAY


def test_contact_trace_observe_tracks_dwell_and_first_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive the read-only observer over a scripted contact sequence: L@2, R@3 -> delay 1, three held steps -> dwell 3."""
    seq = [(False, False), (False, False), (True, False), (True, True), (True, True), (True, True), (False, False)]
    coins = [np.array([0.0, 0.0]), np.array([0.001, 0.0]), np.array([0.003, 0.0]), np.array([0.006, 0.0]),
             np.array([0.008, 0.0]), np.array([0.010, 0.0]), np.array([0.011, 0.0])]
    state = {"i": 0}

    def fake_contacts(_rl: object) -> dict:
        cl, cr = seq[state["i"]]
        return {"left": object() if cl else None, "right": object() if cr else None}

    monkeypatch.setattr(mp, "primary_fingertip_contacts", fake_contacts)
    monkeypatch.setattr(mp, "_coin_xy", lambda _rl: coins[state["i"]])

    class _Metrics:
        disk_vel = np.array([0.12, 0.0, 0.0])

    class _Inner:
        _planar_metrics = _Metrics()

    class _RL:
        inner = _Inner()

    tr = _ContactTrace(coin0=coins[0])
    for i in range(len(seq)):
        state["i"] = i
        tr.observe(_RL(), i)
    assert tr.first_l_step == 2
    assert tr.first_r_step == 3
    assert tr.contact_delay == 1
    assert tr.max_dwell == 3
    assert tr.first_relvel == pytest.approx(0.12)     # coin speed at first any-contact
    assert tr.second_relvel == pytest.approx(0.12)    # coin speed at first bilateral contact
    assert tr.coin_disp_mm == pytest.approx(11.0)     # peak displacement 0.011 m -> 11 mm
