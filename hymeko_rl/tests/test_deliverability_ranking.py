"""Tests for the R11.5++ deliverability-ranked grasp selection (lexicographic, teacher-only oracle)."""
from hymeko_rl.coin_delivery.delivery_teacher.deliverability_ranking import (
    GraspDelivery,
    deliverability_key,
    is_material_dtz_improvement,
    select_deliverable_grasp,
)


def _g(**kw) -> GraspDelivery:
    base = dict(seed=0, certified=True, safe=True, bilateral_dwell=6, kinetic=True, delivered_dtz_mm=30.0,
                gap_closed=0.5, contact_delay=1, k6=False)
    base.update(kw)
    return GraspDelivery(**base)


def test_certified_dominates_dtz() -> None:
    """A certified grasp with a WORSE delivered dtz must still outrank a non-certified grasp with a better dtz."""
    certified_far = _g(certified=True, delivered_dtz_mm=40.0)
    nudge_close = _g(certified=False, delivered_dtz_mm=5.0)
    assert select_deliverable_grasp([nudge_close, certified_far]) is certified_far


def test_safety_is_top_tier() -> None:
    unsafe_perfect = _g(safe=False, delivered_dtz_mm=1.0, k6=True)
    safe_worse = _g(safe=True, delivered_dtz_mm=50.0)
    assert select_deliverable_grasp([unsafe_perfect, safe_worse]) is safe_worse


def test_delivered_dtz_decides_among_equal_class() -> None:
    """Same safety/certified/dwell/kinetic -> the lower delivered dtz wins (the deliverability oracle)."""
    a = _g(delivered_dtz_mm=45.0)
    b = _g(delivered_dtz_mm=18.0, k6=True)
    assert select_deliverable_grasp([a, b]) is b


def test_kinetic_entry_beats_no_kinetic_before_dtz() -> None:
    no_kinetic_close = _g(kinetic=False, delivered_dtz_mm=15.0)
    kinetic_far = _g(kinetic=True, delivered_dtz_mm=35.0)
    assert select_deliverable_grasp([no_kinetic_close, kinetic_far]) is kinetic_far


def test_dwell_is_a_stability_gate_not_a_maximizer() -> None:
    """Regression for the smoke bug: among adequately-stable grasps, higher raw dwell must NOT outrank a strictly more
    DELIVERABLE grasp. A grasp below the stability threshold does lose to a qualified one."""
    # the exact smoke case: dwell 7 / 18.6 mm K6 must beat dwell 8 / 24.2 mm no-K6 (dtz decides, not dwell)
    k6_grasp = _g(bilateral_dwell=7, delivered_dtz_mm=18.6, k6=True)
    higher_dwell_worse = _g(bilateral_dwell=8, delivered_dtz_mm=24.2, k6=False)
    assert select_deliverable_grasp([higher_dwell_worse, k6_grasp]) is k6_grasp
    # but an UNstable grasp (below the gate) loses to a stable one even with a better dtz
    unstable_close = _g(bilateral_dwell=1, delivered_dtz_mm=10.0)
    stable_far = _g(bilateral_dwell=6, delivered_dtz_mm=30.0)
    assert select_deliverable_grasp([unstable_close, stable_far]) is stable_far


def test_progress_then_seating_are_last_tiebreaks() -> None:
    a = _g(delivered_dtz_mm=25.0, gap_closed=0.3, contact_delay=5)
    b = _g(delivered_dtz_mm=25.0, gap_closed=0.6, contact_delay=9)   # more progress wins before seating delay
    c = _g(delivered_dtz_mm=25.0, gap_closed=0.6, contact_delay=2)   # equal progress -> lower seating delay wins
    assert select_deliverable_grasp([a, b, c]) is c
    assert deliverability_key(b) < deliverability_key(a)


def test_material_improvement_floor() -> None:
    assert is_material_dtz_improvement(45.0, 18.0) is True
    assert is_material_dtz_improvement(22.0, 20.0) is False           # 2 mm < 5 mm floor = not material
