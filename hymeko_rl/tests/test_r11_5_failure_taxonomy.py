"""Tests for the canonical R11.5+ trajectory-based failure classifier + bounded-pilot selection."""
from hymeko_rl.experiments.r11_5_failure_taxonomy import (
    DeliveryTrace,
    FailureCategory,
    FailureRecord,
    cclass,
    classify,
    select_pilot,
    summarize,
)


def _cap(sid: str, subtype: str, split: str = "train") -> FailureRecord:
    return FailureRecord(sid, split, cclass(sid), FailureCategory.CAPTURE_SUPPORT_FAILURE, subtype, None)


def _res(sid: str, cat: FailureCategory, split: str, dtz: float) -> FailureRecord:
    return FailureRecord(sid, split, cclass(sid), cat, "", dtz)


def test_classify_canonical_paths() -> None:
    C = FailureCategory
    assert classify(DeliveryTrace(certified=False)) is C.CAPTURE_SUPPORT_FAILURE
    # certified but never moved -> handoff to kinetic failure
    assert classify(DeliveryTrace(True, coin_progress_mm=1.0, gap_closed_final=0.0, min_dtz_mm=90.0)) is C.HANDOFF_TO_KINETIC_FAILURE
    # grasp lost for most of the pre-release window -> contact loss (takes precedence over direction)
    assert classify(DeliveryTrace(True, 40.0, -0.5, 120.0, 0, 0.0, lost_before_release=6, release_step=8)) is C.CONTACT_LOSS_DURING_DELIVERY
    # driven net away from target -> directional bias
    assert classify(DeliveryTrace(True, 21.4, -0.71, 126.8, 0, 0.0, 0, 8)) is C.DELIVERY_DIRECTIONAL_BIAS
    # closest approach still outside the 20mm zone -> insufficient transport progress
    assert classify(DeliveryTrace(True, 61.4, 0.70, 22.4, 0, 0.0, 0, 8)) is C.INSUFFICIENT_TRANSPORT_PROGRESS
    # entered the zone (min<=20) but too fast to settle
    assert classify(DeliveryTrace(True, 80.0, 0.9, 15.0, 0, entry_speed=0.09, release_step=8)) is C.TARGET_ENTRY_SPEED_FAILURE
    # entered the zone, slow, but no held dwell
    assert classify(DeliveryTrace(True, 80.0, 0.9, 15.0, max_k6_dwell=2, entry_speed=0.0, release_step=8)) is C.ZONE_ENTRY_WITHOUT_DWELL


def test_trajectory_min_vs_final_distinguishes_zone_entry() -> None:
    """A coin that transiently entered the zone (min<=20) then drifted out is ZONE_ENTRY, not INSUFFICIENT — the final
    dtz alone (the only thing the ledger records) would misclassify it."""
    entered = DeliveryTrace(True, coin_progress_mm=90.0, gap_closed_final=0.6, min_dtz_mm=12.0, max_k6_dwell=3,
                            entry_speed=0.0, release_step=8)
    stalled = DeliveryTrace(True, coin_progress_mm=60.0, gap_closed_final=0.6, min_dtz_mm=35.0, max_k6_dwell=0,
                            entry_speed=0.0, release_step=8)
    assert classify(entered) is FailureCategory.ZONE_ENTRY_WITHOUT_DWELL
    assert classify(stalled) is FailureCategory.INSUFFICIENT_TRANSPORT_PROGRESS


def test_select_pilot_groups_and_heldout() -> None:
    C = FailureCategory
    caps = [_cap("bank_c1_+0.03_+0.02", "systematic_pp", "test"), _cap("bank_c2_+0.015_+0.025", "systematic_pp", "dev"),
            _cap("bank_c1_+0.01_+0.02", "stochastic_regen"), _cap("bank_c1_-0.01_+0.00", "stochastic_regen"),
            _cap("bank_c2_+0.025_+0.025", "systematic_pp", "test"), _cap("bank_c2_-0.015_+0.000", "stochastic_regen")]
    # the negative-x tail is CONTACT_LOSS (grasp squirts out when pushed off-axis) — the trajectory-truth class
    negx = [_res(s, C.CONTACT_LOSS_DURING_DELIVERY, "train", 150.0)
            for s in ("bank_c1_-0.01_+0.03", "bank_c1_-0.03_+0.02", "bank_c1_-0.03_+0.03", "bank_c2_-0.025_+0.015")]
    prog = [_res("bank_c3_r7_a+15", C.INSUFFICIENT_TRANSPORT_PROGRESS, "train", 22.4),
            _res("bank_c3_r9_a+15", C.INSUFFICIENT_TRANSPORT_PROGRESS, "test", 35.5),
            _res("bank_c3_r9_a-45", C.INSUFFICIENT_TRANSPORT_PROGRESS, "dev", 55.6),
            _res("bank_c2_-0.015_+0.015", C.INSUFFICIENT_TRANSPORT_PROGRESS, "train", 70.0)]
    pilot = select_pilot(caps + negx + prog)
    assert len(pilot) == 12
    cats = [r.category for r in pilot]
    assert cats.count(C.CAPTURE_SUPPORT_FAILURE) == 4 and cats.count(C.CONTACT_LOSS_DURING_DELIVERY) == 4
    prog_pick = [r for r in pilot if r.category is C.INSUFFICIENT_TRANSPORT_PROGRESS]
    assert len({r.split for r in prog_pick}) >= 2                          # transport fix validated off-train
    assert {r.subtype for r in pilot if r.category is C.CAPTURE_SUPPORT_FAILURE} == {"systematic_pp", "stochastic_regen"}
    assert "dev" in {r.split for r in pilot} and "test" in {r.split for r in pilot}


def test_summarize_partitions() -> None:
    recs = [_cap("bank_c1_+0.03_+0.02", "systematic_pp"),
            _res("bank_c1_-0.03_+0.02", FailureCategory.DELIVERY_DIRECTIONAL_BIAS, "train", 160.0)]
    s = summarize(recs)
    assert s["CAPTURE_SUPPORT_FAILURE"] == ["bank_c1_+0.03_+0.02"]
    assert s["DELIVERY_DIRECTIONAL_BIAS"] == ["bank_c1_-0.03_+0.02"]
