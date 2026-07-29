"""Tests for the pure classification/aggregation logic of the R11.4A re-measure runner."""
from hymeko_rl.experiments.r11_4a_bank_regenerate import classify, summarize


def test_classify_partitions_the_five_r11_4a0_classes() -> None:
    assert classify(certified=False, reaches_kinetic=False, k6=False, min_dtz_mm=45.0) == "CAPTURE_FAIL"
    assert classify(certified=True, reaches_kinetic=True, k6=True, min_dtz_mm=8.0) == "K6_WITH_VALID_DELIVERY_MODE"
    assert classify(certified=True, reaches_kinetic=False, k6=True, min_dtz_mm=18.0) == "K6_WITHOUT_DELIVERY_MODE_TRANSITION"
    assert classify(certified=True, reaches_kinetic=True, k6=False, min_dtz_mm=15.0) == "SETTLE_FAILURE_AFTER_VALID_GRASP"
    assert classify(certified=True, reaches_kinetic=True, k6=False, min_dtz_mm=45.0) == "DELIVERY_FAILURE_AFTER_VALID_GRASP"


def test_classify_certification_gates_a_nudge_k6() -> None:
    # an ungrasped nudge that trips K6 is NOT a valid delivery: certification comes first.
    assert classify(certified=False, reaches_kinetic=False, k6=True, min_dtz_mm=15.0) == "CAPTURE_FAIL"


def test_summarize_rates_and_valid_k6_requires_certification() -> None:
    rows = [
        {"certified": True, "kinetic": True, "deliver_k6": True, "cls": "K6_WITH_VALID_DELIVERY_MODE"},
        {"certified": True, "kinetic": True, "deliver_k6": False, "cls": "DELIVERY_FAILURE_AFTER_VALID_GRASP"},
        {"certified": False, "kinetic": False, "deliver_k6": True, "cls": "CAPTURE_FAIL"},   # nudge-K6, uncertified
        {"certified": False, "kinetic": False, "deliver_k6": False, "cls": "CAPTURE_FAIL"},
    ]
    s = summarize(rows)
    assert s["attempts"] == 4
    assert s["certified_grasp_rate"] == 0.5
    assert s["kinetic_rate"] == 0.5
    assert s["valid_k6_rate"] == 0.25          # only the certified grasped-K6 counts, not the uncertified nudge-K6
    assert s["class_counts"]["CAPTURE_FAIL"] == 2


def test_summarize_empty() -> None:
    s = summarize([])
    assert s["attempts"] == 0 and s["certified_grasp_rate"] == 0.0 and s["valid_k6_rate"] == 0.0
