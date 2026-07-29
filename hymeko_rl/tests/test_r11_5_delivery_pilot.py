"""Tests for the R11.5 delivery-teacher pilot's pure config + gate logic."""
from hymeko_rl.coin_delivery.delivery_teacher.solver import FULL_SEARCH, full_transport_spec
from hymeko_rl.experiments.r11_5_delivery_teacher_pilot import _pilot_passed, gate


def test_full_transport_spec_opens_all_dims_and_extends_horizon() -> None:
    s = full_transport_spec()
    assert s.search_idx == FULL_SEARCH and len(s.search_idx) == 6      # push (squeeze/forward/balance) is searched
    assert s.frozen == {}
    assert s.horizon == 90                                             # extended from the shelved Phase-A default (36)
    assert len(s.lo) == len(s.hi) == len(s.init_std) == 6
    assert all(lo < hi for lo, hi in zip(s.lo, s.hi))
    assert full_transport_spec(horizon=120).horizon == 120


def test_pilot_passed_predicate() -> None:
    ok = {"dev": 1, "test": 1}
    assert _pilot_passed(6, ok, safety_ok=True, energy_ok=True) is True
    assert _pilot_passed(5, ok, safety_ok=True, energy_ok=True) is False        # <6 recovered
    assert _pilot_passed(8, {"dev": 0, "test": 2}, safety_ok=True, energy_ok=True) is False   # no dev
    assert _pilot_passed(8, {"dev": 2, "test": 0}, safety_ok=True, energy_ok=True) is False   # no test
    assert _pilot_passed(6, ok, safety_ok=False, energy_ok=True) is False       # safety regression
    assert _pilot_passed(6, ok, safety_ok=True, energy_ok=False) is False       # incomplete energy


def _row(split: str, recovered: bool, safe: bool = True) -> dict:
    return {"certified": True, "split": split, "recovered": recovered, "teacher_safe": safe, "energy_complete": True}


def test_gate_pass_verdict() -> None:
    rows = ([_row("train", True) for _ in range(4)] + [_row("dev", True), _row("dev", False)]
            + [_row("test", True), _row("test", False)] + [_row("train", False) for _ in range(4)])
    g = gate(rows)
    assert g["recovered"] == 6 and g["recovered_by_split"] == {"train": 4, "dev": 1, "test": 1}
    assert g["verdict"] == "R11_5_TARGET_CONDITIONED_DELIVERY_TEACHER_PILOT_PASS"


def test_gate_insufficient_verdict_when_no_test_recovery() -> None:
    rows = [_row("train", True) for _ in range(6)] + [_row("dev", True)] + [_row("test", False) for _ in range(5)]
    g = gate(rows)
    assert g["verdict"] == "R11_5_PIPELINE_PASS_DELIVERY_TEACHER_RECOVERY_INSUFFICIENT"


def test_gate_ignores_uncertified_capture_fails() -> None:
    rows = [_row("train", True) for _ in range(6)] + [_row("dev", True), _row("test", True)]
    rows.append({"certified": False, "split": "test", "recovered": False})
    g = gate(rows)
    assert g["n"] == 9 and g["certified"] == 8 and g["recovered"] == 8
