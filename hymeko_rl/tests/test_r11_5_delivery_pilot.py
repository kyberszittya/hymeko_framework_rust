"""Tests for the R11.5 delivery-teacher pilot's pure config + gate logic."""
import numpy as np

from hymeko_rl.coin_delivery.delivery_teacher.solver import (
    FULL_SEARCH,
    _settle_spec,
    full_transport_settle_spec,
    full_transport_spec,
)
from hymeko_rl.experiments.r11_5_delivery_teacher_pilot import _pilot_passed, gate


def test_full_transport_spec_opens_all_dims_and_extends_horizon() -> None:
    s = full_transport_spec()
    assert s.search_idx == FULL_SEARCH and len(s.search_idx) == 6      # push (squeeze/forward/balance) is searched
    assert s.frozen == {}
    assert s.horizon == 90                                             # extended from the shelved Phase-A default (36)
    assert len(s.lo) == len(s.hi) == len(s.init_std) == 6
    assert all(lo < hi for lo, hi in zip(s.lo, s.hi))
    assert full_transport_spec(horizon=120).horizon == 120


def test_full_transport_settle_spec_adds_only_the_settle_dim() -> None:
    s = full_transport_settle_spec()
    assert len(s.search_idx) == 7 and s.theta_dim == 7 and s.hi[6] == 4.0 and s.lo[6] == 0.0


def test_settle_spec_freezes_transport_searches_only_settle() -> None:
    """The two-stage stage-2 spec: transport θ[0:6] FROZEN at the given values, search ONLY settle_gain (θ[6])."""
    sp = _settle_spec((0.1, 0.2, 0.0, 10.0, 15.0, 1.5), horizon=90, settle_hi=4.0)
    assert sp.search_idx == (6,) and sp.theta_dim == 7 and sp.horizon == 90
    assert set(sp.frozen.keys()) == {0, 1, 2, 3, 4, 5} and sp.frozen[1] == 0.2 and sp.frozen[4] == 15.0
    assert sp.lo == (0.0,) and sp.hi == (4.0,)
    theta = sp.assemble(np.array([2.5]))                               # settle=2.5 on the frozen transport
    assert len(theta) == 7 and theta[6] == 2.5 and theta[1] == 0.2 and theta[4] == 15.0
    assert theta[6] == 0.0 or sp.assemble(np.array([0.0]))[6] == 0.0   # settle_gain=0 reproduces the transport


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
