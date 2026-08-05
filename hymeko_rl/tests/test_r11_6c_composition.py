"""Tests for R11.6C exact-zero composition: delivery classification, certificate validity, drift, and the gate."""
import numpy as np
import pytest

from hymeko_rl.coin_delivery import exact_zero_composition as EZ
from hymeko_rl.coin_delivery.exact_zero_composition import (
    SUCCESS,
    CompositionOutcomeClass,
    ExactZeroCoinDeliveryCertificate,
    _classify_delivery,
    _delivery_signals,
)
from hymeko_rl.experiments.r11_6c_exact_zero_composition import gate


def _metrics(*, k6: bool, touched: bool, gap: float, qdot: float = 1.0, coin_speed: float = 0.5) -> dict:
    return {"k6_delivered": k6, "peak_qdot": qdot, "peak_coin_speed": coin_speed, "touched": touched,
            "gap_closed": gap, "dtz_end": 0.002, "contact_lost_steps": 80, "forward_at_release": 0.0}


def test_valid_delivery_requires_touched_and_gap_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(EZ, "delivery_success", lambda m, cfg: bool(m["k6_delivered"]))
    def run(gap: float, touched: bool = True):
        monkeypatch.setattr(EZ, "rollout_primitive",
                            lambda snap, th, cfg: _metrics(k6=True, touched=touched, gap=gap))
        return _delivery_signals(None, np.zeros(6))
    assert run(0.975).valid_delivery and run(0.5).valid_delivery                # >= 0.5 gap = real transport
    assert not run(0.4).valid_delivery                                          # closed < half the gap = not a transport
    assert not run(0.975, touched=False).valid_delivery                        # never contacted the coin


def test_classify_success_nudge_safety_and_support() -> None:
    C = CompositionOutcomeClass
    assert _classify_delivery(k6=True, valid=True, safe=True, support_dist=0.5, radius=2.0) is SUCCESS
    assert _classify_delivery(k6=True, valid=True, safe=False, support_dist=0.5, radius=2.0) is C.SAFETY_FAILURE
    assert _classify_delivery(k6=True, valid=False, safe=True, support_dist=0.5, radius=2.0) is C.NUDGE_NOT_VALID_DELIVERY
    # failed delivery: split on the table radius
    assert _classify_delivery(False, False, True, support_dist=9.0, radius=2.0) is C.RETRIEVAL_OUT_OF_SUPPORT
    assert _classify_delivery(False, False, True, support_dist=0.5, radius=2.0) is C.DELIVERY_FAILURE_IN_SUPPORT


def test_certificate_is_valid_only_when_all_clauses_hold() -> None:
    ok = ExactZeroCoinDeliveryCertificate("s", 0, True, True, True, True, True, True, True, True, 0.4, 8.0, (0.1,) * 6)
    assert ok.is_valid()
    nudge = ExactZeroCoinDeliveryCertificate("s", 0, True, True, True, True, True, True, False, True, 0.4, 8.0, (0.1,) * 6)
    assert not nudge.is_valid()                                  # a nudge (invalid delivery mode) never certifies


def _rec(group: str, klass: CompositionOutcomeClass, *, k6: bool, cert: bool) -> dict:
    r = {"outcome_class": klass.value, "k6": k6, "valid_delivery": klass is SUCCESS, "safe": klass is not
         CompositionOutcomeClass.SAFETY_FAILURE, "certificate": ({"x": 1} if cert else None)}
    return {"group": group, "radius": 2.0, "primary": dict(r), "control": dict(r)}


def test_gate_pass_when_train_half_dev_positive_zero_safety() -> None:
    rows = [_rec("train-like", SUCCESS, k6=True, cert=True) for _ in range(3)] \
        + [_rec("train-like", CompositionOutcomeClass.DELIVERY_FAILURE_IN_SUPPORT, k6=False, cert=False) for _ in range(3)] \
        + [_rec("dev", SUCCESS, k6=True, cert=True)] \
        + [_rec("dev", CompositionOutcomeClass.RETRIEVAL_OUT_OF_SUPPORT, k6=False, cert=False) for _ in range(3)]
    g = gate(rows)
    assert g["verdict"] == "R11_6C_EXACT_ZERO_RETRIEVAL_COMPOSITION_PASS"
    assert g["primary"]["success_rate"]["train-like"] == 0.5 and g["primary"]["success_rate"]["dev"] == 0.25


def test_gate_fails_on_safety_regression_even_if_rates_pass() -> None:
    rows = [_rec("train-like", SUCCESS, k6=True, cert=True) for _ in range(5)] \
        + [_rec("train-like", CompositionOutcomeClass.SAFETY_FAILURE, k6=True, cert=False)] \
        + [_rec("dev", SUCCESS, k6=True, cert=True)]
    g = gate(rows)
    assert g["primary"]["n_safety_failure"] == 1
    assert g["verdict"] == "R11_6C_COMPOSITION_INSUFFICIENT"     # any safety regression blocks the pass


def test_gate_fails_without_a_certificate() -> None:
    rows = [_rec("train-like", CompositionOutcomeClass.DELIVERY_FAILURE_IN_SUPPORT, k6=False, cert=False)
            for _ in range(4)] + [_rec("dev", CompositionOutcomeClass.RETRIEVAL_OUT_OF_SUPPORT, k6=False, cert=False)]
    g = gate(rows)
    assert g["primary"]["n_certificates"] == 0 and g["verdict"] == "R11_6C_COMPOSITION_INSUFFICIENT"
