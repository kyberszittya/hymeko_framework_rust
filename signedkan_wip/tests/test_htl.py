"""Unit tests for the HTL pure-Python implementation.

Layers:
- Parser: well-formed input → expected AST; malformed → ParseError with message.
- Evaluator: robust-STL semantics for each operator + intervals.
- Online monitor: streaming events through HtlMonitor.
- Predicate registry: signal lookup, custom registrations.
"""

import math
import os
import sys
from typing import List

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO_ROOT, "signedkan_wip", "src")
for _p in (REPO_ROOT, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from htl import (  # noqa: E402
    And,
    CmpOp,
    Eventually,
    Globally,
    HtlMonitor,
    HypergraphEvent,
    Not,
    Or,
    ParseError,
    ScalarPred,
    parse,
    robustness,
    satisfied,
)
from htl.predicates import clear_registry, register  # noqa: E402


# ---------------- parser tests ----------------


def test_parser_simple_scalar_predicate():
    node = parse("val_auc > 0.85")
    assert isinstance(node, ScalarPred)
    assert node.name == "val_auc"
    assert node.op is CmpOp.GT
    assert node.value == pytest.approx(0.85)


def test_parser_globally_around_predicate():
    node = parse("G(val_auc > 0.85)")
    assert isinstance(node, Globally)
    assert node.t1 == 0.0
    assert math.isinf(node.t2) and node.t2 > 0
    assert isinstance(node.inner, ScalarPred)
    assert node.inner.name == "val_auc"


def test_parser_nested_g_f():
    node = parse("G(F(x > 0))")
    assert isinstance(node, Globally)
    assert isinstance(node.inner, Eventually)
    assert isinstance(node.inner.inner, ScalarPred)


def test_parser_bracketed_subscript_predicate():
    node = parse("alpha[c5] >= 0.3 AND val_auc > 0.85")
    assert isinstance(node, And)
    assert isinstance(node.left, ScalarPred)
    assert node.left.name == "alpha[c5]"
    assert node.left.op is CmpOp.GE


def test_parser_not_or_precedence():
    # AND binds tighter than OR; NOT binds tightest.
    node = parse("NOT x > 0 OR y < 1 AND z == 2")
    # Expected: Or(Not(x>0), And(y<1, z==2))
    assert isinstance(node, Or)
    assert isinstance(node.left, Not)
    assert isinstance(node.right, And)


def test_parser_interval_syntax():
    node = parse("G[0, 5](val_auc >= 0.9)")
    assert isinstance(node, Globally)
    assert node.t1 == 0.0
    assert node.t2 == 5.0


def test_parser_rejects_malformed():
    with pytest.raises(ParseError):
        parse("G val_auc > 0.85")  # missing parens
    with pytest.raises(ParseError):
        parse("val_auc >> 0.85")  # bad operator
    with pytest.raises(ParseError):
        parse("")  # empty
    with pytest.raises(ParseError):
        parse("G(val_auc > 0.85")  # unclosed paren


# ---------------- evaluator tests ----------------


def _evt(t: float, **signals: float) -> HypergraphEvent:
    return HypergraphEvent(t=t, scalar_signals=signals)


def test_scalar_pred_robustness_signed_margin():
    history = [_evt(0.0, val_auc=0.90)]
    rho_gt = robustness(parse("val_auc > 0.85"), history)
    assert rho_gt == pytest.approx(0.05)

    rho_lt = robustness(parse("val_auc < 0.85"), history)
    assert rho_lt == pytest.approx(-0.05)

    rho_eq = robustness(parse("val_auc == 0.90"), history)
    assert rho_eq == pytest.approx(0.0)
    rho_eq_off = robustness(parse("val_auc == 0.85"), history)
    assert rho_eq_off == pytest.approx(-0.05)


def test_not_negates_robustness():
    history = [_evt(0.0, x=1.0)]
    rho_pos = robustness(parse("x > 0"), history)
    rho_neg = robustness(parse("NOT x > 0"), history)
    assert rho_neg == pytest.approx(-rho_pos)


def test_and_or_min_max():
    history = [_evt(0.0, a=1.0, b=0.2)]
    rho_and = robustness(parse("a > 0 AND b > 0"), history)
    rho_or = robustness(parse("a > 0 OR b > 0"), history)
    # a>0 has margin 1.0; b>0 has margin 0.2
    assert rho_and == pytest.approx(0.2)
    assert rho_or == pytest.approx(1.0)


def test_globally_inf_over_history():
    history: List[HypergraphEvent] = [
        _evt(0.0, val_auc=0.90),
        _evt(1.0, val_auc=0.86),
        _evt(2.0, val_auc=0.95),
    ]
    rho = robustness(parse("G(val_auc > 0.85)"), history)
    # inf of margins 0.05, 0.01, 0.10 = 0.01
    assert rho == pytest.approx(0.01)
    assert satisfied(rho)

    # If one violates, robustness goes negative.
    history.append(_evt(3.0, val_auc=0.80))
    rho_after = robustness(parse("G(val_auc > 0.85)"), history)
    assert rho_after == pytest.approx(-0.05)
    assert not satisfied(rho_after)


def test_eventually_sup_over_history():
    history = [
        _evt(0.0, val_auc=0.70),
        _evt(1.0, val_auc=0.72),
        _evt(2.0, val_auc=0.88),
    ]
    rho = robustness(parse("F(val_auc > 0.85)"), history)
    # sup of margins -0.15, -0.13, 0.03 = 0.03
    assert rho == pytest.approx(0.03)
    assert satisfied(rho)


def test_boolean_satisfaction_tracks_sign():
    history = [_evt(0.0, x=0.5)]
    assert satisfied(robustness(parse("x > 0"), history))
    assert not satisfied(robustness(parse("x > 1"), history))


# ---------------- monitor tests ----------------


def test_monitor_streams_events_and_emits_robustness():
    mon = HtlMonitor("G(val_auc > 0.85)", horizon=64)
    aucs = [0.88, 0.86, 0.84, 0.90, 0.91]
    rhos = []
    for t, auc in enumerate(aucs):
        rho = mon.observe(_evt(float(t), val_auc=auc))
        rhos.append(rho)

    # Step 0: only 0.88 → +0.03
    assert rhos[0] == pytest.approx(0.03)
    # Step 1: min(0.03, 0.01) = 0.01
    assert rhos[1] == pytest.approx(0.01)
    # Step 2: 0.84 violates → -0.01
    assert rhos[2] == pytest.approx(-0.01)
    # Once violated, the G past observation stays negative.
    assert all(r <= 0.0 for r in rhos[2:])


def test_monitor_horizon_eviction():
    mon = HtlMonitor("G(val_auc > 0.85)", horizon=3)
    # Submit 5 events; only the latest 3 should be retained.
    aucs = [0.50, 0.50, 0.90, 0.91, 0.92]
    for t, auc in enumerate(aucs):
        mon.observe(_evt(float(t), val_auc=auc))
    assert len(mon.history) == 3
    # The retained window violates none → satisfied.
    assert mon.satisfied()
    assert mon.robustness() == pytest.approx(0.05)


def test_monitor_rejects_out_of_order_events():
    mon = HtlMonitor("G(val_auc > 0.85)")
    mon.observe(_evt(1.0, val_auc=0.90))
    with pytest.raises(ValueError):
        mon.observe(_evt(0.5, val_auc=0.90))


# ---------------- predicate registry tests ----------------


def test_registered_custom_predicate():
    clear_registry()
    try:

        @register("custom_metric")
        def _custom(event):  # noqa: D401
            return event.scalar_signals["base"] * 2.0

        mon = HtlMonitor("custom_metric > 1.0")
        rho = mon.observe(_evt(0.0, base=0.4))
        # custom_metric = 0.8 < 1.0 → rho = -0.2
        assert rho == pytest.approx(-0.2)

        rho2 = mon.observe(_evt(1.0, base=0.7))
        # last event: 1.4 > 1.0 → +0.4 (no temporal op, atomic)
        assert rho2 == pytest.approx(0.4)
    finally:
        clear_registry()


def test_unknown_signal_raises_keyerror():
    mon = HtlMonitor("missing_signal > 0")
    with pytest.raises(KeyError):
        mon.observe(_evt(0.0, other_signal=1.0))
