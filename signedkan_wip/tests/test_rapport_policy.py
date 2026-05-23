"""Tests for the rapport policy module."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRIAD_PATH = REPO_ROOT / "data" / "coalitions" / "triad_hri.hymeko"


def test_predicate_parses_simple_threshold():
    from signedkan_wip.src.rapport.policy import eval_condition
    assert eval_condition("sigma(triad) < -0.2", {"triad": -0.5}, {}) is True
    assert eval_condition("sigma(triad) < -0.2", {"triad":  0.2}, {}) is False


def test_predicate_parses_sustained():
    from signedkan_wip.src.rapport.policy import eval_condition
    hist = {"triad": deque([-0.1, -0.2, -0.3, -0.4, -0.5])}
    assert eval_condition("sustained(triad, 5)", {}, hist) is True
    # If the most recent k include a non-negative, sustained fails.
    hist2 = {"triad": deque([-0.1, -0.2, -0.3, 0.1, -0.5])}
    assert eval_condition("sustained(triad, 5)", {}, hist2) is False


def test_predicate_and_chain():
    from signedkan_wip.src.rapport.policy import eval_condition
    hist = {"triad": deque([-0.6] * 10)}
    cond = "sigma(triad) < -0.5 and sustained(triad, 5)"
    assert eval_condition(cond, {"triad": -0.7}, hist) is True
    # Fail on the threshold piece.
    assert eval_condition(cond, {"triad": -0.4}, hist) is False


def test_policy_engine_fires_repair_below_threshold():
    from signedkan_wip.src.rapport.coalition import load_coalition
    from signedkan_wip.src.rapport.policy import PolicyEngine
    coalition = load_coalition(TRIAD_PATH)
    eng = PolicyEngine(coalition, cooldown_frames=5)
    # Frame 0: σ = -0.3 → "repair" should fire (threshold -0.2).
    out = eng.step(0, {"triad": -0.3})
    assert "repair" in out.fired
    assert "signal_alignment" in out.actions
    # Frame 1: within cooldown, must NOT re-fire even though condition holds.
    out2 = eng.step(1, {"triad": -0.3})
    assert "repair" not in out2.fired


def test_policy_engine_sustained_mediate_fires():
    from signedkan_wip.src.rapport.coalition import load_coalition
    from signedkan_wip.src.rapport.policy import PolicyEngine
    coalition = load_coalition(TRIAD_PATH)
    eng = PolicyEngine(coalition, cooldown_frames=2)
    # Drive σ negative for 6 consecutive frames at -0.6 (below -0.5).
    # mediate requires `sigma < -0.5 and sustained(triad, 5)` → fires at frame 5.
    fired_any_mediate = False
    for t in range(7):
        out = eng.step(t, {"triad": -0.6})
        if "mediate" in out.fired:
            fired_any_mediate = True
    assert fired_any_mediate, "mediate policy never fired despite sustained negative σ"
