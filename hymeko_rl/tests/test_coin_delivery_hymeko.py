"""Tests for the canonical coin-delivery HyMeKo task description (data/robotics/coin_delivery_v1.hymeko).

Structural checks over the description (a small text parser for phases / transitions / capabilities / monitor) plus a
CLI `hymeko validate` determinism check. These encode the binding task semantics so goal / phase-event / monitor-truth
/ termination / capability cannot be confused (COIN-DELIVERY-OVERNIGHT-2 PART VII-C).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_FILE = Path("data/robotics/coin_delivery_v1.hymeko")
_BIN = Path("target/debug/hymeko")


def _text() -> str:
    return _FILE.read_text()


def _phases(text: str) -> dict:
    """name -> {law, kind} for every fsm_phase node."""
    out = {}
    for m in re.finditer(r"@(\w+):\s*ph\.fsm_phase\s*\{([^}]*)\}", text):
        body = m.group(2)
        law = re.search(r'law\s+"([^"]+)"', body)
        kind = re.search(r'kind\s+"([^"]+)"', body)
        out[m.group(1)] = {"law": law.group(1) if law else None, "kind": kind.group(1) if kind else None}
    return out


def _transitions(text: str) -> list:
    """(src, dst, event) for every transition node with a (- src, + dst) arc + event label."""
    out = []
    for m in re.finditer(r"@tr_\w+:\s*c\.controller_spec\s*\{\s*\(-\s*(\w+),\s*\+\s*(\w+)\)\s*;\s*event\s+\"([^\"]+)\"", text):
        out.append((m.group(1), m.group(2), m.group(3)))
    return out


def _capabilities(text: str) -> dict:
    """name -> phase for every capability node (act.* / coord.*) carrying a `phase` attr."""
    out = {}
    for m in re.finditer(r"@(\w+):\s*(?:act|coord)\.\w+\s*\{[^}]*phase\s+\"(\w+)\"", text):
        out[m.group(1)] = m.group(2)
    return out


# ── the 10 required semantic tests ───────────────────────────────────────────────────────────────────────────────────
def test_1_handoff_is_not_delivery_success() -> None:
    ph = _phases(_text())
    assert ph["HANDOFF"]["kind"] == "phase_event"
    assert ph["DELIVERED"]["kind"] == "terminal_success"
    # no direct HANDOFF -> DELIVERED transition (delivery is not reachable directly from handoff)
    assert not any(s == "HANDOFF" and d == "DELIVERED" for s, d, _e in _transitions(_text()))


def test_2_handoff_is_not_task_termination() -> None:
    ph, trs = _phases(_text()), _transitions(_text())
    terminals = {n for n, v in ph.items() if v["kind"] in ("terminal_success", "terminal_failure")}
    outgoing = [d for s, d, _e in trs if s == "HANDOFF"]
    # HANDOFF has an outgoing transition to a NON-terminal phase => it does not terminate the task
    assert outgoing and any(d not in terminals for d in outgoing)


def test_3_transport_follows_handoff() -> None:
    trs = _transitions(_text())
    # transport is reachable from handoff: HANDOFF -> CLAMP_VERIFY -> TRANSPORT (COIN-TRANSPORT-1 inserted CLAMP_VERIFY)
    assert ("HANDOFF", "CLAMP_VERIFY", "handoff_logged") in trs
    assert any(s == "CLAMP_VERIFY" and d == "TRANSPORT" for s, d, _e in trs)


def test_4_zone_entry_is_delivery_success() -> None:
    assert 'delivery_success "disk_to_zone<=zone_half"' in _text()


def test_5_center_reach_is_disk_to_zone_le_002() -> None:
    t = _text()
    assert re.search(r"center_tol\s+0\.02", t)
    assert 'center_reached "disk_to_zone<=center_tol"' in t


def test_6_coin_and_zone_centres_are_distance_endpoints() -> None:
    t = _text()
    assert re.search(r"@coin:\s*el\.frame", t)
    assert re.search(r"@target_zone:\s*el\.frame", t)
    # the monitor distance is measured on disk_to_zone (coin centre to zone centre)
    assert "disk_to_zone" in t


def test_7_reward_and_monitor_truth_are_distinct() -> None:
    t = _text()
    assert re.search(r"@reward_shaping:\s*c\.controller_spec", t)
    assert re.search(r"@monitor_truth:\s*c\.controller_spec", t)
    # reward carries a potential term; monitor carries success predicates — different nodes
    assert 'potential "neg_disk_to_zone"' in t


def test_8_objective_hierarchy_delivery_above_acquisition() -> None:
    m = re.search(r"@objective_priority:\s*c\.controller_spec\s*\{\s*\(([^)]*)\)", _text())
    assert m is not None
    order = [x.strip().lstrip("+").strip() for x in m.group(1).split(",")]
    # SETTLE (centre) and TRANSPORT (delivery) must precede STABILIZE / ACQUIRE (acquisition)
    assert order.index("SETTLE") < order.index("ACQUIRE")
    assert order.index("TRANSPORT") < order.index("STABILIZE")


def test_9_acquisition_capability_only_in_acquisition_phases() -> None:
    caps = _capabilities(_text())
    assert caps["regrasp"] == "ACQUIRE"          # regrasp valid only in acquisition/recovery
    assert caps["align"] == "ACQUIRE"
    assert caps["carry"] == "TRANSPORT"          # carry is a transport capability, not acquisition
    assert caps["regrasp"] not in ("TRANSPORT", "SETTLE")


def test_10_description_parses_deterministically() -> None:
    if not _BIN.exists() or shutil.which(str(_BIN)) is None and not _BIN.is_file():
        pytest.skip("hymeko CLI binary not built")
    r1 = subprocess.run([str(_BIN), "validate", str(_FILE)], capture_output=True, text=True)
    r2 = subprocess.run([str(_BIN), "validate", str(_FILE)], capture_output=True, text=True)
    out1 = r1.stdout + r1.stderr
    assert r1.returncode == 0 and "is valid" in out1
    assert out1 == r2.stdout + r2.stderr          # deterministic


# ── FSM well-formedness (structural integrity) ───────────────────────────────────────────────────────────────────────
def test_fsm_all_transitions_reference_declared_phases() -> None:
    ph = set(_phases(_text()))
    for s, d, _e in _transitions(_text()):
        assert s in ph and d in ph


def test_fsm_terminals_have_no_outgoing_transitions() -> None:
    trs = _transitions(_text())
    for terminal in ("DELIVERED", "FAILED"):
        assert not any(s == terminal for s, _d, _e in trs)


def test_fsm_initial_phase_is_acquire() -> None:
    m = re.search(r"@coin_delivery_fsm:\s*c\.controller_spec\s*\{\s*\(([^)]*)\)", _text())
    first = m.group(1).split(",")[0].strip().lstrip("+").strip()
    assert first == "ACQUIRE"
