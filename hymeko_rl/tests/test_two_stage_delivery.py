"""Tests for the two-stage TARGET_RELATIVE_ALIGNMENT_PHASE delivery: the scenario-relative rotation, the align/transport
theta assembly, and the load-bearing contract that the two-stage NEVER regresses below single-stage."""
import types

import numpy as np
import pytest

import hymeko_rl.coin_delivery.delivery_teacher.solver as S
from hymeko_rl.coin_delivery.delivery_teacher.solver import (
    ALIGNED,
    ALIGNMENT_FAILURE,
    AlignSearchSpec,
    DeliveryResult,
    _align_theta,
    _transport_theta,
    solve_delivery_two_stage,
)
from hymeko_rl.coin_delivery.forward_displacement import rotate2d


def test_rotate2d_is_scenario_relative() -> None:
    assert np.allclose(rotate2d(np.array([1.0, 0.0]), 90), [0.0, 1.0], atol=1e-9)
    assert np.allclose(rotate2d(np.array([1.0, 0.0]), 180), [-1.0, 0.0], atol=1e-9)
    assert np.allclose(rotate2d(np.array([0.6, -0.8]), 0), [0.6, -0.8], atol=1e-9)   # 0 offset = the true direction


def test_align_theta_stays_in_push() -> None:
    """ramp = n_steps and release = n_steps+1 -> every align step is PUSH (no brake/release within alignment)."""
    th = _align_theta(0.12, 8)
    assert th[3] == 8.0 and th[4] == 9.0 and th[5] == 0.0 and th[0] == S._ALIGN_SQUEEZE


def test_transport_theta_freezes_squeeze_and_balance() -> None:
    x = np.array([30.0, 0.1, 8.0, 0.2, 12.0, 30.0, 1.5])
    th = _transport_theta(x)
    assert th[0] == S._TRANSPORT_SQUEEZE and th[2] == 0.0 and (th[1], th[3], th[4], th[5]) == (0.2, 12.0, 30.0, 1.5)


def _stub(**kw) -> DeliveryResult:
    base = dict(seed=0, theta=(), k6=False, safe=True, min_dtz_mm=50.0, measurements={}, energy=object())
    base.update(kw)
    return DeliveryResult(**base)


def _patch(monkeypatch: pytest.MonkeyPatch, *, single: DeliveryResult, cem, two_m: dict, two_k6: bool) -> None:
    monkeypatch.setattr(S, "solve_delivery", lambda *a, **k: single)
    monkeypatch.setattr(S, "_two_stage_cem", cem)
    monkeypatch.setattr(S, "_two_stage_rollout", lambda *a, **k: (two_m, {}, ALIGNED, (0.12, 0.1, 0.0, 10.0, 20.0, 1.0)))
    monkeypatch.setattr(S, "delivery_success", lambda m, cfg: two_k6)
    monkeypatch.setattr(S, "build_ledger", lambda probe: object())
    monkeypatch.setattr(S, "PhaseEnergyProbe", lambda **k: object())


_SNAP = types.SimpleNamespace(stack=types.SimpleNamespace(control_dt=0.01))


def test_two_stage_wins_when_better(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, single=_stub(k6=False, min_dtz_mm=50.0), cem=lambda *a: (np.zeros(7), ALIGNED),
           two_m={"dtz_end": 0.010, "peak_qdot": 1.0, "peak_coin_speed": 0.5}, two_k6=True)
    r = solve_delivery_two_stage(_SNAP, seed=0, spec=AlignSearchSpec())
    assert r.k6 and r.stage_used == "align_transport" and r.align_verdict == ALIGNED and r.min_dtz_mm == 10.0


def test_two_stage_never_regresses_below_single(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worse two-stage result (larger dtz, no K6) must return the single-stage result unchanged."""
    _patch(monkeypatch, single=_stub(k6=False, min_dtz_mm=40.0), cem=lambda *a: (np.zeros(7), ALIGNED),
           two_m={"dtz_end": 0.120, "peak_qdot": 1.0, "peak_coin_speed": 0.5}, two_k6=False)
    r = solve_delivery_two_stage(_SNAP, seed=0, spec=AlignSearchSpec())
    assert not r.k6 and r.stage_used == "single" and r.min_dtz_mm == 40.0    # single wins; two-stage discarded


def test_two_stage_falls_back_on_alignment_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """No candidate preserved the grasp (cem returns None) -> single-stage with ALIGNMENT_FAILURE recorded."""
    _patch(monkeypatch, single=_stub(k6=True, min_dtz_mm=12.0), cem=lambda *a: None,
           two_m={"dtz_end": 0.0, "peak_qdot": 1.0, "peak_coin_speed": 0.5}, two_k6=False)
    r = solve_delivery_two_stage(_SNAP, seed=0, spec=AlignSearchSpec())
    assert r.stage_used == "single" and r.align_verdict == ALIGNMENT_FAILURE and r.k6
