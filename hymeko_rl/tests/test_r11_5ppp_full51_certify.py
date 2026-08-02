"""Tests for the R11.5+++ full-51 certification row-building (the coverage_gate itself is tested in test_r11_5_full_coverage)."""
import types

import pytest

import hymeko_rl.experiments.r11_5ppp_full51_certify as C
from hymeko_rl.coin_delivery.delivery_teacher.deliverability_ranking import GraspDelivery


def _gd(seed: int, dtz: float, k6: bool) -> GraspDelivery:
    return GraspDelivery(seed=seed, certified=True, safe=True, bilateral_dwell=6, kinetic=True, delivered_dtz_mm=dtz,
                         gap_closed=0.6, contact_delay=1, k6=k6)


def _dr(dtz: float, k6: bool, *, safe: bool = True, energy_complete: bool = True) -> object:
    return types.SimpleNamespace(k6=k6, min_dtz_mm=dtz, safe=safe,
                                 energy=types.SimpleNamespace(is_complete=lambda: energy_complete))


def _patch_scen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "build_bank_scenarios", lambda: [types.SimpleNamespace(scenario_id="bank_cX", coin_xy=(0.0, 0.0))])


def test_certified_row_selects_ranked_grasp_and_emits_gate_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scen(monkeypatch)
    # bank: seed 0 = 40mm no-K6 (current-first), seed 3 = 10mm K6 (deliverability-ranked pick)
    bank = [(_gd(0, 40.0, False), _dr(40.0, False), {"seed": 0}), (_gd(3, 10.0, True), _dr(10.0, True), {"seed": 3})]
    monkeypatch.setattr(C, "_bank", lambda *a, **k: bank)
    r = C.run_scenario(None, None, None, None, "bank_cX", "test", 40, 11)
    assert r["certified"] and r["teacher_k6"] and r["recovered"]                 # ranked grasp (seed 3) delivers K6
    assert r["selected_seed"] == 3 and r["teacher_dtz_mm"] == 10.0
    assert r["energy_complete"] and r["teacher_safe"] and r["teacher_only"]      # coverage_gate-compatible fields


def test_no_grasp_row_is_uncertified_not_recovered(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_scen(monkeypatch)
    monkeypatch.setattr(C, "_bank", lambda *a, **k: [])
    r = C.run_scenario(None, None, None, None, "bank_cX", "dev", 40, 11)
    assert r["certified"] is False and r["recovered"] is False and r["teacher_k6"] is False
