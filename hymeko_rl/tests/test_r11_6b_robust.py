"""Tests for R11.6B robust reward / certificate / rollout / wide-basin loader / gate (rollout monkeypatched)."""
import json

import numpy as np
import pytest

from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO
from hymeko_rl.coin_delivery.theta_option import robust_delivery as RD
from hymeko_rl.coin_delivery.theta_option.delivery_theta_env import DeliveryReward
from hymeko_rl.coin_delivery.theta_option.robust_delivery import (
    RobustDeliveryReward,
    RobustRewardConfig,
    WideBasinDeliveryCertificate,
    perturb_theta,
    robust_rollout,
)
from hymeko_rl.experiments.r11_6b_robust_rl import _6b_verdict, gate_6b, wide_basin_ids


def _m(*, k6: bool = True, safe: bool = True, dtz: float = 5.0) -> dict:
    return {"peak_qdot": 1.0 if safe else 5.0, "peak_coin_speed": 0.5 if safe else 2.0, "forward": 0.08,
            "gap_closed": 0.9, "k6_max_dwell": 6 if k6 else 0, "k6_delivered": k6, "terminal_coin_speed": 0.05,
            "cross": 0.002, "lost_before_release": 0, "release_step": 30, "dtz_end": dtz / 1000.0}


def test_perturb_clips_to_box() -> None:
    out = perturb_theta(THETA_HI.copy(), 0.5, np.random.default_rng(0))
    assert np.all(out <= THETA_HI + 1e-9) and np.all(out >= THETA_LO - 1e-9)


def test_certificate_is_robust() -> None:
    assert WideBasinDeliveryCertificate(True, 0.01, 0.80, 6.0, True).is_robust(0.75)
    assert not WideBasinDeliveryCertificate(True, 0.01, 0.50, 6.0, True).is_robust(0.75)   # survival too low
    assert not WideBasinDeliveryCertificate(False, 0.01, 1.0, 6.0, True).is_robust(0.75)   # failed nominal never robust


def _patch(monkeypatch: pytest.MonkeyPatch, seq: list) -> None:
    it = iter(seq)
    monkeypatch.setattr(RD, "rollout_primitive", lambda snap, theta, cfg: next(it))
    monkeypatch.setattr(RD, "delivery_success", lambda m, cfg: bool(m["k6_delivered"]))


def test_robust_rollout_survival_and_cvar(monkeypatch: pytest.MonkeyPatch) -> None:
    # nominal K6, then 4 perturbations: 3 K6 (dtz 6/7/8) + 1 miss (40mm) -> survival 0.75, CVaR worst-2 = mean(40,8)=24.
    _patch(monkeypatch, [_m(dtz=5), _m(dtz=6), _m(dtz=7), _m(k6=False, dtz=40), _m(dtz=8)])
    theta = THETA_LO + 0.4 * (THETA_HI - THETA_LO)
    _m_nom, cert = robust_rollout(None, theta, RobustRewardConfig(k=4, sigma=0.01), np.random.default_rng(0))
    assert cert.nominal_k6 and cert.survival_rate == 0.75 and cert.worst_dtz_mm == 24.0 and cert.safe


def test_robust_reward_gates_survival_on_nominal_k6(monkeypatch: pytest.MonkeyPatch) -> None:
    reward = RobustDeliveryReward(DeliveryReward(), RobustRewardConfig(k=4, sigma=0.01))
    # wide: nominal K6 + all perturbations K6 -> big survival bonus.
    _patch(monkeypatch, [_m(dtz=5)] * 5)
    m_nom, cert = robust_rollout(None, THETA_LO.copy(), RobustRewardConfig(k=4), np.random.default_rng(0))
    r_wide = reward(m_nom, cert)
    # narrow: nominal K6 but 0 perturbations survive -> no survival bonus, more tail penalty.
    _patch(monkeypatch, [_m(dtz=5), _m(k6=False, dtz=40), _m(k6=False, dtz=45), _m(k6=False, dtz=50), _m(k6=False, dtz=55)])
    m_nom2, cert2 = robust_rollout(None, THETA_LO.copy(), RobustRewardConfig(k=4), np.random.default_rng(0))
    r_narrow = reward(m_nom2, cert2)
    assert r_wide > r_narrow                                                  # robust reward prefers the wide basin
    # failed nominal + lucky perturbations -> survival credit GATED off.
    _patch(monkeypatch, [_m(k6=False, dtz=40)] + [_m(dtz=5)] * 4)
    m_nom3, cert3 = robust_rollout(None, THETA_LO.copy(), RobustRewardConfig(k=4), np.random.default_rng(0))
    assert not cert3.nominal_k6
    assert reward(m_nom3, cert3) < reward(m_nom, cert)                        # a lucky perturbation cannot rescue it


def test_wide_basin_ids(tmp_path: object) -> None:
    d = tmp_path / "basin"                                                    # type: ignore[operator]
    d.mkdir()
    (d / "basin_000.jsonl").write_text(
        json.dumps({"scenario_id": "wide", "survival": {"0.01": 0.83}}) + "\n"
        + json.dumps({"scenario_id": "narrow", "survival": {"0.01": 0.17}}) + "\n"
        + json.dumps({"scenario_id": "bad", "error": "x"}) + "\n", encoding="utf-8")
    assert wide_basin_ids(d, "0.01", 0.5) == {"wide"}


def _seed(train: float, dev: float, robust: float, safe: float = 1.0) -> dict:
    return {"train": {"k6": train, "safe": safe}, "dev": {"k6": dev, "safe": safe}, "dev_robust": {"robust": robust}}


def test_gate_6b_pass() -> None:
    a2 = [_seed(0.82, 0.57, 0.57)] * 3          # dev 0.57 vs BC 0.29 -> +2 of 7; robust 0.57 >= 0.5; train held
    g = gate_6b(a2, {"k6": 0.286}, 0.82, 0.75, 7)
    assert g["verdict"] == "R11_6B_ROBUST_REWARD_GENERALIZATION_PASS" and g["dev_gain_scenarios"] >= 2


def test_gate_6b_stable_no_gain() -> None:
    a2 = [_seed(0.82, 0.286, 0.30)] * 3         # train held, dev flat -> stable, no gain
    assert gate_6b(a2, {"k6": 0.286}, 0.82, 0.75, 7)["verdict"] == "R11_6B_ROBUST_OBJECTIVE_STABLE_BUT_NO_GENERALIZATION_GAIN"


def test_gate_6b_insufficient_on_collapse() -> None:
    a2 = [_seed(0.30, 0.40, 0.20)] * 3          # train collapsed below warm-start 0.82 -> insufficient
    assert gate_6b(a2, {"k6": 0.286}, 0.82, 0.75, 7)["verdict"] == "R11_6B_LOCAL_ROBUSTNESS_REWARD_INSUFFICIENT"


def test_6b_verdict_needs_seed_majority() -> None:
    # dev gain scenarios ok + robust ok but only 1/3 seeds gain -> not a pass.
    assert _6b_verdict(True, True, 0.82, 3, 0.6, 1, 3) == "R11_6B_ROBUST_OBJECTIVE_STABLE_BUT_NO_GENERALIZATION_GAIN"
