"""Tests for R11.6A delivery-RL env, reward, box mapping, and gate (rollout monkeypatched — no MuJoCo)."""
import numpy as np
import pytest

from hymeko_rl.coin_delivery.delivery_bc.models import THETA_HI, THETA_LO, Standardizer
from hymeko_rl.coin_delivery.theta_option import delivery_theta_env as E
from hymeko_rl.coin_delivery.theta_option.delivery_theta_env import (
    CoinDeliveryThetaOptionEnv,
    DeliveryReward,
    ScenarioHandoff,
    box_to_theta,
    theta_to_box,
)
from hymeko_rl.experiments.r11_6a_delivery_rl import gate


def _m(*, k6: bool = True, safe: bool = True, gap: float = 0.9) -> dict:
    return {"peak_qdot": 1.0 if safe else 5.0, "peak_coin_speed": 0.5 if safe else 2.0, "forward": 0.08,
            "gap_closed": gap, "k6_max_dwell": 6 if k6 else 0, "k6_delivered": k6, "terminal_coin_speed": 0.05,
            "cross": 0.002, "lost_before_release": 0, "release_step": 30, "dtz_end": 0.005}


def test_reward_safety_barrier_and_shaping() -> None:
    r = DeliveryReward()
    assert r(_m(safe=False)) == -20.0                       # hard barrier, independent of the shaping terms
    assert r(_m(k6=True)) > r(_m(k6=False))                 # strict K6 is the dominant terminal term
    assert r(_m(gap=0.9)) > r(_m(gap=0.3))                  # goal-directed progress is monotone


def test_box_mapping_roundtrip_and_clip() -> None:
    theta = THETA_LO + 0.4 * (THETA_HI - THETA_LO)
    assert np.allclose(box_to_theta(theta_to_box(theta)), theta, atol=1e-9)
    assert np.all(box_to_theta(np.full(6, 2.0)) <= THETA_HI + 1e-9)     # out-of-range action clips into the box
    assert np.all(box_to_theta(np.full(6, -2.0)) >= THETA_LO - 1e-9)


def test_env_step_is_terminal_with_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "rollout_primitive", lambda snap, theta, cfg: _m(k6=True))
    monkeypatch.setattr(E, "delivery_success", lambda m, cfg: bool(m["k6_delivered"]))
    h = ScenarioHandoff("s0", "train", 0, object(), np.zeros(30), THETA_LO.copy())
    env = CoinDeliveryThetaOptionEnv([h], Standardizer.fit(np.zeros((2, 30))))
    o = env.reset(0)
    assert o.shape == (30,) and env.obs_dim == 30 and env.act_dim == 6
    o2, r, done, info = env.step(np.zeros(6, np.float32))
    assert done and info["terminal"] == 1.0 and info["tau"] > 0 and info["k6"] == 1.0 and info["safe"]


def _seed(train: float, dev: float, safe: float = 1.0) -> dict:
    return {"seed": 0, "train": {"k6": train, "safe": safe}, "dev": {"k6": dev, "safe": safe}, "final_dev": dev}


def test_gate_pass() -> None:
    g = gate([_seed(0.70, 0.55)], {"k6": 0.1, "safe": 1.0}, {"train": {"k6": 0.3}, "dev": {"k6": 0.3}})
    assert g["verdict"] == "R11_6A_REWARD_DRIVEN_DELIVERY_LEARNS" and g["beats_baselines"]


def test_gate_action_coordinate_insufficient_when_cannot_beat_warmstart() -> None:
    g = gate([_seed(0.30, 0.20)], {"k6": 0.1, "safe": 1.0}, {"train": {"k6": 0.30}, "dev": {"k6": 0.30}})
    assert g["verdict"] == "R11_6A_ACTION_COORDINATE_INSUFFICIENT"


def test_gate_reward_misspecified_when_unsafe() -> None:
    g = gate([_seed(0.70, 0.55, safe=0.5)], {"k6": 0.1, "safe": 1.0}, {"train": {"k6": 0.3}, "dev": {"k6": 0.3}})
    assert g["verdict"] == "R11_6A_REWARD_MISSPECIFIED"


def test_gate_optimization_stalled_when_dev_low_but_safe() -> None:
    g = gate([_seed(0.70, 0.40)], {"k6": 0.1, "safe": 1.0}, {"train": {"k6": 0.3}, "dev": {"k6": 0.3}})
    assert g["verdict"] == "R11_6A_OPTIMIZATION_STALLED"
