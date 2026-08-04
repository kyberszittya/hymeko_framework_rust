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
import hymeko_rl.experiments.r11_6a_delivery_rl as R
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


def test_gate_improvement_pass() -> None:
    # train preserved (0.70 >= 0.30-0.15) AND dev improves over the warm-start (0.55 > 0.30+0.05, >= 0.50), the seed gains.
    g = gate([_seed(0.70, 0.55)], {"k6": 0.1, "safe": 1.0}, {"train": {"k6": 0.30}, "dev": {"k6": 0.30}})
    assert g["verdict"] == "R11_6A_REWARD_DRIVEN_DELIVERY_IMPROVEMENT_PASS" and g["train_preserved"]


def test_gate_rl_unstable_when_train_collapses() -> None:
    # v1 shape: mean train 0.454 well below the 0.932 warm-start = TD3 destabilized it (drift / critic collapse).
    g = gate([_seed(0.454, 0.429)], {"k6": 0.0, "safe": 1.0}, {"train": {"k6": 0.932}, "dev": {"k6": 0.286}})
    assert g["verdict"] == "R11_6A_RL_UNSTABLE" and not g["train_preserved"]


def test_gate_prevents_forgetting_when_train_held_but_dev_flat() -> None:
    # v2 likely shape: the anchor holds train (~ warm-start) but dev stays ~ warm-start -> no improvement, not a failure.
    g = gate([_seed(0.90, 0.29)], {"k6": 0.0, "safe": 1.0}, {"train": {"k6": 0.932}, "dev": {"k6": 0.286}})
    assert g["verdict"] == "R11_6A_POSITIVE_REPLAY_PREVENTS_FORGETTING_STALLED" and g["train_preserved"]


def test_gate_improvement_needs_seed_majority() -> None:
    # mean dev 0.503 >= 0.50 but only 1/3 seeds beat the warm-start dev (0.30) -> majority fails -> not an improvement.
    seeds = [_seed(0.90, 0.95), _seed(0.90, 0.28), _seed(0.90, 0.28)]
    g = gate(seeds, {"k6": 0.0, "safe": 1.0}, {"train": {"k6": 0.90}, "dev": {"k6": 0.30}})
    assert g["verdict"] == "R11_6A_POSITIVE_REPLAY_PREVENTS_FORGETTING_STALLED" and g["seeds_with_dev_gain"] == "1/3"


def test_gate_reward_misspecified_when_unsafe() -> None:
    g = gate([_seed(0.70, 0.55, safe=0.5)], {"k6": 0.1, "safe": 1.0}, {"train": {"k6": 0.3}, "dev": {"k6": 0.3}})
    assert g["verdict"] == "R11_6A_REWARD_MISSPECIFIED"


def test_combined_eval_disqualifies_train_collapse(monkeypatch: pytest.MonkeyPatch) -> None:
    # train_sub idx = [0], dev idx = [1]; warm-start train_sub = 0.85, margin 0.15 => preserve floor 0.70.
    monkeypatch.setattr(R, "eval_actor", lambda a, e, idx: {"k6": (0.30 if idx == [0] else 0.60), "safe": 1.0})
    score, aux = R._make_combined_eval(None, [1], [0], 0.85, 0.15)(object())
    assert score == -1.0 and aux["preserved"] == 0.0                  # dev-lucky but train collapsed -> disqualified
    monkeypatch.setattr(R, "eval_actor", lambda a, e, idx: {"k6": (0.80 if idx == [0] else 0.60), "safe": 1.0})
    score2, aux2 = R._make_combined_eval(None, [1], [0], 0.85, 0.15)(object())
    assert score2 == 0.60 and aux2["preserved"] == 1.0               # train preserved -> score = dev
