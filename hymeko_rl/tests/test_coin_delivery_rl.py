"""Tests for COIN-DELIVERY-RL-1 (train.coin_delivery_rl + experiments.coin_delivery_rl1).

Covers: the residual zero-invariant + clipping, the potential-based reward sign gates, config validation, the
zero-residual head init, the phase-conditioned train env (handoff non-terminal, center-terminal, prefix reaches
handoff), the scripted-baseline band, and a tiny end-to-end PPO micro-smoke over the residual policy.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.agents.policy import build_policy
from hymeko_rl.train.coin_delivery_rl import (
    DeliveryRLConfig,
    delivery_reward,
    eval_delivery,
    greedy_action_fn,
    make_delivery_rl_env,
    residual_action,
    scripted_action_fn,
    zero_residual_head,
)
from hymeko_rl.train.ppo import PPOConfig, train_ppo

_HELD = range(64_000, 64_012)


# ── residual action (pure) ──────────────────────────────────────────────────────────────────────────────────────────
def test_residual_zero_invariant() -> None:
    base = np.array([0.6, -0.4, 0.0, 0.8, 0.1, -0.2], np.float32)
    out = residual_action(base, np.zeros(6, np.float32), 0.3, -1.0, 1.0)
    assert np.allclose(out, np.clip(base, -1.0, 1.0))
    assert out.dtype == np.float32


def test_residual_scales_and_clips() -> None:
    base = np.zeros(6, np.float32)
    # tanh(+inf)=1 → base + delta*1 = 0.3
    out = residual_action(base, np.full(6, 50.0, np.float32), 0.3, -1.0, 1.0)
    assert np.allclose(out, 0.3, atol=1e-4)
    # clipping: base near the ceiling + positive residual saturates at hi
    out2 = residual_action(np.full(6, 0.95, np.float32), np.full(6, 50.0, np.float32), 0.3, -1.0, 1.0)
    assert np.allclose(out2, 1.0)


# ── reward sign gates ────────────────────────────────────────────────────────────────────────────────────────────────
def test_reward_progress_positive() -> None:
    cfg = DeliveryRLConfig()
    r = delivery_reward(0.10, 0.088, entered_zone_now=False, center_now=False, stalled=False, dropped=False, cfg=cfg)
    assert r > 0


def test_reward_holding_nonpositive() -> None:
    cfg = DeliveryRLConfig()
    r = delivery_reward(0.10, 0.10, entered_zone_now=False, center_now=False, stalled=True, dropped=False, cfg=cfg)
    assert r <= 0


def test_reward_away_negative() -> None:
    cfg = DeliveryRLConfig()
    r = delivery_reward(0.10, 0.112, entered_zone_now=False, center_now=False, stalled=False, dropped=False, cfg=cfg)
    assert r < 0


def test_reward_center_is_strongest_event() -> None:
    cfg = DeliveryRLConfig()
    prog = delivery_reward(0.10, 0.088, entered_zone_now=False, center_now=False, stalled=False, dropped=False, cfg=cfg)
    zone = delivery_reward(0.05, 0.039, entered_zone_now=True, center_now=False, stalled=False, dropped=False, cfg=cfg)
    cent = delivery_reward(0.03, 0.019, entered_zone_now=False, center_now=True, stalled=False, dropped=False, cfg=cfg)
    assert cent > zone > prog > 0


def test_reward_drop_penalizes() -> None:
    cfg = DeliveryRLConfig()
    no_drop = delivery_reward(0.10, 0.10, entered_zone_now=False, center_now=False, stalled=False, dropped=False, cfg=cfg)
    drop = delivery_reward(0.10, 0.10, entered_zone_now=False, center_now=False, stalled=False, dropped=True, cfg=cfg)
    assert drop == pytest.approx(no_drop - cfg.w_drop)


# ── config validation (failure cases) ────────────────────────────────────────────────────────────────────────────────
def test_config_rejects_bad_delta() -> None:
    with pytest.raises(ValueError):
        DeliveryRLConfig(delta=0.0)


def test_config_rejects_center_gt_zone() -> None:
    with pytest.raises(ValueError):
        DeliveryRLConfig(center_tol=0.05, zone_half=0.04)


# ── zero-residual head ────────────────────────────────────────────────────────────────────────────────────────────────
def test_zero_residual_head_zeroes_actor_mean() -> None:
    ac = build_policy("mlp", obs_dim=41, action_dim=6)
    zero_residual_head(ac)
    assert torch.all(ac.actor_mean.weight == 0)
    assert torch.all(ac.actor_mean.bias == 0)
    out = ac.action_mean(torch.zeros(3, 41))
    assert torch.allclose(out, torch.zeros(3, 6))


# ── the phase-conditioned train env ──────────────────────────────────────────────────────────────────────────────────
def test_env_reset_shape_and_handoff() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    obs, info = env.reset(seed=64_000)
    assert obs.shape == (41,)
    assert "handoff_event" in info


def test_env_handoff_not_terminal_and_center_terminal() -> None:
    cfg = DeliveryRLConfig()
    env = make_delivery_rl_env(cfg)
    obs, info = env.reset(seed=64_000)
    # the prefix reaches handoff, yet the episode continues (delivery semantics)
    steps = 0
    af = scripted_action_fn()
    for _ in range(cfg.horizon):
        obs, _r, term, trunc, i = env.step(af(obs))
        steps += 1
        if term:
            # termination is on center-reach or safety, NOT on handoff
            assert bool(i["center_reached"]) or i["safety_violation"]
            break
        if trunc:
            break
    assert steps > 1
    assert info["handoff_event"] or steps == cfg.horizon


def test_env_zero_residual_matches_scripted_metrics() -> None:
    cfg = DeliveryRLConfig()
    ac = build_policy("mlp", obs_dim=41, action_dim=6)
    zero_residual_head(ac)
    seeds = list(_HELD)
    scripted = eval_delivery(scripted_action_fn(), seeds, cfg)
    zero_init = eval_delivery(greedy_action_fn(ac), seeds, cfg)
    assert scripted["center_reach"] == zero_init["center_reach"]
    assert scripted["zone_entry"] == zero_init["zone_entry"]
    assert (zero_init["residual_norm_med"] in (0.0, None))


def test_scripted_baseline_in_delivery_env0_band() -> None:
    cfg = DeliveryRLConfig()
    res = eval_delivery(scripted_action_fn(), range(64_000, 64_030), cfg)
    # DELIVERY-ENV-0 reference: zone-entry ~0.73, center-reach ~0.52-0.60
    assert res["zone_entry"] >= 0.55
    assert 0.45 <= res["center_reach"] <= 0.75


def test_eval_delivery_metric_keys() -> None:
    cfg = DeliveryRLConfig()
    res = eval_delivery(scripted_action_fn(), range(64_000, 64_006), cfg)
    for k in ("zone_entry", "center_reach", "final_dtz_med", "time_to_center_med", "return_med", "residual_norm_med"):
        assert k in res


# ── PPO micro-smoke (integration; the training path must run end-to-end) ──────────────────────────────────────────────
def test_ppo_micro_smoke_runs() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = DeliveryRLConfig()
    ac = build_policy("mlp", obs_dim=41, action_dim=6)
    zero_residual_head(ac)
    env = make_delivery_rl_env(cfg)
    pcfg = PPOConfig(n_iters=1, n_steps=256, seed=0)
    hist = train_ppo(ac, env, pcfg)
    assert len(hist) == 1
    assert np.isfinite(hist[0])
