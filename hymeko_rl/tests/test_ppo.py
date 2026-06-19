"""Phase 2 — PPO smoke: the loop runs and improves return on the reaching reward.

Run: pytest -p no:randomly hymeko_rl/tests/test_ppo.py
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.bc import _make_policy
from hymeko_rl.env.arm_reach_env import ArmReachEnv
from hymeko_rl.ppo import PPOConfig, _collect, _gae, train_ppo


def test_collect_handles_truncation_bootstrap() -> None:
    """``_collect`` rolls past episode boundaries with finite rewards and marks each
    truncation as done. The truncation bootstrap (``+ γ·V(next_obs)`` on a time-limit cut)
    is what stopped PPO degrading good policies — here we exercise that path with short,
    always-truncating episodes."""
    torch.manual_seed(0)
    np.random.seed(0)
    env = ArmReachEnv(control_mode="position", max_steps=8)
    ac = _make_policy("hsikan", env, hidden=16)
    obs, _ = env.reset(seed=0)
    buf, nxt, last_val, mean_ret = _collect(env, ac, obs, n_steps=40, gamma=0.99)
    assert buf["rew"].shape == (40,) and np.all(np.isfinite(buf["rew"]))
    assert buf["done"].sum() >= 3 and np.isfinite(mean_ret) and np.isfinite(last_val)


def test_gae_shapes_and_finiteness() -> None:
    rews = np.array([1.0, -0.5, 0.2], dtype=np.float32)
    vals = np.array([0.1, 0.0, -0.1], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    adv, ret = _gae(rews, vals, dones, last_val=0.0, gamma=0.99, lam=0.95)
    assert adv.shape == (3,) and ret.shape == (3,)
    assert np.all(np.isfinite(adv)) and np.allclose(ret, adv + vals)


def test_ppo_improves_return() -> None:
    """A short PPO run increases the mean episodic return on the reaching reward (negative
    EE distance) — the loop learns. Uses position control, where learning is fast and
    reliable; torque (the harder mode) is exercised at experiment scale, not in CI."""
    torch.manual_seed(0)
    np.random.seed(0)
    env = ArmReachEnv(control_mode="position")
    ac = _make_policy("hsikan", env, hidden=32)
    history = train_ppo(ac, env, PPOConfig(n_iters=15, n_steps=512, minibatch=128))
    # later iterations should beat the first (more reward = closer to target).
    assert np.mean(history[-3:]) > history[0]


def test_value_warmup_runs_and_keeps_finite_history() -> None:
    """The critic-warm-up path (the cold-critic fix) runs and PPO still produces a finite
    return history; the warmed critic is non-trivial (not the cold all-zero predictor)."""
    torch.manual_seed(0)
    np.random.seed(0)
    env = ArmReachEnv()
    ac = _make_policy("hsikan", env, hidden=32)
    history = train_ppo(
        ac, env, PPOConfig(n_iters=4, n_steps=512, minibatch=128, value_warmup=2))
    assert len(history) == 4 and all(np.isfinite(h) for h in history)
    with torch.no_grad():
        obs, _ = env.reset(seed=0)
        v = ac.act(torch.as_tensor(obs[None], dtype=torch.float32))[2]
    assert abs(float(v.item())) > 1e-3   # critic learned a non-zero value scale
