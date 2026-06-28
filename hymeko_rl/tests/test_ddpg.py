"""Tests for the DDPG off-policy core: replay buffer, actor/critic, and the train loop.

Unit: ReplayBuffer ring/sample; DeterministicActor bounded output; QCritic shape; build_ddpg for mlp/hsikan.
Integration: train_ddpg runs end-to-end on a tiny budget and returns a finite curve.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.ddpg import (
    DeterministicActor, OffPolicyConfig, QCritic, build_offpolicy, td3_config, train_offpolicy,
)
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.replay import ReplayBuffer

_MJCF = emit_cartpole_mjcf()


def _env(max_steps: int = 200) -> InvertedPendulumEnv:
    return InvertedPendulumEnv(mjcf=_MJCF, max_steps=max_steps)


# ── ReplayBuffer ──────────────────────────────────────────────────────────────
def test_replay_ring_caps_size_and_overwrites() -> None:
    buf = ReplayBuffer(capacity=3, obs_shape=(2, 2), action_dim=1)
    for i in range(5):
        buf.add(np.full((2, 2), i, np.float32), np.array([i], np.float32), float(i),
                np.zeros((2, 2), np.float32), done=False)
    assert buf.size == 3                       # capped at capacity, not 5
    s, a, r, s2, d = buf.sample(3, generator=np.random.default_rng(0))
    assert s.shape == (3, 2, 2) and a.shape == (3, 1) and r.shape == (3,) and d.shape == (3,)
    assert set(np.unique(r.numpy())) <= {2.0, 3.0, 4.0}   # only the last 3 transitions survive


def test_replay_rejects_bad_batch_and_params() -> None:
    buf = ReplayBuffer(capacity=4, obs_shape=(2, 2), action_dim=1)
    with pytest.raises(ValueError):
        buf.sample(1, generator=np.random.default_rng(0))   # empty buffer
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0, obs_shape=(2, 2), action_dim=1)


# ── actor / critic ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ["mlp", "hsikan"])
def test_actor_bounded_and_critic_scalar(kind: str) -> None:
    torch.manual_seed(0)
    env = _env()
    kw = {} if kind == "mlp" else {"hg_state": env.hg}
    actor, critics = build_offpolicy(kind, obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                                     action_scale=10.0, hidden=16, **kw)
    assert isinstance(actor, DeterministicActor) and len(critics) == 1 and isinstance(critics[0], QCritic)
    obs = torch.randn(5, env.hg.n_vertices, 2)
    a = actor(obs)
    assert a.shape == (5, 1) and a.abs().max() <= 10.0          # bounded to ±action_scale
    q = critics[0](obs, a)
    assert q.shape == (5,) and torch.isfinite(q).all()


def test_td3_builds_twin_critics() -> None:
    """TD3 uses two critics (clipped double-Q); DDPG uses one — the same builder, an n_critics axis."""
    env = _env()
    _, c2 = build_offpolicy("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                            action_scale=10.0, n_critics=2, hidden=16)
    assert len(c2) == 2
    assert td3_config().n_critics == 2 and td3_config().policy_delay == 2 and td3_config().target_noise > 0


# ── train loop (integration, tiny budget) ────────────────────────────────────
@pytest.mark.parametrize("algo_cfg", [
    OffPolicyConfig(total_steps=300, start_steps=50, batch_size=16, capacity=500,
                    eval_every=150, n_eval=2, seed=0),                         # DDPG
    td3_config(total_steps=300, start_steps=50, batch_size=16, capacity=500,
               eval_every=150, n_eval=2, seed=0)])                            # TD3
def test_train_offpolicy_runs_and_returns_finite_curve(algo_cfg: OffPolicyConfig) -> None:
    torch.manual_seed(0)
    env = _env(max_steps=50)
    actor, critics = build_offpolicy("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                                     action_scale=10.0, n_critics=algo_cfg.n_critics, hidden=16)
    hist = train_offpolicy(actor, critics, env, algo_cfg)
    assert len(hist) == 2 and all(np.isfinite(hist))           # two evals at 150, 300
