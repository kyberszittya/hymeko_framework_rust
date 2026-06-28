"""Tests for SAC: the squashed-Gaussian actor, twin soft-Q build, and the train loop.

Unit: actor sample is bounded with a finite (corrected) log-prob; action_mean is deterministic; build_sac
makes twin critics for mlp/hsikan. Integration: train_sac runs end-to-end on a tiny budget → finite curve.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hymeko_rl.ddpg import QCritic
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.sac import SACConfig, SquashedGaussianActor, build_sac, train_sac

_MJCF = emit_cartpole_mjcf()


def _env(max_steps: int = 200) -> InvertedPendulumEnv:
    return InvertedPendulumEnv(mjcf=_MJCF, max_steps=max_steps)


def test_squashed_actor_sample_bounded_with_finite_logprob() -> None:
    torch.manual_seed(0)
    env = _env()
    actor, _ = build_sac("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                         action_scale=10.0, hidden=16)
    assert isinstance(actor, SquashedGaussianActor)
    obs = torch.randn(8, env.hg.n_vertices, 2)
    a, logp = actor.sample(obs)
    assert a.shape == (8, 1) and a.abs().max() <= 10.0           # tanh-squashed into ±scale
    assert logp.shape == (8,) and torch.isfinite(logp).all()     # corrected log-prob, no NaN
    mean = actor.action_mean(obs)
    assert mean.shape == (8, 1) and mean.abs().max() <= 10.0      # deterministic, also bounded


@pytest.mark.parametrize("kind", ["mlp", "hsikan"])
def test_build_sac_twin_critics(kind: str) -> None:
    env = _env()
    kw = {} if kind == "mlp" else {"hg_state": env.hg}
    actor, critics = build_sac(kind, obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                               action_scale=10.0, n_critics=2, hidden=16, **kw)
    assert len(critics) == 2 and all(isinstance(c, QCritic) for c in critics)
    q = critics[0](torch.randn(3, env.hg.n_vertices, 2), torch.zeros(3, 1))
    assert q.shape == (3,) and torch.isfinite(q).all()


def test_train_sac_runs_and_returns_finite_curve() -> None:
    torch.manual_seed(0)
    env = _env(max_steps=50)
    actor, critics = build_sac("mlp", obs_dim=2, flat_dim=env.hg.n_vertices * 2, action_dim=1,
                               action_scale=10.0, hidden=16)
    hist = train_sac(actor, critics, env, SACConfig(total_steps=300, start_steps=50, batch_size=16,
                                                    capacity=500, eval_every=150, n_eval=2, seed=0))
    assert len(hist) == 2 and all(np.isfinite(hist))
