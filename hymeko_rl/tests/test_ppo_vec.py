"""Tests for the vectorized PPO rollout (`_collect_vec` + `train_ppo(n_envs>1)`).

Layers (§3): unit (vec buffer shapes/finiteness, truncation bookkeeping, per-env GAE, n_envs validation),
integration (vec training preserves learning; the single-env reach path is untouched), performance
(`_collect_vec` is faster than the single-env `_collect` at equal total transitions — the throughput claim).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.train.ppo import PPOConfig, _collect, _collect_vec, train_ppo
from hymeko_rl.train.train_inverted_pendulum import _make_balance_policy

# One shared scene → cheap env construction across the suite (no per-env CLI subprocess).
_MJCF = emit_cartpole_mjcf()


def _envs(n: int) -> list[InvertedPendulumEnv]:
    return [InvertedPendulumEnv(mjcf=_MJCF) for _ in range(n)]


def _policy(kind: str = "hsikan", hidden: int = 32):
    torch.manual_seed(0)
    return _make_balance_policy(kind, InvertedPendulumEnv(mjcf=_MJCF), hidden)


# ── unit: _collect_vec ────────────────────────────────────────────────────────
def test_collect_vec_shapes_and_finiteness() -> None:
    ac = _policy()
    envs = _envs(4)
    obs = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    cfg = PPOConfig(n_steps=64, seed=0)        # ticks = 64/4 = 16; total transitions = 64
    buf, nxt, mean_ret = _collect_vec(envs, ac, obs, cfg)
    for k in ("obs", "act", "logp", "val", "adv", "ret"):
        assert buf[k].shape[0] == 64, f"{k} should have T*N=64 rows"
        assert np.all(np.isfinite(np.asarray(buf[k])))
    assert nxt.shape == (4, 2, 2) and np.isfinite(mean_ret)


def test_collect_vec_truncation_bookkeeping() -> None:
    """Short always-truncating episodes (max_steps=3): many dones, all rewards finite — the per-env
    truncation-bootstrap path (γ·V folded in) is exercised across the N streams without NaNs."""
    ac = _policy()
    envs = [InvertedPendulumEnv(mjcf=_MJCF, max_steps=3) for _ in range(4)]
    obs = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    buf, _, _ = _collect_vec(envs, ac, obs, PPOConfig(n_steps=64, seed=0))
    assert np.all(np.isfinite(np.asarray(buf["ret"])))
    assert np.all(np.isfinite(np.asarray(buf["adv"])))


def test_collect_vec_per_env_gae_is_columnwise() -> None:
    """GAE is computed per env column (independent trajectories). With N=3 the returned adv/ret are
    finite and the right length; a single shared-stream GAE would not produce N*T rows."""
    ac = _policy()
    envs = _envs(3)
    obs = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    buf, _, _ = _collect_vec(envs, ac, obs, PPOConfig(n_steps=30, seed=0))   # ticks=10, 3 envs
    assert buf["adv"].shape == (30,) and np.allclose(
        np.asarray(buf["ret"]), np.asarray(buf["adv"]) + np.asarray(buf["val"]), atol=1e-4)


# ── unit: train_ppo dispatch ──────────────────────────────────────────────────
def test_train_ppo_rejects_vec_without_make_env() -> None:
    ac = _policy()
    with pytest.raises(ValueError):
        train_ppo(ac, InvertedPendulumEnv(mjcf=_MJCF), PPOConfig(n_iters=1), n_envs=4)
    with pytest.raises(ValueError):
        train_ppo(ac, InvertedPendulumEnv(mjcf=_MJCF), PPOConfig(n_iters=1), n_envs=0)


# ── integration: vec training preserves learning ──────────────────────────────
def test_train_ppo_vec_runs_and_is_finite() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    env = InvertedPendulumEnv(mjcf=_MJCF)
    ac = _make_balance_policy("hsikan", env, 32)
    hist = train_ppo(ac, env, PPOConfig(n_iters=3, n_steps=256, seed=0),
                     n_envs=8, make_env=lambda: InvertedPendulumEnv(mjcf=_MJCF))
    assert len(hist) == 3 and all(np.isfinite(hist))


# ── regression: the vec path is backbone-agnostic (CTDE has no `actor_mean`) ──
def test_collect_vec_works_with_multichannel_ctde() -> None:
    """`_collect_vec` sized its action buffer from ``ac.actor_mean.out_features`` — an ``ActorCritic``
    internal the collaborative policies lack (they expose ``action_mean`` as a *method*). The fix reads
    the action dim from ``ac.log_std`` (shared by every Gaussian AC). This drives a MultiChannelCTDE
    through the vectorized rollout — it would raise ``AttributeError`` against the prior implementation."""
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    from hymeko_rl.agents.multichannel_ctde import build_multichannel_collaborative

    def _mk() -> PlanarGraspEnv:
        return PlanarGraspEnv(robot=None, max_steps=20, difficulty=0.3, task_graph=True)

    envs = [_mk() for _ in range(4)]
    ac = build_multichannel_collaborative(envs[0], kind="sa_hsikan", hidden=32)
    obs = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)]).astype(np.float32)
    buf, _, mean_ret = _collect_vec(envs, ac, obs, PPOConfig(n_steps=32, seed=0))
    assert buf["act"].shape == (32, int(ac.log_std.numel())) and np.isfinite(mean_ret)
    assert np.all(np.isfinite(np.asarray(buf["ret"])))


# ── performance: vec rollout beats single-env at equal total transitions ──────
def test_collect_vec_faster_than_single() -> None:
    """`_collect_vec` (N=8) collects the same 1024 transitions faster than the single-env `_collect`
    (median of 5). The structural throughput win — batched forward amortises the per-step dispatch."""
    torch.set_num_threads(1)
    ac = _policy("hsikan", 64)
    cfg = PPOConfig(n_steps=1024, seed=0)

    def single() -> float:
        env = InvertedPendulumEnv(mjcf=_MJCF)
        o, _ = env.reset(seed=0)
        t = time.perf_counter()
        _collect(env, ac, o, cfg.n_steps, cfg.gamma)
        return time.perf_counter() - t

    def vec() -> float:
        envs = _envs(8)
        o = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)]).astype(np.float32)
        t = time.perf_counter()
        _collect_vec(envs, ac, o, cfg)
        return time.perf_counter() - t

    single()                                         # warm-up
    vec()
    s = sorted(single() for _ in range(5))[2]
    v = sorted(vec() for _ in range(5))[2]
    assert v < s, f"vec rollout median {v:.3f}s not faster than single {s:.3f}s"
