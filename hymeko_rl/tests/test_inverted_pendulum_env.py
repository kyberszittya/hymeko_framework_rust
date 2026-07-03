"""Tests for the HyMeKo cart-pole HSiKAN actor-critic wire-in.

Layers (CLAUDE.md §3): unit (strip_actuators, env shapes/reset/termination/reward, task-non-triviality),
integration (both backbones forward + the shared PPO loop runs end-to-end), performance (a fixed-seed
short train asserts a wall-time budget over 5+ measured iterations + a peak-RSS budget).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from hymeko_rl.env.arm_world import emit_arm_mjcf, strip_actuators
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv
from hymeko_rl.train.ppo import PPOConfig, train_ppo
from hymeko_rl.train.train_inverted_pendulum import _make_balance_policy, run_balance

_HYMEKO = "data/robotics/inverted_pendulum.hymeko"


# ── unit: strip_actuators ─────────────────────────────────────────────────────
def test_strip_actuators_drops_only_named_motor() -> None:
    mjcf = emit_arm_mjcf(_HYMEKO, name="cp")
    assert 'joint="hinge"' in mjcf and 'joint="rail"' in mjcf
    stripped = strip_actuators(mjcf, ["hinge"])
    assert 'name="act_hinge"' not in stripped       # pole motor gone
    assert 'name="act_rail"' in stripped            # cart motor intact


def test_strip_actuators_empty_is_identity() -> None:
    mjcf = emit_arm_mjcf(_HYMEKO, name="cp")
    assert strip_actuators(mjcf, []) == mjcf


# ── unit: env contract ────────────────────────────────────────────────────────
def test_env_shapes_and_single_actuator() -> None:
    env = InvertedPendulumEnv()
    assert env.observation_space.shape == (2, 2)    # (cart, pole) × [qpos, qvel]
    assert env.action_space.shape == (1,)           # cart force only
    assert int(env.model.nu) == 1                   # pole is passive
    assert env.hg.n_vertices == 2
    obs, info = env.reset(seed=0)
    assert obs.shape == (2, 2) and obs.dtype == np.float32


def test_reset_starts_near_upright() -> None:
    env = InvertedPendulumEnv(init_noise=0.05)
    for s in range(8):
        env.reset(seed=s)
        assert abs(float(env.data.qpos[env._pole_q])) <= 0.05 + 1e-6


def test_invalid_params_rejected() -> None:
    with pytest.raises(ValueError):
        InvertedPendulumEnv(angle_limit=0.0)
    with pytest.raises(ValueError):
        InvertedPendulumEnv(frame_skip=0)


def test_termination_on_pole_fall() -> None:
    env = InvertedPendulumEnv()
    env.reset(seed=0)
    env.data.qpos[env._pole_q] = 0.5            # well past angle_limit 0.2
    _, reward, terminated, _, info = env.step(np.zeros(1, dtype=np.float32))
    assert terminated and info["fell"]
    assert reward == 0.0                          # no alive reward on the terminating step


def test_termination_on_cart_out_of_bounds() -> None:
    env = InvertedPendulumEnv()
    env.reset(seed=0)
    env.data.qpos[env._cart_q] = 2.0            # past cart_limit 1.0
    _, _, terminated, _, info = env.step(np.zeros(1, dtype=np.float32))
    assert terminated and info["out_of_bounds"]


def test_alive_reward_is_one_while_upright() -> None:
    env = InvertedPendulumEnv()
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(np.zeros(1, dtype=np.float32))
    assert not terminated and reward == 1.0


def test_task_is_nontrivial_uncontrolled_pole_falls_within_horizon() -> None:
    """Regression guard for the frame_skip calibration: with zero control the pole must fall (terminate)
    *before* max_steps — else the benchmark would be trivially survivable and could not discriminate."""
    env = InvertedPendulumEnv()
    env.reset(seed=1)
    env.data.qpos[env._pole_q] = 0.05           # a tilt the agent would have to correct
    steps = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.zeros(1, dtype=np.float32))
        steps += 1
        done = terminated or truncated
    assert steps < env.max_steps                 # uncontrolled → falls inside the horizon


# ── integration: both backbones + the shared PPO loop ─────────────────────────
@pytest.mark.parametrize("kind", ["hsikan", "mlp"])
def test_policy_forwards_on_env_obs(kind: str) -> None:
    torch.manual_seed(0)
    env = InvertedPendulumEnv()
    ac = _make_balance_policy(kind, env, hidden=32)
    obs, _ = env.reset(seed=0)
    action, logp, value = ac.act(torch.as_tensor(obs[None], dtype=torch.float32))
    assert action.shape == (1, 1) and value.shape == (1,)


@pytest.mark.parametrize("kind", ["hsikan", "mlp"])
def test_train_ppo_runs_end_to_end(kind: str) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    env = InvertedPendulumEnv()
    ac = _make_balance_policy(kind, env, hidden=32)
    hist = train_ppo(ac, env, PPOConfig(n_iters=2, n_steps=256, seed=0))
    assert len(hist) == 2 and all(np.isfinite(hist))


def test_run_balance_summary_keys() -> None:
    res = run_balance("hsikan", hidden=32, seed=0, n_eval=3,
                      cfg=PPOConfig(n_iters=2, n_steps=256, seed=0))
    for k in ("policy", "n_params", "init_return", "final_return", "upright_steps", "max_steps"):
        assert k in res
    assert 0.0 <= float(res["upright_steps"]) <= float(res["max_steps"])


# ── performance: wall-time + peak-RSS budgets (§3) ────────────────────────────
def test_train_ppo_perf_budget() -> None:
    """Median over 5 measured single-iteration trains (fresh policy each round) must stay under a wall
    budget; peak RSS must stay well under the 16 GB cap. Reports median/IQR/worst (§3 benchmark stability).

    pytest-benchmark's fixture loop is unsuited here (each round needs a fresh env+policy and mutates env
    state), so this is a documented manual median-of-N with per-round setup and an explicit budget assert.
    """
    env = InvertedPendulumEnv()
    cfg = PPOConfig(n_iters=1, n_steps=512, seed=0)
    # warm-up (JIT/alloc), then 5 measured iterations.
    torch.manual_seed(0)
    train_ppo(_make_balance_policy("hsikan", env, 64), InvertedPendulumEnv(), cfg)
    times = []
    for r in range(5):
        ac = _make_balance_policy("hsikan", env, 64)
        t = time.perf_counter()
        train_ppo(ac, InvertedPendulumEnv(), cfg)
        times.append(time.perf_counter() - t)
    times.sort()
    median, worst = times[2], times[-1]
    iqr = times[3] - times[1]
    # Budget: one 512-step PPO iter on the 2-body env is well under 5 s median on CPU (smoke measured
    # ~0.4 s/iter). Generous ceiling so the gate is a regression alarm, not a flaky timer.
    assert median < 5.0, f"median {median:.2f}s (iqr {iqr:.2f}, worst {worst:.2f}) over budget"
    try:
        import psutil
        mi = psutil.Process().memory_info()
        peak_mb = getattr(mi, "peak_wset", mi.rss) / 1e6
        assert peak_mb < 4000.0, f"peak RSS {peak_mb:.0f} MB over the 4 GB test budget"
    except ImportError:
        pytest.skip("psutil unavailable — RSS budget not asserted")
