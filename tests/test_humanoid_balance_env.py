"""HumanoidBalanceEnv (SAC env) + run_humanoid_sac eval helper — integration.

Slower than the pure-Lyapunov unit tests (build a MuJoCo model via the hymeko CLI),
but bounded: small max_steps, few episodes. Covers the gym contract, deterministic
reset (§3), the fall-termination path, the "gravity-comp alone does not balance"
design invariant, and the SAC eval helper end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

from scenarios.humanoid.balance_env import HumanoidBalanceEnv  # noqa: E402


def _env(max_steps: int = 40) -> HumanoidBalanceEnv:
    return HumanoidBalanceEnv(max_steps=max_steps, seed=0)


def test_spaces_and_step_contract() -> None:
    env = _env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert env.action_space.shape == (env.model.nu,)
    obs2, r, term, trunc, info = env.step(np.zeros(env.model.nu))
    assert np.all(np.isfinite(obs2)) and np.isfinite(r)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert "V" in info and "upright" in info and info["V"] >= 0.0


def test_reset_is_deterministic() -> None:
    env = _env()
    o1, _ = env.reset(seed=7)
    o2, _ = env.reset(seed=7)
    assert np.allclose(o1, o2)                       # same seed -> identical init (§3)
    o3, _ = env.reset(seed=8)
    assert not np.allclose(o1, o3)                   # different seed -> different perturbation


def test_truncates_at_max_steps_when_upright() -> None:
    # zero residual = gravity-comp feedforward; it may drift but should survive a few steps
    env = _env(max_steps=5)
    env.reset(seed=0)
    steps, trunc = 0, False
    for _ in range(5):
        _o, _r, term, trunc, _i = env.step(np.zeros(env.model.nu))
        steps += 1
        if term or trunc:
            break
    assert steps <= 5 and (trunc or term)            # bounded episode


def test_fall_terminates() -> None:
    # max one-sided torque tips the underactuated humanoid -> terminated within the horizon
    env = _env(max_steps=200)
    env.reset(seed=0)
    fell = False
    for _ in range(200):
        _o, _r, term, _tr, _i = env.step(np.ones(env.model.nu))
        if term:
            fell = True
            break
    assert fell                                      # the fall-termination path is reachable


def test_gravity_comp_alone_does_not_balance() -> None:
    # design invariant: the feedforward is NOT a balancer — V must grow above V0 (it tips)
    env = _env(max_steps=120)
    _o, _ = env.reset(seed=0)
    v0 = env.V(env._com_sig())
    vmax = v0
    for _ in range(120):
        _o, _r, term, trunc, info = env.step(np.zeros(env.model.nu))
        vmax = max(vmax, info["V"])
        if term or trunc:
            break
    assert vmax > v0 + 0.05                           # energy grows -> gravity-comp does not stabilize


def test_eval_balance_helper_end_to_end() -> None:
    from hymeko_rl.train.sac import build_sac

    from scenarios.humanoid.run_humanoid_sac import _eval_balance

    env = _env(max_steps=30)
    od = int(env.observation_space.shape[0])
    torch.manual_seed(0)
    actor, _critics = build_sac("mlp", obs_dim=od, flat_dim=od,
                                action_dim=env.model.nu, action_scale=1.0, hidden=32)
    frac, lyap_rate = _eval_balance(env, actor, [1234, 1235])
    assert 0.0 <= frac <= 1.0 and 0.0 <= lyap_rate <= 1.0
