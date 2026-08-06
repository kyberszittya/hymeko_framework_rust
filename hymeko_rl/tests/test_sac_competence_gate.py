"""Focused regression tests for the Phase-B corrected-SAC integration:

  1. SACConfig.stable(...) accepts log_every/eval_every at CONSTRUCTION (the FrozenInstanceError repair — the config is
     frozen and must never be mutated post-construction).
  2. train_sac's competence-gated bc_coef hook (bc_coef_fn) is honoured — the anchor uses bc_coef_fn(step), not the
     constant cfg.bc_coef, when the hook is supplied.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hymeko_rl.train.sac import SACConfig, build_sac, train_sac


def test_sacconfig_stable_sets_log_every_at_construction() -> None:
    # the repair: pass log_every/eval_every through stable()'s **overrides, not by mutating a frozen instance
    cfg = SACConfig.stable(total_steps=100, seed=0, bc_coef=1.0, log_every=250, eval_every=250)
    assert cfg.log_every == 250 and cfg.eval_every == 250 and cfg.bc_coef == 1.0
    # frozen: mutation must still raise (we did NOT drop frozen=True to suppress the error)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.log_every = 10  # type: ignore[misc]


def test_train_sac_honours_bc_coef_fn_hook() -> None:
    """The competence gate: with demos + a bc_coef_fn, the anchor is modulated by the hook (called per update step)."""
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=act_dim, action_scale=2.0)
    demo_o = np.zeros((32, obs_dim), np.float32)
    demo_a = np.zeros((32, act_dim), np.float32)
    seen: list[int] = []

    def bc_coef_fn(step: int) -> float:
        seen.append(step)
        return 0.3

    cfg = SACConfig.stable(total_steps=60, seed=0, bc_coef=1.0, start_steps=10, batch_size=16,
                           log_every=0, eval_every=1000)
    train_sac(actor, critics, env, cfg, offline_data=(demo_o, demo_a), bc_coef_fn=bc_coef_fn)
    assert seen, "bc_coef_fn was never called — the competence hook is not wired into the actor loss"
    assert all(s >= 10 for s in seen)                              # only after start_steps (when updates begin)
