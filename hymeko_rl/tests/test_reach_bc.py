"""Phase 1 — the reaching env + behaviour-cloning the HSiKAN / MLP policies.

Run: pytest -p no:randomly hymeko_rl/tests/test_reach_bc.py
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.bc import behaviour_clone, collect_demos, run_bc
from hymeko_rl.env.arm_reach_env import _NODE_FEAT, ArmReachEnv
from hymeko_rl.policy import build_policy


@pytest.mark.parametrize("mode", ["torque", "position", "velocity"])
def test_expert_reaches_in_each_control_mode(mode: str) -> None:
    """The closed-loop demonstrator reaches in every actuator interface (torque uses proper
    inverse dynamics ``τ = M·a_des + bias``) — the ablation axis is sound end to end."""
    env = ArmReachEnv(control_mode=mode, max_steps=80)
    fins = []
    for s in range(6):
        obs, _ = env.reset(seed=s)
        done, last = False, 1.0
        while not done:
            obs, _, term, trunc, info = env.step(env.expert_action)
            last = float(info["dist"])
            done = term or trunc
        fins.append(last)
    assert np.mean(fins) < 0.25   # the expert genuinely reaches the target


def test_unknown_control_mode_raises() -> None:
    with pytest.raises(ValueError):
        ArmReachEnv(control_mode="impedance")


def test_env_obs_action_contract() -> None:
    env = ArmReachEnv(max_steps=10)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.hg.n_vertices, _NODE_FEAT)
    assert info["target"].shape == (3,) and env.expert_action.shape == (env.n_actions,)
    obs2, reward, term, trunc, info2 = env.step(env.expert_action)
    assert obs2.shape == obs.shape and np.isfinite(reward)
    assert "dist" in info2 and isinstance(term, bool)


def test_target_is_fk_reachable() -> None:
    """The sampled target is a real forward-kinematics position (non-degenerate arm),
    so reaching is a genuine task — guards against the all-Z degenerate emitted arms."""
    env = ArmReachEnv()
    targets = np.stack([env.reset(seed=s)[1]["target"] for s in range(8)])
    assert targets.std(axis=0).sum() > 0.05   # the EE workspace actually spans 3-D


def test_collect_demos_shapes() -> None:
    env = ArmReachEnv(max_steps=12)
    obs, acts = collect_demos(env, n_episodes=3, seed=0)
    assert obs.ndim == 3 and obs.shape[1:] == (env.hg.n_vertices, _NODE_FEAT)
    assert acts.shape == (len(obs), env.n_actions) and len(obs) > 0


def test_behaviour_clone_reduces_loss() -> None:
    env = ArmReachEnv(max_steps=12)
    obs, acts = collect_demos(env, n_episodes=6, seed=0)
    ac = build_policy("hsikan", obs_dim=_NODE_FEAT, action_dim=env.n_actions,
                      hg_state=env.hg, hidden=16)
    losses = behaviour_clone(ac, obs, acts, n_epochs=40, seed=0)
    assert losses[-1] < losses[0]            # learning happened


def test_hsikan_bc_beats_untrained_floor() -> None:
    """The end-to-end Phase-1 claim: the hypergraph-reading policy, behaviour-cloned from
    the closed-loop expert, reaches closer to the target than the untrained policy."""
    r = run_bc("hsikan", n_demos=20, n_epochs=100, hidden=32, seed=0, n_eval=12)
    assert r["reach_err_m"] < r["untrained_floor_m"]      # BC learned to reach
    assert r["n_samples"] > 0 and r["n_params"] > 0


def test_mlp_bc_trains() -> None:
    """The MLP baseline *trains* (loss falls). Whether it reaches as well as HSiKAN is the
    matched-capacity ablation deferred to Phase 2 (PPO / on-policy data) — not asserted
    here, where capacity is unmatched and BC suffers covariate shift."""
    env = ArmReachEnv(max_steps=12)
    obs, acts = collect_demos(env, n_episodes=6, seed=0)
    ac = build_policy("mlp", obs_dim=env.hg.n_vertices * _NODE_FEAT,
                      action_dim=env.n_actions, hidden=32)
    losses = behaviour_clone(ac, obs, acts, n_epochs=40, seed=0)
    assert losses[-1] < losses[0]
