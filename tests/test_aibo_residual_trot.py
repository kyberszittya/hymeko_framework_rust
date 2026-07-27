"""Residual-over-trot env — the bounded-residual goal-reaching scaffold behind the AIBO training.

Certifies: obs/action dims; reset samples a goal in the configured range; the bounded residual never
departs from the scaffold by more than ``residual_scale`` (the safe-scaffold / coin-R8 prerequisite);
the scaffold (a=0) never falls; the reward rewards progress toward the goal; and reset is seed-det.
"""

from __future__ import annotations

import numpy as np
import pytest

from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv


@pytest.fixture(scope="module")
def env() -> ResidualTrotEnv:
    return ResidualTrotEnv(ResidualTrotConfig(), seed=0)


def test_obs_and_action_dims(env: ResidualTrotEnv) -> None:
    obs, _ = env.reset(seed=1)
    assert obs.shape == (9,)
    assert env.action_space.shape == (12,)


def test_reset_samples_goal_in_configured_range(env: ResidualTrotEnv) -> None:
    for s in (1, 2, 3):
        env.reset(seed=s)
        assert env.cfg.dist_lo - 1e-6 <= env._env.dist_to_goal() <= env.cfg.dist_hi + 0.05


def test_reset_is_seed_deterministic(env: ResidualTrotEnv) -> None:
    env.reset(seed=7)
    g1 = np.array(env._env.goal)
    env.reset(seed=7)
    assert np.allclose(g1, env._env.goal)


def test_bounded_residual_never_departs_more_than_scale(env: ResidualTrotEnv) -> None:
    base = np.zeros(12)
    for r in (np.ones(12), -np.ones(12), np.full(12, 5.0), env.action_space.sample()):
        final = env.blend_action(base, r)
        assert np.all(np.abs(final - base) <= env.cfg.residual_scale + 1e-9)


def test_zero_residual_is_pure_scaffold(env: ResidualTrotEnv) -> None:
    base = np.array([0.3, -0.4, 0.1, 0.0, 0.2, -0.2, 0.5, -0.5, 0.1, -0.1, 0.3, -0.3])
    assert np.allclose(env.blend_action(base, np.zeros(12)), base)   # a=0 applies the scaffold exactly


def test_scaffold_never_falls_safe_prerequisite(env: ResidualTrotEnv) -> None:
    # the coin-R8 prerequisite: the scaffold (a=0) is a SAFE base — it stays upright on every goal.
    zero = lambda _o: np.zeros(12)  # noqa: E731
    for i, goal in enumerate([(0.5, 0), (0.7, 30), (0.6, -30)]):
        _md, _ok, min_up = env.rollout_min_dist(zero, goal, seed=200 + i, horizon=600)
        assert min_up > 0.5


def test_steer_mode_has_two_dim_param_action() -> None:
    steer = ResidualTrotEnv(ResidualTrotConfig(residual_mode="steer"), seed=0)
    steer.reset(seed=1)
    assert steer.action_space.shape == (2,)          # (Δyaw, Δdrive) gait-parameter residual


def test_zero_residual_matches_pure_scaffold_in_both_modes() -> None:
    # a = 0 must apply the IDENTICAL pure-scaffold action in leg and steer mode (both leave the gait
    # untouched) — so a zero-residual rollout reaches the same min-distance either way.
    leg = ResidualTrotEnv(ResidualTrotConfig(residual_mode="leg"), seed=0)
    steer = ResidualTrotEnv(ResidualTrotConfig(residual_mode="steer"), seed=0)
    zero_leg = lambda _o: np.zeros(12)   # noqa: E731
    zero_steer = lambda _o: np.zeros(2)  # noqa: E731
    md_leg, _o1, _u1 = leg.rollout_min_dist(zero_leg, (0.6, 20), seed=321, horizon=500)
    md_steer, _o2, _u2 = steer.rollout_min_dist(zero_steer, (0.6, 20), seed=321, horizon=500)
    assert md_leg == pytest.approx(md_steer, abs=1e-6)


def test_reward_rewards_progress(env: ResidualTrotEnv) -> None:
    # force a STRAIGHT goal (the scaffold's strength) so progress is unambiguous, then check the
    # progress reward is positive as the scaffold closes distance.
    env.reset(seed=5)
    tx = float(env._env.data.xpos[env._env.torso, 0])
    ty = float(env._env.data.xpos[env._env.torso, 1])
    env._env.goal = np.array([tx + 0.6, ty], np.float32)
    env._env._prev_dist = env._env.dist_to_goal()
    env._prev_dist = float(env._env.dist_to_goal())
    d0 = env._prev_dist
    rewards = []
    for _ in range(300):
        _o, r, term, trunc, info = env.step(np.zeros(12))
        rewards.append(r)
        if term or trunc:
            break
    assert info["dist"] < d0 - 0.05                 # the scaffold clearly progresses on a straight goal
    assert sum(rewards) > 0.0                        # net progress reward is positive
