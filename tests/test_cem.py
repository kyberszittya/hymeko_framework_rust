"""The shared CEM optimiser + linear policy (framework core the humanoid trainers reuse)."""

from __future__ import annotations

import numpy as np

from scenarios.humanoid.cem import cem_optimize, linear_policy, policy_dim

_TARGET = np.array([0.5, -0.3, 0.8])


def _quad_eval(args):
    """Top-level (picklable) eval: maximise −‖theta − target‖² — a smooth CEM sanity target."""
    theta, _cfg = args
    return (-float(np.sum((theta - _TARGET) ** 2)),)


def test_cem_converges_to_target() -> None:
    best_theta, best = cem_optimize(_quad_eval, None, dim=3, iters=40, pop=40, elite=8, seed=0)
    assert np.allclose(best_theta, _TARGET, atol=0.1)   # finds the optimum
    assert best[0] > -0.05                              # near-zero residual


def test_cem_is_deterministic_per_seed() -> None:
    t1, _ = cem_optimize(_quad_eval, None, dim=3, iters=10, pop=20, elite=5, seed=7)
    t2, _ = cem_optimize(_quad_eval, None, dim=3, iters=10, pop=20, elite=5, seed=7)
    assert np.array_equal(t1, t2)


def test_linear_policy_and_dim() -> None:
    obs_dim, act_dim = 5, 2
    assert policy_dim(obs_dim, act_dim) == act_dim * obs_dim + act_dim
    theta = np.zeros(policy_dim(obs_dim, act_dim))
    a = linear_policy(theta, np.ones(obs_dim), obs_dim, act_dim)
    assert a.shape == (act_dim,)
    assert np.allclose(a, 0.0)                          # tanh(0) = 0

    theta2 = np.ones(policy_dim(obs_dim, act_dim))       # W=1, b=1 -> tanh(obs·1 + 1)
    a2 = linear_policy(theta2, np.ones(obs_dim), obs_dim, act_dim)
    assert np.allclose(a2, np.tanh(obs_dim + 1.0))
