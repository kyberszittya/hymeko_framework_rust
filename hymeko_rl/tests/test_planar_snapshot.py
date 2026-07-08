"""Snapshot/restore determinism + branch-return correctness for the fair vector-critic retest.

If restore is not byte-deterministic, the gradient-alignment probe (different first actions from the SAME state)
is measuring noise, not counterfactuals — so these are load-bearing regression tests, not smoke.
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.env.planar_snapshot import (
    branch_component_returns,
    restore_planar,
    snapshot_planar,
)
from hymeko_rl.train.search_objective import COMPONENTS, SearchObjective


def _env():
    return PlanarGraspEnv(robot=None, max_steps=60, difficulty=0.3)


def _walk_to_contact(env, steps=25, seed=9000):
    """Step a fixed pseudo-action forward so the coin is engaged (a non-trivial mid-episode state)."""
    env.reset(seed=seed)
    rng = np.random.default_rng(0)
    a = np.zeros(env.n_actions, dtype=np.float32)
    for _ in range(steps):
        a = 0.9 * a + 0.1 * rng.uniform(env._ctrl_lo, env._ctrl_hi).astype(np.float32)
        _o, _r, term, trunc, _i = env.step(a)
        if term or trunc:
            env.reset(seed=seed)
    return a


def test_restore_is_deterministic():
    """Restoring a snapshot and replaying the SAME action reproduces the exact transition."""
    env = _env()
    _walk_to_contact(env)
    snap = snapshot_planar(env)
    act = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)

    nobs1, r1, t1, tr1, info1 = env.step(act)
    restore_planar(env, snap)
    nobs2, r2, t2, tr2, info2 = env.step(act)

    assert np.allclose(nobs1, nobs2, atol=1e-8), "restore+replay must reproduce next_obs exactly"
    assert r1 == r2 and t1 == t2 and tr1 == tr2
    assert info1["disk_to_zone"] == info2["disk_to_zone"]
    assert info1["both_contact"] == info2["both_contact"]


def test_restore_clears_prior_detour():
    """A detour (step action A) between snapshot and restore must not leak into the restored branch."""
    env = _env()
    _walk_to_contact(env)
    snap = snapshot_planar(env)
    a_detour = np.array([3.0, 3.0, -3.0, -3.0], dtype=np.float32)
    a_probe = np.array([0.2, -0.2, 0.1, -0.1], dtype=np.float32)

    # branch B directly from snap
    restore_planar(env, snap)
    nobs_direct, *_ = env.step(a_probe)
    # now take a detour, then restore and branch B again
    restore_planar(env, snap)
    env.step(a_detour)
    env.step(a_detour)
    restore_planar(env, snap)
    nobs_after_detour, *_ = env.step(a_probe)

    assert np.allclose(nobs_direct, nobs_after_detour, atol=1e-8), "restore must fully clear the detour"


def test_snapshot_does_not_mutate_env():
    """Taking a snapshot leaves the env able to step identically to a no-snapshot control."""
    env_a = _env()
    _walk_to_contact(env_a, seed=123)
    env_b = _env()
    _walk_to_contact(env_b, seed=123)
    act = np.array([0.5, 0.5, -0.5, -0.5], dtype=np.float32)
    _snap = snapshot_planar(env_a)  # should not perturb env_a
    na, *_ = env_a.step(act)
    nb, *_ = env_b.step(act)
    assert np.allclose(na, nb, atol=1e-8)


class _NullActor:
    """Frozen 'policy' returning zeros — deterministic continuation for the branch-return test."""

    def __call__(self, obs):
        n = obs.shape[-2] if obs.ndim == 3 else 1
        return torch.zeros((n, 4), dtype=torch.float32)


def test_branch_component_returns_shape_and_determinism():
    env = _env()
    _walk_to_contact(env)
    snap = snapshot_planar(env)
    so = SearchObjective()
    act = np.array([0.3, -0.3, 0.3, -0.3], dtype=np.float32)

    r1, fs1 = branch_component_returns(env, _NullActor(), snap, act, objective=so, gamma=0.99, horizon=20)
    r2, fs2 = branch_component_returns(env, _NullActor(), snap, act, objective=so, gamma=0.99, horizon=20)

    assert set(r1) == set(COMPONENTS)
    assert all(np.isfinite(v) for v in r1.values())
    assert r1 == r2, "branch returns must be deterministic given the same snapshot + first action"
    assert fs1["next_obs"].shape == (env._n_vertices, 8) if hasattr(env, "_n_vertices") else fs1["next_obs"].ndim == 2
    assert 1 <= fs1["branch_len"] <= 20
    # signed components are non-negative by construction (contact/delivery/approach are indicators/[0,1])
    assert r1["contact"] >= 0.0 and r1["delivery"] >= 0.0 and r1["approach"] >= 0.0


def test_gamma_and_horizon_guards():
    env = _env()
    _walk_to_contact(env)
    snap = snapshot_planar(env)
    so = SearchObjective()
    act = np.zeros(4, dtype=np.float32)
    for bad_gamma in (0.0, -0.1, 1.5):
        try:
            branch_component_returns(env, _NullActor(), snap, act, objective=so, gamma=bad_gamma, horizon=5)
        except ValueError:
            pass
        else:
            raise AssertionError(f"gamma={bad_gamma} must raise")
    try:
        branch_component_returns(env, _NullActor(), snap, act, objective=so, gamma=0.99, horizon=0)
    except ValueError:
        pass
    else:
        raise AssertionError("horizon=0 must raise")
