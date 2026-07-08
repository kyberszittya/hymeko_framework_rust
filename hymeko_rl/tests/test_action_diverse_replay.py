"""Action-diverse replay: schema, action diversity (the whole point), phase-aware perturbation, determinism."""
from __future__ import annotations

import numpy as np

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.train.action_diverse_replay import DiverseReplayConfig, generate_action_diverse_replay
from hymeko_rl.train.search_objective import COMPONENTS


def _env_actor():
    env = PlanarGraspEnv(robot=None, max_steps=120, difficulty=0.3)
    actor = build_collaborative_offpolicy(env, kind="mlp", hidden=64)[0]  # random weights suffice for the mechanics
    actor.eval()
    return env, actor


def _cfg(**kw):
    base = dict(n_visit_episodes=3, max_steps=120, n_targets=30, branch_horizon=25, log_every=0, seed=0)
    base.update(kw)
    return DiverseReplayConfig(**base)


def test_schema_and_shapes():
    env, actor = _env_actor()
    rep = generate_action_diverse_replay(env, actor, _cfg())
    d = rep.data
    n = d["obs"].shape[0]
    assert d["obs"].shape == (n, env._n_vertices, 8) if hasattr(env, "_n_vertices") else d["obs"].ndim == 3
    assert d["action"].shape == (n, env.n_actions)
    assert d["z"].shape == (n, 5)
    for c in COMPONENTS:
        assert d[f"mc_r_{c}"].shape == (n,)
        assert np.all(np.isfinite(d[f"mc_r_{c}"]))
    # provably-signed components are non-negative
    for c in ("approach", "contact", "delivery", "progress"):
        assert d[f"mc_r_{c}"].min() >= -1e-6, c
    assert set(d["is_ood"].tolist()) <= {0.0, 1.0}
    assert int(d["is_ood"].sum()) == rep.n_targets * _cfg().n_ood_per_state


def test_action_diversity_present():
    """The entire fix: stored actions must vary (the prior replay was near-deterministic)."""
    env, actor = _env_actor()
    rep = generate_action_diverse_replay(env, actor, _cfg(n_ood_per_state=1))
    assert np.all(rep.data["action"].std(axis=0) > 0.05), "actions must vary across rows (OOD + perturbations)"


def test_approach_perturbation_is_zero():
    """APPROACH must be left undisturbed (approach_scale=0): non-OOD approach rows have perturb_norm 0."""
    env, actor = _env_actor()
    rep = generate_action_diverse_replay(env, actor, _cfg())
    d = rep.data
    approach = (d["phase_id"] == 0) & (d["is_ood"] < 0.5)
    if approach.any():
        assert np.allclose(d["perturb_norm"][approach], 0.0, atol=1e-6)


def test_determinism():
    env, actor = _env_actor()
    r1 = generate_action_diverse_replay(env, actor, _cfg())
    env2, _ = _env_actor()
    r2 = generate_action_diverse_replay(env2, actor, _cfg())
    assert r1.data["obs"].shape == r2.data["obs"].shape
    assert np.allclose(r1.data["action"], r2.data["action"], atol=1e-6)
    assert np.allclose(r1.data["mc_r_contact"], r2.data["mc_r_contact"], atol=1e-6)


def test_probe_pool_engaged_only():
    env, actor = _env_actor()
    rep = generate_action_diverse_replay(env, actor, _cfg())
    assert all(s.phase in ("CONTACT", "PUSH", "DELIVERY") for s in rep.probe_pool)
