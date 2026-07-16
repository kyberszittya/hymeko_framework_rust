"""Tests for the Aibo walking extension — the `QuadrupedTrotGait` scripted demonstrator on the restored
22-DOF Aibo goal-reach task. Locks that the gait produces *forward*, upright, finite locomotion (a slow but
stable walk; RL is the lever for real speed, per the standing-DAgger lesson).

Run: pytest -p no:randomly hymeko_rl/tests/test_aibo_walk.py"""
from __future__ import annotations

import numpy as np

from hymeko_rl.env.locomotion_experts import QuadrupedTrotGait
from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv


def test_aibo_is_the_22dof_plant() -> None:
    """The restored plant is the faithful Aibo: 12 leg actuators + the expressive DOF held passive."""
    env = QuadrupedGoalEnv(base="free", task="goal", max_steps=50)
    assert env.n_actions == 12                       # 4 legs × (abduct, flex, knee)
    assert env.model.nbody >= 20                     # full ERS-1000 body (legs + head/neck/waist/ears/tail)


def test_trot_gait_shape_and_bounds() -> None:
    env = QuadrupedGoalEnv(base="free", task="goal", max_steps=50)
    env.reset(seed=0)
    a = QuadrupedTrotGait().action(env)
    assert a.shape == (env.n_actions,) and a.dtype == np.float32
    assert np.all(np.abs(a) <= 1.0 + 1e-6)


def test_trot_gait_walks_forward_upright() -> None:
    """The PD-trot moves the Aibo forward (+x, toward the goal) while staying upright and finite — a real if
    slow walk, not the backward drift a naive open-loop CPG produced."""
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=3.0, max_steps=700)
    env.reset(seed=0)
    gait = QuadrupedTrotGait()
    x0 = float(env.data.xpos[env.torso, 0])
    minup = 1.0
    for _ in range(700):
        _, _, term, trunc, _ = env.step(gait.action(env))
        minup = min(minup, env._torso_uprightness())
        assert np.isfinite(env.data.qacc).all()
        if term or trunc:
            break
    dx = float(env.data.xpos[env.torso, 0]) - x0
    assert dx > 0.2, f"trot should walk forward (dx={dx:+.2f} m)"
    assert minup > 0.5, f"trot should stay upright (min upright {minup:.2f})"


def test_cip_vertical_bounce_reward_and_env() -> None:
    """The CIP-informed `vertical_bounce` term is registered and penalizes the torso's vertical speed (the
    bounce channel DirectLiNGAM found); `CipAiboEnv` exposes it in its reward + a privileged state."""
    from hymeko_rl.env.reward import _REWARD_TERMS
    from hymeko_rl.experiments.exp_aibo_cip_walk import CIP_GOAL_REWARD, CipAiboEnv

    assert "vertical_bounce" in _REWARD_TERMS
    assert "vertical_bounce" in [k for k, _ in CIP_GOAL_REWARD.terms]
    env = CipAiboEnv(max_steps=50)
    env.reset(seed=0)
    z = env.privileged_state()
    assert z.ndim == 1 and np.isfinite(z).all()
    term = _REWARD_TERMS["vertical_bounce"]
    _ = term(env, 0.0, np.zeros(env.n_actions))          # caches the free-base z dof
    assert env._base_vz_dof is not None and env._base_vz_dof >= 0
    env.data.qvel[env._base_vz_dof] = 2.0                 # inject upward bounce
    assert term(env, 0.0, np.zeros(env.n_actions)) < 0.0


def test_trot_gait_deterministic() -> None:
    env = QuadrupedGoalEnv(base="free", task="goal", max_steps=50)
    env.reset(seed=0)
    for _ in range(5):
        env.step(QuadrupedTrotGait().action(env))
    e2 = QuadrupedGoalEnv(base="free", task="goal", max_steps=50)
    e2.reset(seed=0)
    for _ in range(5):
        e2.step(QuadrupedTrotGait().action(e2))
    assert np.allclose(QuadrupedTrotGait().action(env), QuadrupedTrotGait().action(e2))
