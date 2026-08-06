"""Tests for the Hamiltonian-momenta + Lyapunov reward terms (underactuated locomotion, 2026-07-17).

The terms read MuJoCo-native centroidal momenta + energy and are normalised to O(1) so the reward weights (not
raw physical units) set their balance. These pin the physics (correct signs), the O(1) scaling (so the full
stack doesn't have one term dominate), the per-step cache (all terms share one compute), graceful 0 on a
fixed-base env, and `.hymeko`-declarability (registered in `_REWARD_TERMS`)."""
from __future__ import annotations

import mujoco
import numpy as np

from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.env.reward import _REWARD_TERMS, RewardSpec, _centroidal
from hymeko_rl.experiments.exp_aibo_cip_walk import CipAiboEnv

_HAM = ["forward_momentum", "transverse_momentum", "centroidal_angular_momentum",
        "energy_regulation", "capture_point"]


def _aibo() -> CipAiboEnv:
    env = CipAiboEnv(goal_distance=4.0, max_steps=300, bounce=3.0)
    env.reset(seed=0)
    for _ in range(15):
        env.step(np.full(env.n_actions, 0.3, np.float32))
    return env


def test_terms_registered_and_hymeko_declarable() -> None:
    """All 5 terms are in `_REWARD_TERMS` (so any `.hymeko` reward_spec can declare them) and a full-stack
    RewardSpec constructs + evaluates to a finite scalar."""
    for name in _HAM:
        assert name in _REWARD_TERMS
    spec = RewardSpec((("forward_momentum", 1.0), ("transverse_momentum", 0.5),
                       ("centroidal_angular_momentum", 0.3), ("energy_regulation", 0.2),
                       ("capture_point", 0.4), ("alive", 1.0)))
    r = spec.evaluate(_aibo(), 0.0, np.zeros(12, np.float32))
    assert np.isfinite(r)


def test_terms_finite_commensurate_and_penalties_nonpositive() -> None:
    """The terms are finite and O(1)-commensurate (no unnormalised term dominates — the full-stack fight guard),
    and the four Lyapunov/penalty terms are ≤ 0."""
    env = _aibo()
    a = np.zeros(env.n_actions, np.float32)
    vals = {n: _REWARD_TERMS[n](env, 0.0, a) for n in _HAM}
    assert all(np.isfinite(v) for v in vals.values())
    assert all(abs(v) < 50 for v in vals.values()), f"a term is not O(1): {vals}"
    for pen in ("transverse_momentum", "centroidal_angular_momentum", "energy_regulation", "capture_point"):
        assert vals[pen] <= 1e-9


def test_physics_signs_forward_vs_transverse() -> None:
    """Injected forward COM velocity → forward_momentum > 0; injected lateral velocity → transverse < 0 and
    capture_point (lateral DCM) < 0. This is the load-bearing physics contract."""
    env = _aibo()
    a = np.zeros(env.n_actions, np.float32)
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = 1.0                                   # free-base +x (forward)
    mujoco.mj_forward(env.model, env.data)
    env._centroidal_c = None
    assert _REWARD_TERMS["forward_momentum"](env, 0.0, a) > 0.0
    env.data.qvel[0] = 0.0
    env.data.qvel[1] = 1.0                                   # free-base +y (lateral)
    mujoco.mj_forward(env.model, env.data)
    env._centroidal_c = None
    assert _REWARD_TERMS["transverse_momentum"](env, 0.0, a) < 0.0
    assert _REWARD_TERMS["capture_point"](env, 0.0, a) < 0.0


def test_per_step_cache_shared() -> None:
    """All Hamiltonian terms in one reward evaluation share a single centroidal compute (cached on data.time)."""
    env = _aibo()
    assert _centroidal(env) is _centroidal(env)
    env.step(np.zeros(env.n_actions, np.float32))            # advances data.time → cache invalidates
    assert _centroidal(env) is _centroidal(env)


def test_graceful_zero_on_fixed_base_env() -> None:
    """On an env with no free base (cartpole), every Hamiltonian term returns exactly 0.0 — the terms are
    env-agnostic like the rest of the registry."""
    cp = InvertedPendulumEnv(mjcf=emit_cartpole_mjcf())
    cp.reset(seed=0)
    a = np.zeros(1, np.float32)
    assert all(_REWARD_TERMS[n](cp, 0.0, a) == 0.0 for n in _HAM)
