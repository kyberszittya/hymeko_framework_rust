"""Semi-MDP footstep env — one WBC-executed footstep per step, action = bounded foothold residual.

Locks the RL interface for walking: the env constructs, obs is finite and deterministic per seed, the
``action = 0`` scaffold (analytical fixed march) survives many footsteps, a nonzero action measurably
changes the outcome (so there is something to learn), and a fall terminates the episode.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

from scenarios.humanoid.footstep_env import FootstepConfig, HumanoidFootstepEnv  # noqa: E402


def _env(**kw):
    return HumanoidFootstepEnv(FootstepConfig(**kw), seed=0)


def test_constructs_with_finite_obs() -> None:
    e = _env()
    obs, _ = e.reset(seed=0)
    assert obs.shape == e.observation_space.shape and np.all(np.isfinite(obs))
    assert e.action_space.shape == (2,)


def test_reset_is_deterministic_per_seed() -> None:
    e = _env()
    o1, _ = e.reset(seed=3)
    o2, _ = e.reset(seed=3)
    assert np.allclose(o1, o2)


def test_scaffold_survives_many_footsteps() -> None:
    e = _env(max_footsteps=25)
    e.reset(seed=0)
    steps = 0
    for _ in range(25):
        _o, _r, done, trunc, info = e.step(np.zeros(2, np.float32))
        steps = info["steps"]
        if done or trunc:
            break
    assert steps >= 15                                    # a=0 = analytical march is a strong scaffold


def test_action_zero_is_the_scaffold_and_action_changes_outcome() -> None:
    e = _env()
    e.reset(seed=0)
    o0, _r0, _d0, _t0, _i0 = e.step(np.zeros(2, np.float32))
    e.reset(seed=0)
    o1, _r1, _d1, _t1, _i1 = e.step(np.array([1.0, 1.0], np.float32))
    assert not np.allclose(o0, o1, atol=1e-3)             # a large foothold residual moves the state

    e.reset(seed=0)
    o0b, *_ = e.step(np.zeros(2, np.float32))
    assert np.allclose(o0, o0b)                            # a = 0 is reproducible (the scaffold)


def test_reward_is_finite_and_fall_terminates() -> None:
    e = _env(max_footsteps=40)
    e.reset(seed=0)
    saw_done = False
    for _ in range(40):
        # step the swing foot INWARD (toward the walking centreline, narrowing the stance) each footstep —
        # a deliberately destabilising foothold that reliably tips the humanoid
        inward = 1.0 if e._stance == "L" else -1.0
        _o, r, done, trunc, _i = e.step(np.array([0.0, inward], np.float32))
        assert np.isfinite(r)
        if done:
            saw_done = True
            break
        if trunc:
            break
    assert saw_done                                       # a persistent inward foothold falls (episode ends)
