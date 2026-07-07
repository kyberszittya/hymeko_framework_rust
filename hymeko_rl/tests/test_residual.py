"""Invariant tests for the residual-control wrapper (hymeko_rl/env/residual.py).

Residual RL is the floor-preserving path to a learned coin-toss policy: the action is ``clip(base + delta)``
over a fixed scripted controller, so a ZERO delta reproduces the base controller exactly (the learned start
IS the base's performance). That zero-delta invariant is the load-bearing guarantee — if it fails, "residual"
learning silently starts off the base and the floor is lost. These tests pin it, plus the action bound,
passthrough, and the delta-scale precondition. Fakes (no MuJoCo) keep them fast and deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from gymnasium.spaces import Box

from hymeko_rl.env.residual import ResidualControllerEnv


class _FakeEnv:
    """Minimal gym-5-tuple env that records the action it received (so we can assert what the wrapper applied)."""

    def __init__(self, action_low: float = -1.0, action_high: float = 1.0) -> None:
        self.action_space = Box(low=action_low, high=action_high, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3, 4), dtype=np.float32)
        self.n_actions = 2
        self.last_action: "np.ndarray | None" = None
        self.reset_seed: "int | None" = -1
        self.tag = "inner"                                    # for the passthrough test
        self.closed = False

    def reset(self, *, seed: "int | None" = None, **_kw: Any) -> "tuple[np.ndarray, dict]":
        self.reset_seed = seed
        return np.zeros((3, 4), dtype=np.float32), {}

    def step(self, action: np.ndarray) -> "tuple[np.ndarray, float, bool, bool, dict]":
        self.last_action = np.asarray(action, dtype=np.float32)
        return np.zeros((3, 4), dtype=np.float32), 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


class _FakeController:
    """A ``reset()``/``action(env)`` controller returning a fixed base action."""

    def __init__(self, base: np.ndarray) -> None:
        self.base = np.asarray(base, dtype=np.float32)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def action(self, _env: Any) -> np.ndarray:
        return self.base.copy()


def test_zero_delta_reproduces_base_action() -> None:
    inner, ctrl = _FakeEnv(), _FakeController([0.3, -0.4])
    env = ResidualControllerEnv(inner, ctrl, delta_scale=0.15)
    env.reset(seed=7)
    env.step(np.zeros(2, dtype=np.float32))
    assert np.allclose(inner.last_action, [0.3, -0.4])        # THE invariant: zero delta == base controller


def test_delta_adds_then_clips_to_inner_space() -> None:
    inner, ctrl = _FakeEnv(action_low=-1.0, action_high=1.0), _FakeController([0.95, -0.95])
    env = ResidualControllerEnv(inner, ctrl, delta_scale=0.2)
    env.reset()
    env.step(np.array([0.1, -0.1], dtype=np.float32))         # 0.95+0.1=1.05 → clip 1.0; -0.95-0.1=-1.05 → -1.0
    assert np.allclose(inner.last_action, [1.0, -1.0])


def test_action_space_is_bounded_delta() -> None:
    env = ResidualControllerEnv(_FakeEnv(), _FakeController([0.0, 0.0]), delta_scale=0.15)
    assert isinstance(env.action_space, Box)
    assert np.allclose(env.action_space.low, -0.15) and np.allclose(env.action_space.high, 0.15)
    assert env.action_space.shape == (2,)


def test_reset_forwards_seed_and_resets_controller() -> None:
    inner, ctrl = _FakeEnv(), _FakeController([0.0, 0.0])
    env = ResidualControllerEnv(inner, ctrl, delta_scale=0.15)
    env.reset(seed=123)
    assert inner.reset_seed == 123 and ctrl.reset_calls == 1  # seed forwarded, controller re-initialised


def test_attribute_passthrough_to_inner_env() -> None:
    env = ResidualControllerEnv(_FakeEnv(), _FakeController([0.0, 0.0]), delta_scale=0.15)
    assert env.tag == "inner"                                 # __getattr__ forwards unknown attrs (metrics read hg, …)


def test_nonpositive_delta_scale_raises() -> None:
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="delta_scale"):
            ResidualControllerEnv(_FakeEnv(), _FakeController([0.0, 0.0]), delta_scale=bad)


def test_close_forwards_to_inner_env() -> None:
    inner = _FakeEnv()
    env = ResidualControllerEnv(inner, _FakeController([0.0, 0.0]), delta_scale=0.15)
    env.close()
    assert inner.closed
