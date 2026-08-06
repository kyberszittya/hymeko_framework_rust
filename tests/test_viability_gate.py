"""Learned viability boundary — logistic fit/predict/save (always) + the balance_env reward gate (mujoco)."""

from __future__ import annotations

import numpy as np
import pytest

from scenarios.humanoid.viability_gate import LearnedViabilityBoundary


def _separable(n: int = 400, seed: int = 0):
    """Viable iff feature 0 (an uprightness proxy) is positive — a linearly separable toy."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] > 0.0).astype(float)
    return x, y


def test_boundary_fits_separable_data() -> None:
    x, y = _separable()
    b = LearnedViabilityBoundary().fit(x, y)
    p = b.predict(x)
    assert ((p > 0.5).astype(float) == y).mean() > 0.9   # learns the boundary
    assert np.all((p >= 0.0) & (p <= 1.0))               # probabilities
    assert b.w[0] > abs(b.w[1]) and b.w[0] > abs(b.w[2])  # feature 0 dominates


def test_boundary_predict_single_and_batch() -> None:
    x, y = _separable(seed=2)
    b = LearnedViabilityBoundary().fit(x, y)
    assert b.predict(x[0]).shape == (1,)                 # a single row is accepted (atleast_2d)
    assert b.predict(x).shape == (x.shape[0],)


def test_boundary_save_roundtrip(tmp_path) -> None:
    x, y = _separable(seed=3)
    b = LearnedViabilityBoundary().fit(x, y)
    path = tmp_path / "viab.npz"
    b.save(path)
    d = np.load(path)
    assert set(d.files) == {"mean", "std", "w", "b"}
    assert d["w"].shape == (3,)


def test_env_gates_forward_reward_by_viability(tmp_path) -> None:
    pytest.importorskip("mujoco")
    from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv

    # A boundary over the 5-D viability state that says "viable when upright" (feature 0 high).
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 5))
    y = (x[:, 0] > 0.0).astype(float)
    path = tmp_path / "viab.npz"
    LearnedViabilityBoundary().fit(x, y).save(path)

    env = HumanoidBalanceEnv(BalanceConfig(torque_action=True, w_velocity=40.0,
                                           viability_boundary=str(path), max_steps=5), seed=0)
    env.reset(seed=0)
    _o, _r, _term, _trunc, info = env.step(np.zeros(env.action_space.shape[0], np.float32))
    assert "viab_gate" in info and 0.0 <= info["viab_gate"] <= 1.0   # the gate is applied, in [0, 1]
