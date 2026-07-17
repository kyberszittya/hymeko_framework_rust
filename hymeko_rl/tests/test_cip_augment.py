"""Tests for the CIP counterfactual-augmentation baseline (Cao/Ito) — unit, integration, performance.

Covers the pure helpers (:func:`softmax_rescale`, :func:`estimate_cip_weights`, :func:`counterfactual_swap`),
the :class:`CipReplayAugmentor` strategy (cadence, buffer growth, degenerate/constant-column robustness), the
``ReplayAugmentor`` seam on ``train_sac`` (byte-identical when the strategy no-ops), an end-to-end few-hundred-step
CIP-SAC run on a toy flat env, and the DirectLiNGAM fit-time budget. Run: ``pytest -p no:randomly``.
"""
from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
import pytest

from hymeko_rl.eval.causal.lingam import DirectLiNGAM, sample_linear_sem
from hymeko_rl.eval.cip.cip_augment import (
    CipAugmentConfig,
    CipReplayAugmentor,
    counterfactual_swap,
    estimate_cip_weights,
    softmax_rescale,
)
from hymeko_rl.train.replay import ReplayBuffer


# --------------------------------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------------------------------
def _fill_buffer(buf: ReplayBuffer, n: int, d_s: int, d_a: int, *, seed: int = 0,
                 const_dim: "int | None" = None, reward_of: "int | None" = 0) -> None:
    """Fill ``buf`` with ``n`` random flat transitions; ``const_dim`` (if given) is held constant (zero-padding
    analogue); reward is a linear function of state dim ``reward_of`` plus noise (so it has a real parent)."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, d_s)).astype(np.float32)
    nxt = rng.standard_normal((n, d_s)).astype(np.float32)
    if const_dim is not None:
        obs[:, const_dim] = 0.0
        nxt[:, const_dim] = 0.0
    act = rng.uniform(-1, 1, (n, d_a)).astype(np.float32)
    rew = (obs[:, reward_of] * 2.0 + 0.1 * rng.standard_normal(n)).astype(np.float32) if reward_of is not None \
        else rng.standard_normal(n).astype(np.float32)
    done = np.zeros(n, np.float32)
    buf.add_batch(obs, act, rew, nxt, done)


class _ToyFlatEnv:
    """A minimal gym-5-tuple env with flat obs — enough to drive ``train_sac`` in a test (no MuJoCo)."""

    def __init__(self, obs_dim: int = 6, act_dim: int = 2, horizon: int = 20, seed: int = 0) -> None:
        self._d, self._a, self._h = obs_dim, act_dim, horizon
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._obs = np.zeros(obs_dim, np.float32)
        self.observation_space = _Space((obs_dim,))
        self.action_space = _Space((act_dim,), high=1.0)

    def reset(self, *, seed: "int | None" = None) -> "tuple[np.ndarray, dict[str, Any]]":
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._obs = self._rng.standard_normal(self._d).astype(np.float32)
        return self._obs.copy(), {}

    def step(self, action: np.ndarray) -> "tuple[np.ndarray, float, bool, bool, dict[str, Any]]":
        self._t += 1
        self._obs = (0.9 * self._obs + 0.1 * self._rng.standard_normal(self._d)).astype(np.float32)
        self._obs[: self._a] += 0.05 * np.asarray(action, np.float32)
        reward = float(-np.square(self._obs[0]))          # keep obs[0] near 0
        trunc = self._t >= self._h
        return self._obs.copy(), reward, False, trunc, {"success": 0.0}


class _Space:
    def __init__(self, shape: tuple[int, ...], high: float = 1.0) -> None:
        self.shape = shape
        self.high = np.full(shape, high, np.float32)
        self.low = -self.high


# --------------------------------------------------------------------------------------------------------------
# unit: pure helpers
# --------------------------------------------------------------------------------------------------------------
def test_softmax_rescale_sums_to_dim_and_uniform_is_ones() -> None:
    out = softmax_rescale(np.array([1.0, 2.0, 3.0, 0.5]))
    assert out.shape == (4,)
    assert np.isclose(out.sum(), 4.0)
    assert np.all(out >= 0.0)
    uni = softmax_rescale(np.zeros(5))
    assert np.allclose(uni, np.ones(5))


def test_estimate_weights_recovers_reward_parents_and_lowest_dim() -> None:
    # SEM: s0 -> reward (strong), s1 -> reward (weak); s2 irrelevant. Vars: [s0,s1,s2,a0,reward] (reward idx 4).
    x, _b = sample_linear_sem([(0, 4, 2.0), (1, 4, 0.5)], n_vars=5, n_samples=4000, seed=1, noise="uniform")
    obs, act, rew = x[:, :3], x[:, 3:4], x[:, 4]
    w = estimate_cip_weights(obs, act, rew, lingam=DirectLiNGAM(), n_swap_dims=1)
    assert w.w_s.shape == (3,) and w.w_r.shape == (1,)
    # s0 is the strongest reward parent; the irrelevant s2 is the least important → the swap target.
    assert int(np.argmax(np.abs(w.w_s))) == 0
    assert w.lowest_dims == [2]
    assert not w.degenerate


def test_counterfactual_swap_touches_only_target_dim() -> None:
    rng = np.random.default_rng(3)
    obs = rng.standard_normal((16, 5)).astype(np.float32)
    nxt = rng.standard_normal((16, 5)).astype(np.float32)
    act = rng.standard_normal((16, 2)).astype(np.float32)
    rew = rng.standard_normal(16).astype(np.float32)
    done = np.zeros(16, np.float32)
    s_obs, s_act, s_rew, s_next, s_done = counterfactual_swap(obs, nxt, act, rew, done, [2], rng)
    # action / reward / done unchanged; every column except dim 2 unchanged in obs and next_obs.
    assert np.array_equal(s_act, act) and np.array_equal(s_rew, rew) and np.array_equal(s_done, done)
    for d in range(5):
        same = np.array_equal(s_obs[:, d], obs[:, d]) and np.array_equal(s_next[:, d], nxt[:, d])
        assert same == (d != 2), f"dim {d}: expected {'unchanged' if d != 2 else 'swapped'}"
    # the swapped column is a permutation of the original column (same multiset)
    assert np.array_equal(np.sort(s_obs[:, 2]), np.sort(obs[:, 2]))


def test_estimate_weights_constant_column_does_not_raise_and_is_uncontrollable() -> None:
    # A constant (zero-padding) state column must not crash the fit and must be the lowest-importance dim.
    rng = np.random.default_rng(5)
    n = 3000
    s0 = rng.standard_normal(n)
    s_const = np.zeros(n)
    reward = 1.5 * s0 + 0.1 * rng.standard_normal(n)
    obs = np.column_stack([s0, s_const, rng.standard_normal(n)]).astype(np.float32)
    act = rng.uniform(-1, 1, (n, 2)).astype(np.float32)
    w = estimate_cip_weights(obs, act, reward.astype(np.float32), lingam=DirectLiNGAM(), n_swap_dims=1)
    assert w.lowest_dims == [1]                         # the constant column is uncontrollable


# --------------------------------------------------------------------------------------------------------------
# unit: CipReplayAugmentor cadence + buffer growth + robustness
# --------------------------------------------------------------------------------------------------------------
def test_augmentor_respects_cadence() -> None:
    buf = ReplayBuffer(20_000, (10,), 3)
    _fill_buffer(buf, 3000, 10, 3, seed=7)
    aug = CipReplayAugmentor(10, 3, CipAugmentConfig(refresh_every=1000, sample_n=800, min_buffer=2000, log=False))
    before = buf.size
    aug.maybe_augment(buf, step=999)                    # off-cadence → nothing
    assert aug.n_refresh == 0 and buf.size == before
    aug.maybe_augment(buf, step=1000)                   # on-cadence → one refresh, buffer grew
    assert aug.n_refresh == 1
    assert buf.size == before + min(800, before)
    assert aug.n_augmented == min(800, before)
    assert aug.last is not None and aug.last_fit_ms > 0.0


def test_augmentor_cold_buffer_is_noop() -> None:
    buf = ReplayBuffer(20_000, (10,), 3)
    _fill_buffer(buf, 500, 10, 3, seed=8)               # below min_buffer
    aug = CipReplayAugmentor(10, 3, CipAugmentConfig(refresh_every=100, min_buffer=2000, log=False))
    aug.maybe_augment(buf, step=100)
    assert aug.n_refresh == 0


def test_augmentor_rejects_non_flat_obs() -> None:
    buf = ReplayBuffer(5000, (4, 3), 2)                 # 2-D hypergraph obs
    rng = np.random.default_rng(9)
    buf.add_batch(rng.standard_normal((2500, 4, 3)).astype(np.float32),
                  rng.standard_normal((2500, 2)).astype(np.float32),
                  rng.standard_normal(2500).astype(np.float32),
                  rng.standard_normal((2500, 4, 3)).astype(np.float32),
                  np.zeros(2500, np.float32))
    aug = CipReplayAugmentor(4, 2, CipAugmentConfig(refresh_every=1, min_buffer=2000, log=False))
    with pytest.raises(ValueError, match="flat obs"):
        aug.maybe_augment(buf, step=1)


def test_augmentor_bounded_growth() -> None:
    buf = ReplayBuffer(20_000, (12, ), 4)
    _fill_buffer(buf, 5000, 12, 4, seed=11)
    aug = CipReplayAugmentor(12, 4, CipAugmentConfig(refresh_every=1000, sample_n=1500, min_buffer=2000, log=False))
    aug.maybe_augment(buf, step=1000)
    assert aug.n_augmented <= 1500                      # never more than sample_n per refresh


# --------------------------------------------------------------------------------------------------------------
# unit: the train_sac ReplayAugmentor seam is byte-identical when the strategy no-ops
# --------------------------------------------------------------------------------------------------------------
class _SpyAugmentor:
    """A ReplayAugmentor that records calls but never touches the buffer (the byte-identical control)."""

    def __init__(self) -> None:
        self.calls = 0

    def maybe_augment(self, buf: Any, step: int) -> None:
        self.calls += 1


def _train_toy(augmentor: Any) -> "tuple[list[float], int]":
    import torch

    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    torch.manual_seed(0)
    env = _ToyFlatEnv(obs_dim=6, act_dim=2, horizon=15, seed=0)
    actor, critics = build_sac("mlp", obs_dim=6, flat_dim=6, action_dim=2, action_scale=1.0, hidden=16)
    cfg = SACConfig(total_steps=240, start_steps=60, batch_size=32, eval_every=120, log_every=0, seed=0)
    curve = train_sac(actor, critics, env, cfg, eval_fn=lambda _e, _a: 0.0, augmentor=augmentor)
    return curve, cfg.total_steps


def test_train_sac_seam_byte_identical_when_augmentor_noops() -> None:
    base, steps = _train_toy(None)
    spy = _SpyAugmentor()
    withspy, _ = _train_toy(spy)
    assert withspy == base                              # no-op augmentor must not change the curve
    assert spy.calls == steps                           # seam is wired (called once per env step)


# --------------------------------------------------------------------------------------------------------------
# integration: end-to-end CIP-SAC on the toy env — augmentor fires, buffer grows, no NaN
# --------------------------------------------------------------------------------------------------------------
def test_cip_sac_end_to_end_toy() -> None:
    import torch

    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    torch.manual_seed(0)
    env = _ToyFlatEnv(obs_dim=39, act_dim=4, horizon=25, seed=1)   # MetaWorld-shaped flat obs
    actor, critics = build_sac("mlp", obs_dim=39, flat_dim=39, action_dim=4, action_scale=1.0, hidden=32)
    aug = CipReplayAugmentor(39, 4, CipAugmentConfig(refresh_every=300, sample_n=600, min_buffer=400, seed=0,
                                                     log=False))
    cfg = SACConfig(total_steps=900, start_steps=200, batch_size=64, eval_every=900, log_every=0, seed=0)
    curve = train_sac(actor, critics, env, cfg, eval_fn=lambda _e, _a: 0.0, augmentor=aug)
    assert aug.n_refresh >= 1                            # fired at least once
    assert aug.n_augmented > 0
    assert all(np.isfinite(c) for c in curve)           # no divergence


# --------------------------------------------------------------------------------------------------------------
# performance: DirectLiNGAM fit-time budget at the CIP scale (d≈44)
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sample_n", [1500])
def test_lingam_fit_time_budget(sample_n: int) -> None:
    rng = np.random.default_rng(0)
    d_s, d_a = 39, 4
    obs = rng.standard_normal((sample_n, d_s)).astype(np.float32)
    act = rng.uniform(-1, 1, (sample_n, d_a)).astype(np.float32)
    rew = (obs[:, 0] * 1.5 + 0.2 * rng.standard_normal(sample_n)).astype(np.float32)
    lingam = DirectLiNGAM()
    estimate_cip_weights(obs, act, rew, lingam=lingam, n_swap_dims=1)   # warm-up (JIT/caches)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        estimate_cip_weights(obs, act, rew, lingam=lingam, n_swap_dims=1)
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    worst = max(times)
    print(f"\n[perf] LiNGAM fit d={d_s + d_a + 1} n={sample_n}: "
          f"median={med * 1e3:.0f}ms IQR≈[{min(times) * 1e3:.0f},{worst * 1e3:.0f}]ms worst={worst * 1e3:.0f}ms")
    assert med < 10.0                                   # budget: <10s/fit ⇒ <15min overhead over 1M steps (100 fits)
