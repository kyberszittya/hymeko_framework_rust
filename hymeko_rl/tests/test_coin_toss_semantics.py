"""Coverage for the M0 execution-semantics audit (directive §1): S0 (frozen base) vs S1 (recomputed base) demonstrably
differ when the base tracks the observation, and S1 reproduces the confirmed M0 deployment (recompute base + held r)."""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.experiments.exp_coin_toss_semantics import SemanticPolicy


class _M:
    left_contact = True; right_contact = False; arm_self_contact = False; in_zone = False


class _Env:
    _planar_metrics = _M()


class _ObsDagger:
    """action_mean = first 4 flattened obs dims → base tracks the observation (so S0≠S1 is observable)."""
    def action_mean(self, x):
        return x.reshape(x.shape[0], -1)[:, :4].float()


def _r0(_o):
    return np.array([0.1, 0.0, 0.0, 0.0], np.float32)


def test_s1_recomputes_base_s0_freezes() -> None:
    lo, hi = -np.ones(4, np.float32) * 9, np.ones(4, np.float32) * 9
    s1 = SemanticPolicy(_ObsDagger(), _r0, lo, hi, base_mode="recompute")
    s0 = SemanticPolicy(_ObsDagger(), _r0, lo, hi, base_mode="frozen")
    env = _Env(); a1, a0 = [], []
    for t in range(4):                                           # one chunk (K=4), obs changes each step
        obs = np.full((6, 8), float(t), np.float32)
        a1.append(s1(env, obs)); a0.append(s0(env, obs))
    a1, a0 = np.array(a1), np.array(a0)
    # S1 base tracks obs (grows with t); S0 base frozen at chunk-start obs (t=0)
    assert not np.allclose(a1, a0)                               # the two semantics diverge within a chunk
    assert np.allclose(a0[:, 1:], 0.0)                           # S0 base frozen at obs=0 → coords 1-3 stay 0
    assert a1[3, 1] > a1[0, 1]                                   # S1 base coord grows as obs grows


def test_s1_equals_dagger_plus_held_residual() -> None:
    """S1 (deploy semantics) = dagger(obs_t) + held residual each step — the confirmed M0 rule."""
    lo, hi = -np.ones(4, np.float32) * 9, np.ones(4, np.float32) * 9
    s1 = SemanticPolicy(_ObsDagger(), _r0, lo, hi, base_mode="recompute")
    env = _Env(); obs = np.full((6, 8), 2.0, np.float32)
    out = s1(env, obs)
    assert np.allclose(out, np.array([2.1, 2.0, 2.0, 2.0], np.float32))   # base=2 (obs) + r=[.1,0,0,0]
