"""Tests for the SB3-style early-concat flat critic (the Coffee-Push reach-then-regress fix). ``pytest -p no:randomly``."""
from __future__ import annotations

import torch

from hymeko_rl.train.flat_critic import EarlyConcatCritic, build_flat_sac


def test_early_concat_critic_shape() -> None:
    c = EarlyConcatCritic(39, 4, 256)
    q = c(torch.randn(8, 39), torch.randn(8, 4))
    assert q.shape == (8,)                                   # matches QCritic's (B,) output


def test_early_concat_critic_discriminates_actions() -> None:
    """The whole point: Q must be sensitive to the action (early fusion) — dQ/da is nonzero and varies with a."""
    torch.manual_seed(0)
    c = EarlyConcatCritic(6, 2, 64)
    obs = torch.randn(1, 6)
    a = torch.zeros(1, 2, requires_grad=True)
    q = c(obs, a)
    q.backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 1e-6   # dQ/da nonzero
    # Q changes when the action changes (same state)
    with torch.no_grad():
        q0 = c(obs, torch.full((1, 2), -0.9))
        q1 = c(obs, torch.full((1, 2), 0.9))
    assert abs(float(q0 - q1)) > 1e-4


def test_build_flat_sac_wires_early_concat_critics() -> None:
    actor, critics = build_flat_sac(39, 4, action_scale=1.0, hidden=128, n_critics=2)
    assert len(critics) == 2 and all(isinstance(c, EarlyConcatCritic) for c in critics)
    obs = torch.randn(4, 39)
    with torch.no_grad():
        a = actor.action_mean(obs)
        q = critics[0](obs, a)
    assert a.shape == (4, 4) and q.shape == (4,)             # drop-in for train_sac (c(s,a) -> (B,))
