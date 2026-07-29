"""Balance (foot-support) entropy H_bal — the movement/balance-grounded reward for fast turning.

Locks: H_bal is a normalized entropy in [0, 1] (~1 when the robot stands evenly on 4 feet); the reward
term is off by default (backward-compatible) and adds ``balance_w · H_bal`` when on.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from scenarios.aibo.residual_trot import ResidualTrotConfig, ResidualTrotEnv


def test_balance_w_defaults_off() -> None:
    assert ResidualTrotConfig().balance_w == 0.0


def test_foot_support_entropy_in_unit_range_and_high_when_standing() -> None:
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="leg", obs_mode="flat"), seed=0)
    env.reset(seed=0)
    h = env.foot_support_entropy()
    assert 0.0 <= h <= 1.0
    assert h > 0.8                                          # a settled 4-foot stance is near-uniform support


def test_balance_reward_adds_term() -> None:
    fields = {f.name for f in dataclasses.fields(ResidualTrotConfig)}
    assert "balance_w" in fields
    on = ResidualTrotEnv(ResidualTrotConfig(residual_mode="leg", obs_mode="flat", heading_mode="turn_then_walk",
                                            balance_w=1.0), seed=0)
    off = ResidualTrotEnv(ResidualTrotConfig(residual_mode="leg", obs_mode="flat", heading_mode="turn_then_walk",
                                             balance_w=0.0), seed=0)
    on.reset(seed=1)
    off.reset(seed=1)
    z = np.zeros(12, np.float32)
    _o, r_on, *_ = on.step(z)
    _o2, r_off, *_ = off.step(z)
    assert r_on > r_off                                    # balance term (balance_w·H_bal > 0) lifts the reward
