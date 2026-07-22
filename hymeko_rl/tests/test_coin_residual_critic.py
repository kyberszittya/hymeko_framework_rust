"""§6.12 critic/encoder contract tests: encoder determinism/fingerprint, distinct modes → distinct encodings,
composite-action input, twin independence, frozen-base isolation during critic fitting."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.coin_residual_controller import assert_base_absent_from_optimizer, assert_frozen_base
from hymeko_rl.coin_delivery.coin_residual_critic import (
    ENCODER_DIM,
    CompositeTwinCritic,
    encode_controller_state,
    encode_controller_states,
    encoder_fingerprint,
)


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(48, 4)

    def action_mean(self, obs):
        return torch.clamp(self.lin(obs), -4, 4)


def _cs(gate=1.0, mode="LATE_CONTROL_ARMED", bc=0, uc=6, side="R", lc=0):
    return {"gate": gate, "mode": mode, "bilateral_counter": bc, "uni_counter": uc, "uni_side": side,
            "loss_counter": lc}


def test_encoder_determinism():
    c = _cs()
    assert np.array_equal(encode_controller_state(c), encode_controller_state(dict(c)))


def test_encoder_fingerprint_stable_and_dim():
    assert len(encoder_fingerprint()) == 64
    assert encode_controller_state(_cs()).shape == (ENCODER_DIM,)


def test_distinct_modes_distinct_encoding():
    early = encode_controller_state(_cs(gate=0.0, mode="EARLY_CONTROL", uc=0, side=None))
    armed = encode_controller_state(_cs(gate=1.0, mode="LATE_CONTROL_ARMED"))
    reacq = encode_controller_state(_cs(gate=0.0, mode="REACQUIRE"))
    assert not np.array_equal(early, armed)
    assert not np.array_equal(armed, reacq)
    assert not np.array_equal(early, reacq)


def test_distinct_sides_distinct_encoding():
    left = encode_controller_state(_cs(side="L"))
    right = encode_controller_state(_cs(side="R"))
    assert not np.array_equal(left, right)


def test_critic_consumes_composite_action_dim4():
    crit = CompositeTwinCritic()
    obs = torch.randn(5, 48); act = torch.randn(5, 4)
    enc = encode_controller_states([_cs() for _ in range(5)])
    q1, q2 = crit(obs, act, enc)
    assert q1.shape == (5,) and q2.shape == (5,)


def test_twin_independent_parameters():
    crit = CompositeTwinCritic()
    ids1 = {id(p) for p in crit.q1.parameters()}
    ids2 = {id(p) for p in crit.q2.parameters()}
    assert ids1.isdisjoint(ids2)


def test_frozen_base_isolated_during_critic_fit():
    base = _Base()
    for p in base.parameters():
        p.requires_grad_(False)
    assert_frozen_base(base)
    crit = CompositeTwinCritic()
    opt = torch.optim.Adam(crit.parameters(), lr=1e-3)
    assert_base_absent_from_optimizer(base, opt)
    obs = torch.randn(8, 48)
    with torch.no_grad():
        act = torch.clamp(base.action_mean(obs), -4, 4)
    enc = encode_controller_states([_cs() for _ in range(8)])
    q1, q2 = crit(obs, act, enc)
    y = torch.zeros(8)
    loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    for p in base.parameters():
        assert p.grad is None                       # base never receives gradient


def test_encoder_only_stored_fields():
    # extra unrelated keys must not change the encoding (no leakage of target/success/future if present)
    c = _cs()
    c_leak = dict(c, disk_to_zone=0.001, success=True, future_obs=[1, 2, 3])
    assert np.array_equal(encode_controller_state(c), encode_controller_state(c_leak))
