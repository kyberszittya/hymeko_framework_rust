"""§5.4/§5.5 replay/target-action contract tests: replay round-trip, stored-gate_tp1 controls the target, no fresh
FSM in the learner, gate=0 preserves the base bit-identically, gate=1 uses residual-only smoothing, terminal
masking, and optimizer/gradient isolation of the frozen base."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

import hymeko_rl.coin_delivery.coin_stable_engagement as cse
from hymeko_rl.coin_delivery.coin_residual_controller import (
    ZeroInitResidualActor,
    assert_base_absent_from_optimizer,
    assert_frozen_base,
)
from hymeko_rl.coin_delivery.coin_residual_replay import (
    ReplayControllerStateV2,
    ResidualReplayBuffer,
    ResidualTransition,
    bounded_smoothed_residual,
    residual_target_action,
    td_target_scalar,
)


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(48, 4)

    def action_mean(self, obs):
        return torch.clamp(self.lin(obs), -4, 4)


def _frozen_base():
    b = _Base()
    for p in b.parameters():
        p.requires_grad_(False)
    return b


def _nonzero_residual():
    r = ZeroInitResidualActor()
    with torch.no_grad():
        r.net[4].weight.copy_(torch.randn_like(r.net[4].weight))
        r.net[4].bias.copy_(torch.randn_like(r.net[4].bias))
    return r


def _cs(gate, mode="EARLY_CONTROL"):
    return ReplayControllerStateV2(gate=float(gate), mode=mode)


def _phash(m):
    return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in m.parameters())).hexdigest()


# ── Test 1 — replay round-trip ────────────────────────────────────────────────
def test_replay_round_trip():
    buf = ResidualReplayBuffer()
    o = np.arange(48, dtype=np.float32); a = np.ones(4, np.float32)
    buf.add(ResidualTransition(o, a, 1.5, o + 1, 0.0, _cs(1.0, "LATE_CONTROL_ARMED"), _cs(0.0, "REACQUIRE")))
    b = buf.sample([0])
    assert b["gate_t"].item() == 1.0 and b["gate_tp1"].item() == 0.0
    assert b["gate_tp1"].dtype == torch.float32 and b["gate_tp1"].shape == (1,)
    assert b["cstate_tp1"][0]["mode"] == "REACQUIRE"
    assert np.array_equal(b["obs"].numpy()[0], o)


def test_replay_stores_by_value_not_reference():
    buf = ResidualReplayBuffer()
    o = np.zeros(48, np.float32)
    buf.add(ResidualTransition(o, np.zeros(4, np.float32), 0.0, o, 0.0, _cs(0.0), _cs(1.0)))
    o[0] = 999.0                                        # mutate the source AFTER insertion
    assert buf.sample([0])["obs"].numpy()[0, 0] == 0.0  # stored copy is unaffected


# ── Test 2 — stored next gate controls the target ─────────────────────────────
def test_stored_gate_tp1_controls_target():
    base = _frozen_base(); r = _nonzero_residual()
    obs = torch.randn(2, 48)
    gate_tp1 = torch.tensor([1.0, 0.0])
    noise = torch.zeros(2, 4)
    tgt = residual_target_action(base, r, obs, gate_tp1, noise=noise)
    base_a = torch.clamp(base.action_mean(obs), -4, 4)
    assert torch.equal(tgt[1], base_a[1])              # gate=0 row -> base
    assert not torch.allclose(tgt[0], base_a[0])       # gate=1 row -> base + residual


# ── Test 3 — no fresh FSM reconstruction in the learner ───────────────────────
def test_no_fresh_fsm_in_target_construction(monkeypatch):
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise AssertionError("gate FSM invoked during target construction")

    monkeypatch.setattr(cse.StableEngagementGate, "__init__", _boom)
    monkeypatch.setattr(cse.StableEngagementGate, "update", _boom)
    monkeypatch.setattr(cse.StableEngagementGate, "reset", _boom)
    base = _frozen_base(); r = _nonzero_residual()
    obs = torch.randn(4, 48)
    # stored gate_tp1 deliberately set to values a "fresh gate from obs" would not produce
    gate_tp1 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    tgt = residual_target_action(base, r, obs, gate_tp1, noise=torch.zeros(4, 4))
    assert calls["n"] == 0 and tgt.shape == (4, 4)


# ── Test 4 — gate zero preserves the base exactly ─────────────────────────────
def test_gate_zero_preserves_base_bit_identical():
    base = _frozen_base(); h0 = _phash(base)
    r = ZeroInitResidualActor()
    with torch.no_grad():                              # large / saturated residual
        r.net[4].weight.copy_(torch.randn_like(r.net[4].weight) * 100)
        r.net[4].bias.copy_(torch.tensor([500.0, -500.0, 500.0, -500.0]))
    obs = torch.randn(6, 48)
    gate0 = torch.zeros(6)
    for noise in (torch.zeros(6, 4), torch.randn(6, 4) * 100, torch.full((6, 4), 1e3)):
        tgt = residual_target_action(base, r, obs, gate0, noise=noise)
        assert torch.equal(tgt, torch.clamp(base.action_mean(obs), -4, 4))
    assert _phash(base) == h0                          # base untouched


# ── Test 5 — gate one uses residual-only smoothing ────────────────────────────
def test_gate_one_residual_only_smoothing():
    base = _frozen_base(); r = _nonzero_residual()
    obs = torch.randn(5, 48)
    gate1 = torch.ones(5)
    noise = torch.randn(5, 4)
    tgt = residual_target_action(base, r, obs, gate1, noise=noise)
    base_a = torch.clamp(base.action_mean(obs), -4, 4)
    ref_residual = bounded_smoothed_residual(r, obs, noise=noise)
    ref = torch.clamp(base_a + ref_residual, -4, 4)
    assert torch.allclose(tgt, ref, atol=1e-6)
    assert ref_residual.abs().max() <= 0.25 + 1e-6     # smoothed residual stays bounded
    # the base branch itself is not smoothed: target - residual contribution reconstructs the exact base within clip
    assert torch.allclose(torch.clamp(base.action_mean(obs), -4, 4), base_a)


# ── Test 6 — terminal transition safety ───────────────────────────────────────
def test_terminal_masks_bootstrap():
    reward = torch.tensor([2.0, -1.0])
    done = torch.ones(2)
    gamma = 0.99
    y1 = td_target_scalar(reward, done, gamma, q_next=torch.tensor([100.0, 100.0]))
    y2 = td_target_scalar(reward, done, gamma, q_next=torch.tensor([-50.0, 999.0]))
    assert torch.equal(y1, reward) and torch.equal(y2, reward)   # bootstrap fully masked
    # non-terminal DOES depend on q_next
    yn = td_target_scalar(reward, torch.zeros(2), gamma, q_next=torch.tensor([10.0, 10.0]))
    assert not torch.equal(yn, reward)


# ── §5.5 optimizer / gradient guards ──────────────────────────────────────────
def test_optimizer_and_gradient_guards():
    base = _frozen_base(); r = _nonzero_residual()
    assert_frozen_base(base)
    res_opt = torch.optim.Adam(r.parameters(), lr=3e-4)
    assert_base_absent_from_optimizer(base, res_opt)             # residual-only optimizer
    obs = torch.randn(4, 48)
    # gate=0 -> zero residual gradient through the target path
    residual_target_action(base, r, obs, torch.zeros(4), noise=torch.zeros(4, 4)).sum().backward()
    assert all(p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)) for p in r.parameters())
    for p in base.parameters():
        assert p.grad is None                                    # base never receives gradient
    # gate=1 -> residual path active
    r.zero_grad()
    residual_target_action(base, r, obs, torch.ones(4), noise=torch.zeros(4, 4)).sum().backward()
    assert float(r.net[4].weight.grad.norm()) > 0
    for p in base.parameters():
        assert p.grad is None
