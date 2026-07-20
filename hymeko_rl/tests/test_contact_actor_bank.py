"""Tests for the two-actor contact bank (F21): the HYMeko_CONTACT_MODE selector + ContactActorBank routing.

Certifies: the mode gate is an explicit named-field function (TRANSPORT on clean bilateral, REPOSITION otherwise, with
prev-frame hysteresis); the bank routes per-sample so each head trains only on its mode's states (both heads receive
gradient); the bank satisfies the shared actor interface so train_sac drives it; param overhead < 15%; and the
bank+strategy validation is loud.
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.eval.team_tensor import field_index
from hymeko_rl.train.rl_config import (
    ContactModeReason, CriticMode, PolicyKind, Strategy, UnsupportedRLConfig, select_contact_mode, validate_rl_config,
)
from hymeko_rl.train.sac import ContactActorBank, SACConfig, build_sac, train_sac

_OBS, _ACT = 41, 6
_I = {n: field_index(n) for n in ("left_contact", "right_contact", "both_contact", "arm_body_contact",
                                  "contact_lost_after_handoff", "prev_left_contact", "prev_right_contact")}


def _obs_row(**flags) -> torch.Tensor:
    o = torch.zeros(1, _OBS)
    for name, v in flags.items():
        o[0, _I[name]] = 1.0 if v else 0.0
    return o


def _reason(o: torch.Tensor) -> ContactModeReason:
    _, r = select_contact_mode(o)
    return ContactModeReason(int(r[0]))


# ── the explicit named-field mode gate ───────────────────────────────────────────────────────────────────────────
def test_transport_when_clean_bilateral() -> None:
    o = _obs_row(left_contact=True, right_contact=True, both_contact=True)
    t, _ = select_contact_mode(o)
    assert bool(t[0]) and _reason(o) is ContactModeReason.TRANSPORT_VALID_BILATERAL


def test_body_shove_forces_reposition_even_if_bilateral() -> None:
    o = _obs_row(left_contact=True, right_contact=True, both_contact=True, arm_body_contact=True)
    t, _ = select_contact_mode(o)
    assert not bool(t[0]) and _reason(o) is ContactModeReason.REPOSITION_BODY_SHOVE


def test_reposition_reasons() -> None:
    assert _reason(_obs_row()) is ContactModeReason.REPOSITION_NO_CONTACT
    assert _reason(_obs_row(left_contact=True)) is ContactModeReason.REPOSITION_ONE_SIDED
    assert _reason(_obs_row(contact_lost_after_handoff=True)) is ContactModeReason.REPOSITION_LOST


def test_lost_latch_does_not_bar_transport_re_entry() -> None:
    """The latched contact_lost flag must NOT gate transport — re-established bilateral contact returns to TRANSPORT."""
    o = _obs_row(left_contact=True, right_contact=True, both_contact=True, contact_lost_after_handoff=True)
    t, _ = select_contact_mode(o)
    assert bool(t[0])                                                # clean bilateral wins over the latch


def test_hysteresis_holds_transport_through_one_frame_flicker() -> None:
    o = _obs_row(left_contact=True, right_contact=False, prev_left_contact=True, prev_right_contact=True)
    t, _ = select_contact_mode(o)
    assert bool(t[0]) and _reason(o) is ContactModeReason.TRANSPORT_HYSTERESIS_HOLD


def test_select_contact_mode_batched() -> None:
    o = torch.zeros(3, _OBS)
    o[0, [_I["left_contact"], _I["right_contact"], _I["both_contact"]]] = 1.0    # transport
    o[1, _I["left_contact"]] = 1.0                                                # one-sided
    #  o[2] all-zero → no contact
    t, r = select_contact_mode(o)
    assert t.tolist() == [True, False, False]
    assert r.shape == (3,)


# ── the bank ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _bank() -> ContactActorBank:
    torch.manual_seed(0)
    actor, _ = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0,
                         actor_head="contact_bank", hidden=16)
    assert isinstance(actor, ContactActorBank)
    return actor


def test_bank_param_overhead_under_15pct() -> None:
    """Capacity control (§6) at the PRODUCTION width (hidden=64) — where the shared encoder dominates and the two tiny
    mode heads add ~10%. (At toy widths the heads are a larger fraction; the gate is on the real experiment config.)"""
    torch.manual_seed(0)
    f11, _ = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0, hidden=64)
    f21, _ = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0,
                       actor_head="contact_bank", hidden=64)
    n11 = sum(p.numel() for p in f11.parameters())
    n21 = sum(p.numel() for p in f21.parameters())
    assert (n21 - n11) / n11 < 0.15, f"overhead {(n21 - n11) / n11:.1%}"   # shared encoder + tiny heads → ~10%


def _mixed_batch(n: int = 8) -> torch.Tensor:
    o = torch.randn(n, _OBS) * 0.1
    half = n // 2
    o[:half, [_I["left_contact"], _I["right_contact"], _I["both_contact"]]] = 1.0    # TRANSPORT
    o[half:, _I["left_contact"]] = 1.0                                               # REPOSITION (one-sided)
    o[half:, [_I["right_contact"], _I["both_contact"]]] = 0.0
    return o


def test_bank_routes_and_both_heads_get_gradient() -> None:
    bank = _bank()
    o = _mixed_batch(8)
    a, lp = bank.sample(o)
    assert a.shape == (8, _ACT) and lp.shape == (8,)
    (a.pow(2).sum() - lp.sum()).backward()
    assert bank.transport.mu.weight.grad.abs().sum() > 0            # TRANSPORT states trained the TRANSPORT head
    assert bank.reposition.mu.weight.grad.abs().sum() > 0           # REPOSITION states trained the REPOSITION head


def test_bank_action_mean_bounded_and_deterministic() -> None:
    bank = _bank()
    o = _mixed_batch(6)
    m1, m2 = bank.action_mean(o), bank.action_mean(o)
    assert m1.shape == (6, _ACT) and m1.abs().max() <= 1.0
    assert torch.equal(m1, m2)                                       # deterministic (greedy)


def test_bank_diagnostics_track_occupancy() -> None:
    bank = _bank()
    bank.action_mean(_mixed_batch(8))                               # 4 transport, 4 reposition
    d = bank.pop_diagnostics()
    assert d["samples_transport"] == 4 and d["samples_reposition"] == 4
    assert d["mode_occupancy_transport"] == 0.5
    assert bank.pop_diagnostics()["samples_transport"] == 0         # consumed (reset)


class _ModeToyEnv:
    """41-dim env that drives both contact modes so train_sac exercises both heads (contact fields cycle)."""

    class _Space:
        def __init__(self, shape): self.shape = shape

    def __init__(self, max_steps: int = 40) -> None:
        self.observation_space = self._Space((_OBS,))
        self.action_space = self._Space((_ACT,))
        self.max_steps = max_steps
        self._t = 0

    def _obs(self) -> np.ndarray:
        o = np.zeros(_OBS, np.float32)
        if self._t % 3 == 0:                                       # bilateral → TRANSPORT
            o[[_I["left_contact"], _I["right_contact"], _I["both_contact"]]] = 1.0
        elif self._t % 3 == 1:                                     # one-sided → REPOSITION
            o[_I["left_contact"]] = 1.0
        return o                                                    # else no contact → REPOSITION

    def reset(self, *, seed=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        return self._obs(), -float(np.mean(np.square(action))), False, self._t >= self.max_steps, {}


def test_bank_trains_end_to_end_both_heads_change() -> None:
    bank = _bank()
    before = {k: v.detach().clone() for k, v in bank.state_dict().items()}
    _, critics = build_sac("mlp", obs_dim=_OBS, flat_dim=_OBS, action_dim=_ACT, action_scale=1.0, hidden=16)
    hist = train_sac(bank, critics, _ModeToyEnv(), SACConfig(total_steps=240, start_steps=40, batch_size=16,
                     capacity=500, eval_every=1000, log_every=0, n_eval=1, seed=0), eval_fn=lambda *_: 0.0)
    assert all(np.isfinite(hist))
    tr_changed = not torch.equal(before["transport.mu.weight"], bank.state_dict()["transport.mu.weight"])
    rp_changed = not torch.equal(before["reposition.mu.weight"], bank.state_dict()["reposition.mu.weight"])
    assert tr_changed and rp_changed                               # both heads actually updated


def test_bank_requires_contact_mode_strategy() -> None:
    validate_rl_config(PolicyKind.SAC_CONTACT_ACTOR_BANK, Strategy.HYMEKO_CONTACT_MODE, CriticMode.TASK_ONLY)  # ok
    try:
        validate_rl_config(PolicyKind.SAC_CONTACT_ACTOR_BANK, Strategy.DIRECT, CriticMode.TASK_ONLY)
        raise AssertionError("expected UnsupportedRLConfig")
    except UnsupportedRLConfig as e:
        assert "requires the HYMeko_CONTACT_MODE" in str(e)
