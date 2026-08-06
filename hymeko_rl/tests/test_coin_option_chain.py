"""Tests for the option chain: reward-env termination boundaries, controller state machine, bank bucketing, classify."""
from __future__ import annotations

import numpy as np

from hymeko_rl.experiments.coin_bridge_relay import _FEAT_IDX, ReadinessDetector
from hymeko_rl.experiments.coin_option_chain import (
    ApproachReward, CaptureReward, OptionChainController, _min_ft_dist,
)
from hymeko_rl.experiments.coin_option_chain_run import _classify
from hymeko_rl.eval.team_tensor import field_index

_I = {n: field_index(n) for n in ("left_contact", "right_contact", "both_contact", "arm_body_contact")}


def _obs(**flags) -> np.ndarray:
    o = np.zeros(41, np.float32)
    for k, v in flags.items():
        o[_I[k]] = 1.0 if v else 0.0
    return o


def test_approach_reward_terminal_dominates() -> None:
    rw = ApproachReward()
    assert rw.r_contact > rw.w_approach and rw.r_contact > rw.w_corridor   # first-contact terminal dominates


def test_capture_reward_terminal_dominates() -> None:
    rw = CaptureReward()
    assert rw.r_ready > rw.w_potential and rw.r_ready > rw.w_bilateral     # first-ready terminal dominates


def test_min_ft_dist_named_fields() -> None:
    o = np.zeros(41, np.float32)
    o[field_index("l_to_coin_x")] = 3.0
    o[field_index("l_to_coin_y")] = 4.0                             # left dist 5
    o[field_index("r_to_coin_x")] = 0.0
    o[field_index("r_to_coin_y")] = 2.0                             # right dist 2 → min
    assert _min_ft_dist(o) == 2.0


class _StubActor:
    """A greedy actor stub returning a constant per-option action so we can read the controller's option choice."""
    def __init__(self, tag: float) -> None:
        self.action_dim, self.action_scale = 6, 1.0
        self._tag = tag

    def action_mean(self, obs):  # torch tensor in, torch tensor out
        import torch
        b = obs.shape[0]
        return torch.full((b, 6), self._tag)


_READY_OBS = _obs(both_contact=True, left_contact=True, right_contact=True)   # a bilateral-contact "ready" state


def _detector() -> ReadinessDetector:
    feats = _READY_OBS[_FEAT_IDX]
    bank = np.stack([feats, feats + 0.01, feats - 0.01]).astype(np.float32)   # ready bank = bilateral-contact features
    return ReadinessDetector(bank, enter_thresh=0.5, exit_thresh=1.0)


def test_controller_starts_in_approach_then_captures_on_contact() -> None:
    ctrl = OptionChainController(_StubActor(0.1), _StubActor(0.2), _StubActor(0.3), _detector())
    from hymeko_rl.experiments.coin_option_chain import ChainLog
    log = ChainLog()
    fn = ctrl.act_fn(log)
    a0 = fn(None, 0, _obs())                                        # no contact → APPROACH (0.1)
    assert np.allclose(a0, 0.1) and log.opt_trace[-1] == "A"
    a1 = fn(None, 1, _obs(left_contact=True))                       # first contact → switches to CAPTURE (0.2)
    assert np.allclose(a1, 0.2) and log.capture_step == 1


def test_controller_handoff_is_sticky_not_readiness_gated() -> None:
    """Once TRANSPORT_READY fires (all-zero obs ≈ the ready bank), hand off; DO NOT fall back when readiness goes false
    (the coin leaves the basin during transport) — only a body shove falls back."""
    ctrl = OptionChainController(_StubActor(0.1), _StubActor(0.2), _StubActor(0.3), _detector())
    from hymeko_rl.experiments.coin_option_chain import ChainLog
    log = ChainLog()
    fn = ctrl.act_fn(log)
    fn(None, 0, _obs(left_contact=True))                            # APPROACH→CAPTURE
    fn(None, 1, _READY_OBS)                                         # matches the ready bank → handoff
    assert log.handoffs == 1
    # readiness now false (coin moved far from the ready bank) but NO body shove → stays TRANSPORT (sticky), no fallback
    far = _obs(left_contact=True, right_contact=True)
    far[field_index("coin_to_target_x")] = 9.0                      # coin far from target (transporting), NO body flag
    a = fn(None, 2, far)
    assert np.allclose(a, 0.3) and log.fallbacks == 0
    # body shove → fall back to CAPTURE
    fn(None, 3, _obs(arm_body_contact=True, both_contact=True))
    assert log.fallbacks == 1


def test_classify_taxonomy() -> None:
    be = {"+0.030-0.045": dict(strict=3, handoff=4, first_contact_rate=0.9)}
    assert _classify(be, dict(transport_alone=0, full_chain=5)) == "CHAIN_POSITIVE"
    be2 = {"+0.018-0.030": dict(strict=1, handoff=2, first_contact_rate=0.9)}
    assert _classify(be2, dict(transport_alone=1, full_chain=2)) == "CAPTURE_POSITIVE"
    be3 = {"+0.018-0.030": dict(strict=0, handoff=0, first_contact_rate=0.7)}
    assert _classify(be3, dict(transport_alone=1, full_chain=0)) == "APPROACH_POSITIVE"
    be4 = {"+0.018-0.030": dict(strict=0, handoff=0, first_contact_rate=0.1)}
    assert _classify(be4, dict(transport_alone=1, full_chain=0)) == "NO_EFFECT"
