"""Tests for TRANSACTIONAL_TD3_ACTOR_UPDATE_V1 (Stage-1b): trust region, transactional accept/backtrack/reject with
restore, critic-authorization gate, and the stage-1b pass rule."""
import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_phase_switched_late import make_late_actor_from_pi0
from hymeko_rl.coin_delivery.coin_td3_contracts import LateTwinCritic
from hymeko_rl.coin_delivery.coin_td3_transactional import (
    TransactionalConfig,
    critic_authorization,
    stage1b_gate,
    transactional_actor_step,
    within_trust_region,
)
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


def test_within_trust_region_bounds():
    cfg = TransactionalConfig()
    small = np.full(20, 0.002)
    assert within_trust_region(small, np.full(20, 0.01), cfg)                 # within both step and cum
    assert not within_trust_region(np.full(20, 0.02), np.full(20, 0.01), cfg)  # step too big
    assert not within_trust_region(small, np.full(20, 0.10), cfg)             # cumulative too big


def _params(actor):
    return {k: v.clone() for k, v in actor.state_dict().items()}


def _same(pa, pb):
    return all(torch.equal(pa[k], pb[k]) for k in pa)


def test_transactional_reject_restores_actor_and_optimizer():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    late = make_late_actor_from_pi0(pi0, trainable=True)
    critic = LateTwinCritic()
    opt = torch.optim.Adam(late.parameters(), lr=1.0)                         # huge lr ⇒ step blows the trust region
    cfg = TransactionalConfig()
    obs = torch.randn(16, 48); anchor = torch.randn(12, 48)
    a0 = torch.clamp(pi0.action_mean(anchor), -4, 4).detach()
    before = _params(late)
    r = transactional_actor_step(late, opt, critic, pi0, obs, obs, anchor, a0, cfg)
    assert r["outcome"] == "rejected"                                         # no scale can satisfy the region
    assert _same(before, _params(late))                                       # actor fully restored


def test_transactional_accept_small_step():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    late = make_late_actor_from_pi0(pi0, trainable=True)
    critic = LateTwinCritic()
    opt = torch.optim.Adam(late.parameters(), lr=1e-7)                        # tiny lr ⇒ within trust region
    cfg = TransactionalConfig()
    obs = torch.randn(16, 48); anchor = torch.randn(12, 48)
    a0 = torch.clamp(pi0.action_mean(anchor), -4, 4).detach()
    before = _params(late)
    r = transactional_actor_step(late, opt, critic, pi0, obs, obs, anchor, a0, cfg)
    assert r["outcome"] == "accepted" and r["scale"] == 1.0
    assert not _same(before, _params(late))                                   # actor moved (slightly)


def test_critic_authorization_rejects_nonfinite():
    class NanCritic(torch.nn.Module):
        def forward(self, o, a):
            return torch.full((o.shape[0],), float("nan")), torch.zeros(o.shape[0])
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    late = make_late_actor_from_pi0(pi0, trainable=True)
    checks = critic_authorization(NanCritic(), late, torch.randn(10, 48), TransactionalConfig())
    assert checks["finite_Q"] is False and checks["authorized"] is False


def test_critic_authorization_shape_and_keys():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    late = make_late_actor_from_pi0(pi0, trainable=True)
    checks = critic_authorization(LateTwinCritic(), late, torch.randn(10, 48), TransactionalConfig())
    for k in ("finite_Q", "finite_grad", "boundary_ok", "twin_ok", "perturb_ok", "authorized"):
        assert k in checks
    assert isinstance(checks["authorized"], bool)


def test_stage1b_gate_match_passes_degradation_fails():
    cfg = TransactionalConfig()
    match = {"auth": {"authorized": True}, "anchor_cum_max": 0.01,
             "delta_vs_pi0": {"strict_success": 0.0, "max_dwell": 0.0, "exited": 0.0, "contact_retention": 0.0}}
    degrade = {"auth": {"authorized": True}, "anchor_cum_max": 0.01,
               "delta_vs_pi0": {"strict_success": -0.1, "max_dwell": 0.0, "exited": 0.0, "contact_retention": 0.0}}
    unauth = {"auth": {"authorized": False}, "anchor_cum_max": 0.01,
              "delta_vs_pi0": {"strict_success": 0.0, "max_dwell": 0.0, "exited": 0.0, "contact_retention": 0.0}}
    assert stage1b_gate(match, match, cfg) is True                            # beat-or-match, authorized, TR ok
    assert stage1b_gate(match, degrade, cfg) is False                         # one checkpoint degrades
    assert stage1b_gate(match, unauth, cfg) is False                          # not authorized
