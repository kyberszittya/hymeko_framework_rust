"""Audit tests: §1 masked actor loss (gate_t routing) and §5 phase-conditioning update-0 identity."""
import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_phase_conditioning import (
    N_PHASE,
    PHASES,
    assert_phase_actor_is_pi0_at_update0,
    make_phase_actor_from_pi0,
    make_phase_critic,
    phase_onehot,
)
from hymeko_rl.coin_delivery.coin_phase_switched_late import make_late_actor_from_pi0
from hymeko_rl.coin_delivery.coin_td3_contracts import LateTwinCritic
from hymeko_rl.coin_delivery.coin_td3_trainer import masked_actor_loss
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


def _grad(actor):
    return torch.cat([p.grad.flatten() for p in actor.parameters()]).clone()


# ── §1 masked actor loss (gate_t drives pi_late; gate-off rows contribute zero) ──
def test_masked_actor_loss_gate_off_zero_and_invariant():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    late = make_late_actor_from_pi0(pi0, trainable=True); critic = LateTwinCritic()
    obs = torch.randn(8, 48); gate = torch.tensor([1., 1., 0., 0., 1., 0., 0., 1.])
    late.zero_grad(); masked_actor_loss(critic, late, obs, gate).backward(); g1 = _grad(late)
    # (a) changing ONLY gate-off observations cannot change the actor update
    obs2 = obs.clone(); obs2[gate == 0] = torch.randn_like(obs2[gate == 0])
    late.zero_grad(); masked_actor_loss(critic, late, obs2, gate).backward(); g2 = _grad(late)
    assert torch.allclose(g1, g2, atol=1e-6)
    # (b) all-gate-off ⇒ exactly zero pi_late gradient
    late.zero_grad(); masked_actor_loss(critic, late, obs, torch.zeros(8)).backward()
    assert _grad(late).abs().max().item() < 1e-9
    # (c) gate-on rows produce a nonzero gradient
    late.zero_grad(); masked_actor_loss(critic, late, obs, torch.ones(8)).backward()
    assert _grad(late).abs().max().item() > 0


def test_masked_actor_loss_equals_gate_on_mean():
    """The masked form equals the mean over the gate-on subset (the equivalent sampling form used by the runs)."""
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    late = make_late_actor_from_pi0(pi0, trainable=True); critic = LateTwinCritic()
    obs = torch.randn(10, 48); gate = torch.tensor([1., 0., 1., 1., 0., 0., 1., 0., 1., 0.])
    masked = masked_actor_loss(critic, late, obs, gate)
    on = obs[gate == 1.0]
    q1_on, _ = critic(on, late(on)); direct = -q1_on.mean()
    assert torch.allclose(masked, direct, atol=1e-6)


# ── §5 phase conditioning: update-0 identity ──
def test_phase_actor_update0_is_pi0_for_every_phase():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    pa = make_phase_actor_from_pi0(pi0, trainable=True)
    assert_phase_actor_is_pi0_at_update0(pi0, pa, tol=0.0)              # zero phase weights ⇒ exact pi_0 ∀ phase
    assert pa.backbone[0].in_features == 48 + N_PHASE
    assert all(p.requires_grad for p in pa.parameters())


def test_phase_onehot_and_critic_zero_init():
    assert phase_onehot("braking").sum() == 1.0 and len(phase_onehot("braking")) == N_PHASE
    assert phase_onehot("nonexistent").sum() == 0.0
    c = make_phase_critic()
    assert c.q1[0].in_features == 48 + N_PHASE + 4
    for q in (c.q1, c.q2):
        assert torch.count_nonzero(q[0].weight[:, 48:48 + N_PHASE]) == 0    # phase columns zeroed at init


def test_phases_cover_stage1_families():
    for fam in ("target_entry", "braking", "settling_dwell"):
        assert fam in PHASES
