"""DYNAMIC_PHASE_CURRICULUM_SCOPE_V1 (§3-§5): contact demoted to an orthogonal flag; control_phase is 4-way and
contact-independent; Stage-1 actor mask excludes non-Stage-1 phases (zero gradient)."""
import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_dynamic_phase_scope import (
    CONTACT_FLAGS,
    CONTROL_PHASES,
    N_STATE,
    STAGE1_CONTROL,
    augment_state,
    contact_flag,
    control_phase,
    matching_predicates,
    stage1_actor_trainable,
    state_onehot,
)
from hymeko_rl.coin_delivery.coin_phase_conditioning import make_phase_actor_from_pi0
from hymeko_rl.coin_delivery.coin_td3_contracts import LateTwinCritic
from hymeko_rl.coin_delivery.coin_td3_trainer import masked_actor_loss
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


# §3/§4 — contact is orthogonal; control_phase is 4-way and does NOT collapse to contact_retention
def test_control_phase_is_contact_independent():
    # far, moving, UNILATERAL contact → transport (NOT "contact_retention"); the old detector erased this to contact.
    assert control_phase(dtz=0.20, prev_dtz=0.25, min_dtz=0.20, speed=0.5, prev_speed=0.5, strict=0) == "transport"
    assert control_phase(dtz=0.04, prev_dtz=0.10, min_dtz=0.10, speed=0.4, prev_speed=0.4, strict=0) == "target_entry"
    assert control_phase(dtz=0.01, prev_dtz=0.02, min_dtz=0.01, speed=0.02, prev_speed=0.02, strict=0) == "settling_dwell"
    assert set(CONTROL_PHASES) == {"transport", "target_entry", "braking", "settling_dwell"}
    assert "contact_retention" not in CONTROL_PHASES and "overshoot" not in CONTROL_PHASES


def test_contact_flag_orthogonal():
    assert contact_flag(True, False) == "contact_present" and contact_flag(False, False) == "contact_lost"
    # a unilateral-contact state can still be any control phase (flag coexists, does not replace)
    mp = matching_predicates(dtz=0.20, prev_dtz=0.25, min_dtz=0.20, speed=0.5, prev_speed=0.5, strict=0, lc=True, rc=False)
    assert mp["selected"] == "transport" and mp["contact_present"] and mp["unilateral_contact"]


def test_state_onehot_layout():
    v = state_onehot("braking", "contact_present")
    assert len(v) == N_STATE == 6 and v.sum() == 2.0
    assert v[CONTROL_PHASES.index("braking")] == 1.0 and v[len(CONTROL_PHASES) + CONTACT_FLAGS.index("contact_present")] == 1.0


def test_stage1_actor_trainable_mask():
    assert stage1_actor_trainable(True, "braking") == 1.0
    assert stage1_actor_trainable(True, "transport") == 0.0          # excluded phase
    assert stage1_actor_trainable(False, "braking") == 0.0           # gate off
    assert stage1_actor_trainable(True, "settling_dwell") == 1.0 and stage1_actor_trainable(True, "target_entry") == 1.0


# §5 — excluded-phase transitions contribute EXACTLY zero to the Stage-1 actor update
def test_excluded_phase_zero_actor_gradient():
    pi0 = load_frozen_clip_actor(PI0, freeze=True); pa = make_phase_actor_from_pi0(pi0, trainable=True)
    with torch.no_grad():
        torch.manual_seed(1); pa.backbone[0].weight[:, 48:48 + N_STATE].add_(torch.randn(pa.backbone[0].weight.shape[0], N_STATE))
    critic = LateTwinCritic(obs_dim=48 + N_STATE)
    phases = ["braking", "transport", "settling_dwell", "transport", "target_entry", "transport"]
    obs = torch.randn(6, 48)
    aug = torch.tensor(np.stack([augment_state(obs[i].numpy(), phases[i], "contact_present") for i in range(6)]))
    mask = torch.tensor([stage1_actor_trainable(True, p) for p in phases])   # transport rows → 0
    pa.zero_grad(); masked_actor_loss(critic, pa, aug, mask).backward()
    g1 = torch.cat([p.grad.flatten() for p in pa.parameters()]).clone()
    # change ONLY the excluded (transport) rows' observations → the Stage-1 actor update is unchanged
    obs2 = obs.clone()
    for i, p in enumerate(phases):
        if p == "transport":
            obs2[i] = torch.randn(48)
    aug2 = torch.tensor(np.stack([augment_state(obs2[i].numpy(), phases[i], "contact_present") for i in range(6)]))
    pa.zero_grad(); masked_actor_loss(critic, pa, aug2, mask).backward()
    g2 = torch.cat([p.grad.flatten() for p in pa.parameters()])
    assert torch.allclose(g1, g2, atol=1e-6)
    # all-excluded batch ⇒ exactly zero actor gradient
    allmask = torch.zeros(6)
    pa.zero_grad(); masked_actor_loss(critic, pa, aug, allmask).backward()
    assert torch.cat([p.grad.flatten() for p in pa.parameters()]).abs().max().item() < 1e-9


def test_stage1_control_set():
    assert set(STAGE1_CONTROL) == {"target_entry", "braking", "settling_dwell"}
    assert "transport" not in STAGE1_CONTROL
