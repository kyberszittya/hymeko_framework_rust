"""TRANSPORT_TO_DWELL_TD3_BASELINE_V1 ontology + conditioning tests (no training)."""
import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.coin_phase_conditioning import make_phase_actor_from_pi0
from hymeko_rl.coin_delivery.coin_transport_dwell import (
    BANK_MIN,
    CONTACT_FLAGS,
    CONTROL_MODES,
    N_COND,
    SAMPLE_TARGET,
    actor_trainable,
    augment_td,
    control_mode,
    state_vector,
)
from hymeko_rl.coin_delivery.coin_td3_contracts import LateTwinCritic
from hymeko_rl.coin_delivery.coin_td3_trainer import masked_actor_loss
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


def test_control_modes_exclude_target_entry():
    assert set(CONTROL_MODES) == {"transport", "braking", "settling_dwell"}
    assert "target_entry" not in CONTROL_MODES
    assert control_mode(dtz=0.04, speed=0.4, prev_speed=0.4, strict=0) == "transport"      # entry state is NOT a mode
    assert control_mode(dtz=0.03, speed=0.1, prev_speed=0.4, strict=0) == "braking"
    assert control_mode(dtz=0.01, speed=0.02, prev_speed=0.02, strict=0) == "settling_dwell"


def test_state_vector_layout():
    ev = np.array([1, 0, 0, 0.03, -0.01], np.float32)
    v = state_vector("braking", "contact_present", ev)
    assert len(v) == N_COND == 10
    assert v[CONTROL_MODES.index("braking")] == 1.0
    assert v[len(CONTROL_MODES) + CONTACT_FLAGS.index("contact_present")] == 1.0
    assert np.allclose(v[5:], ev)                                   # 5 event features tail
    assert augment_td(np.zeros(48), "transport", "contact_lost", ev).shape == (58,)


def test_actor_trainable_all_modes():
    for m in CONTROL_MODES:
        assert actor_trainable(True, m) == 1.0
    assert actor_trainable(False, "transport") == 0.0
    assert actor_trainable(True, "target_entry") == 0.0            # not a control mode


def test_sample_target_proportions():
    assert abs(sum(SAMPLE_TARGET.values()) - 1.0) < 1e-9
    assert SAMPLE_TARGET == {"transport": 0.5, "braking": 0.3, "settling_dwell": 0.2}


# §5 update-0: the event/mode-conditioned actor equals pi_0 exactly for any conditioning
def test_transport_dwell_actor_update0_is_pi0():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    pa = make_phase_actor_from_pi0(pi0, trainable=True, n_cond=N_COND)
    assert pa.backbone[0].in_features == 48 + N_COND
    probe = torch.randn(16, 48)
    base = torch.clamp(pi0.action_mean(probe), -4, 4)
    for cm in CONTROL_MODES:
        ev = np.random.default_rng(0).standard_normal(5).astype(np.float32)
        aug = torch.tensor(np.stack([augment_td(probe[i].numpy(), cm, "contact_present", ev) for i in range(16)]))
        out = torch.clamp(pa.action_mean(aug), -4, 4)
        assert torch.allclose(out, base, atol=0.0)                 # zero conditioning weights ⇒ exact pi_0


def test_masked_actor_loss_excludes_non_trainable():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    pa = make_phase_actor_from_pi0(pi0, trainable=True, n_cond=N_COND)
    with torch.no_grad():
        torch.manual_seed(1); pa.backbone[0].weight[:, 48:48 + N_COND].add_(torch.randn(pa.backbone[0].weight.shape[0], N_COND))
    critic = LateTwinCritic(obs_dim=48 + N_COND)
    ev = np.zeros(5, np.float32); obs = torch.randn(4, 48)
    aug = torch.tensor(np.stack([augment_td(obs[i].numpy(), "transport", "contact_present", ev) for i in range(4)]))
    pa.zero_grad(); masked_actor_loss(critic, pa, aug, torch.zeros(4)).backward()   # all weight 0
    assert torch.cat([p.grad.flatten() for p in pa.parameters()]).abs().max().item() < 1e-9


@pytest.mark.slow
def test_persistent_banks_meet_thresholds():
    from hymeko_rl.coin_delivery.coin_transport_dwell import rebuild_control_mode_bank
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    _banks, counts = rebuild_control_mode_bank(pi0, range(6000, 6260), min_persist=2)
    for m in CONTROL_MODES:
        assert counts[m] >= BANK_MIN[m], f"{m}: {counts[m]} < {BANK_MIN[m]}"
