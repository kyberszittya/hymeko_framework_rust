"""DYNAMIC_PHASE_TRANSITION_CONTRACT_V1 (§9): current per-transition phase drives actor/critic/target — NOT the static
LateStart.family."""
import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.coin_phase_conditioning import (
    N_PHASE,
    PHASES,
    PhaseDetector,
    make_phase_actor_from_pi0,
    phase_onehot,
)
from hymeko_rl.coin_delivery.coin_td3_phase_stage1c import (
    collect_late_episode_phase,
    phase_target_action_c,
)
from hymeko_rl.coin_delivery.coin_td3_trainer import masked_actor_loss
from hymeko_rl.coin_delivery.coin_td3_contracts import LateTwinCritic
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


class _MockRL:
    """Scriptable env for the detector: set per-state metrics, read them back."""
    def __init__(self):
        self._strict = 0; self._spd = 0.0
        self.inner = type("I", (), {"_planar_metrics": None})()

    def set(self, dtz, speed, lc, rc, strict):
        self.inner._planar_metrics = type("M", (), {"disk_to_zone": dtz, "left_contact": lc, "right_contact": rc})()
        self._spd, self._strict = speed, strict

    def _speed(self):
        return self._spd


# §9.1 — a synthetic entry → braking → settling trajectory yields three different phase labels
def test_detector_entry_braking_settling_distinct():
    det = PhaseDetector(); rl = _MockRL(); seq = []
    for dtz, spd, strict in [(0.10, 0.50, 0), (0.04, 0.40, 0), (0.045, 0.10, 0), (0.01, 0.02, 0)]:
        rl.set(dtz, spd, True, True, strict); seq.append(det.phase_of(rl))
    assert "target_entry" in seq and "braking" in seq and "settling_dwell" in seq
    assert len(set(seq)) >= 3


# §9.2 / §9.3 — phase changes output after training, but NOT at update 0 (== pi_0 ∀ phase)
def test_phase_changes_output_only_after_training():
    pi0 = load_frozen_clip_actor(PI0, freeze=True); pa = make_phase_actor_from_pi0(pi0, trainable=True)
    obs = torch.randn(4, 48)
    a_entry = pa.action_mean(torch.cat([obs, torch.tensor(np.tile(phase_onehot("target_entry"), (4, 1)))], -1))
    a_brake = pa.action_mean(torch.cat([obs, torch.tensor(np.tile(phase_onehot("braking"), (4, 1)))], -1))
    assert torch.equal(a_entry, a_brake)                       # update 0: phase-independent (zero phase weights)
    with torch.no_grad():                                      # "train" the phase-input weights
        torch.manual_seed(1); pa.backbone[0].weight[:, 48:48 + N_PHASE].add_(torch.randn(pa.backbone[0].weight.shape[0], N_PHASE))
    b_entry = pa.action_mean(torch.cat([obs, torch.tensor(np.tile(phase_onehot("target_entry"), (4, 1)))], -1))
    b_brake = pa.action_mean(torch.cat([obs, torch.tensor(np.tile(phase_onehot("braking"), (4, 1)))], -1))
    assert not torch.allclose(b_entry, b_brake)                # after training: phase matters


# §9.4 — phase_t enters the actor loss (changing only the phase one-hot changes the gradient, once trained)
def test_phase_t_used_in_actor_loss():
    pi0 = load_frozen_clip_actor(PI0, freeze=True); pa = make_phase_actor_from_pi0(pi0, trainable=True)
    with torch.no_grad():
        torch.manual_seed(1); pa.backbone[0].weight[:, 48:48 + N_PHASE].add_(torch.randn(pa.backbone[0].weight.shape[0], N_PHASE))
    critic = LateTwinCritic(obs_dim=48 + N_PHASE); obs = torch.randn(6, 48); gate = torch.ones(6)
    aug_a = torch.cat([obs, torch.tensor(np.tile(phase_onehot("target_entry"), (6, 1)))], -1)
    aug_b = torch.cat([obs, torch.tensor(np.tile(phase_onehot("settling_dwell"), (6, 1)))], -1)
    pa.zero_grad(); masked_actor_loss(critic, pa, aug_a, gate).backward()
    ga = torch.cat([p.grad.flatten() for p in pa.parameters()]).clone()
    pa.zero_grad(); masked_actor_loss(critic, pa, aug_b, gate).backward()
    gb = torch.cat([p.grad.flatten() for p in pa.parameters()])
    assert not torch.allclose(ga, gb)


# §9.5 — phase_tp1 / phase_boot enters the TD target
def test_phase_boot_used_in_td_target():
    pi0 = load_frozen_clip_actor(PI0, freeze=True); tgt = make_phase_actor_from_pi0(pi0, trainable=False)
    with torch.no_grad():
        torch.manual_seed(1); tgt.backbone[0].weight[:, 48:48 + N_PHASE].add_(torch.randn(tgt.backbone[0].weight.shape[0], N_PHASE))
    boot48 = torch.randn(5, 48); gate_next = torch.ones(5); gen = torch.Generator().manual_seed(0)
    oh_a = torch.tensor(np.tile(phase_onehot("braking"), (5, 1))); oh_b = torch.tensor(np.tile(phase_onehot("settling_dwell"), (5, 1)))
    ta = phase_target_action_c(pi0, tgt, boot48, oh_a, gate_next, std=0.0, clip=0.25, gen=gen)
    tb = phase_target_action_c(pi0, tgt, boot48, oh_b, gate_next, std=0.0, clip=0.25, gen=gen)
    assert not torch.allclose(ta, tb)                          # different bootstrap phase ⇒ different target


# §9.6 / §9.7 — replay reconstruction preserves the exact phase sequence; no transition uses the static family
@pytest.mark.slow
def test_replay_preserves_phase_sequence_and_is_dynamic():
    from hymeko_rl.coin_delivery.coin_late_start import build_late_start_bank
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    bank = build_late_start_bank(pi0, range(6000, 6040), per_family=3)
    pl = make_phase_actor_from_pi0(pi0, trainable=True)
    # §9.6 deterministic phase sequence (explore=False ⇒ reproducible); phase_tp1(k) == phase_t(k+1)
    trs1 = collect_late_episode_phase(pi0, pl, bank[0], None, horizon=30, explore=False)
    trs2 = collect_late_episode_phase(pi0, pl, bank[0], None, horizon=30, explore=False)
    assert [t["phase_t"] for t in trs1] == [t["phase_t"] for t in trs2]
    for k in range(len(trs1) - 1):
        assert trs1[k]["phase_tp1"] == trs1[k + 1]["phase_t"]
    # §9.7 the stored phase is dynamic — at least one transition's phase differs from its static LateStart.family
    dynamic = False
    for ls in bank:
        trs = collect_late_episode_phase(pi0, pl, ls, None, horizon=30, explore=False)
        if any(t["phase_t"] != t["family"] for t in trs):
            dynamic = True; break
    assert dynamic, "phase_t never differs from the static family — conditioning would be on the start label"
    for t in trs1:
        assert t["phase_t"] in PHASES and t["phase_tp1"] in PHASES
