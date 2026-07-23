"""Contract tests for PHASE_SWITCHED_LEARNED_LATE_CONTROLLER_V1 (no training). Covers: exact pi_0-copy init + gate-off
bit-identity, deterministic replay-to-handoff reconstruction, n-step target masking (terminated vs truncated), coherent
noise hold-length, and full-action target smoothing bounds."""
import numpy as np
import pytest
import torch

from hymeko_rl.coin_delivery.coin_phase_switched_late import (
    PhaseSwitchedController,
    assert_late_is_pi0_copy,
    make_late_actor_from_pi0,
)
from hymeko_rl.coin_delivery.coin_td3_contracts import (
    ACTION_SCALE,
    CoherentNoise,
    LateReplayBuffer,
    LateTwinCritic,
    TD3Config,
    nstep_return,
    td3_target_action,
)
from hymeko_rl.coin_delivery.rl_clip_actor import load_frozen_clip_actor

PI0 = "experiments/2026_07_22_coin_v3_learning/rl_entry/frozen/pi0_shared_clip_actor.pt"


# ── controller / exact-copy init ──
def test_late_actor_is_exact_pi0_copy_and_trainable():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    late = make_late_actor_from_pi0(pi0, trainable=True)
    assert_late_is_pi0_copy(pi0, late, tol=0.0)                     # byte-identical at init
    assert all(p.requires_grad for p in late.parameters())         # trainable
    probe = torch.randn(8, 48)
    assert torch.equal(pi0.action_mean(probe), late.action_mean(probe))   # identical outputs at update 0


def test_controller_gate_off_is_pi0_bitidentical_regardless_of_late():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    late = make_late_actor_from_pi0(pi0, trainable=True)
    with torch.no_grad():                                          # perturb pi_late so it differs from pi_0
        for p in late.parameters():
            p.add_(1.0)
    ctrl = PhaseSwitchedController(pi0, late)
    obs = torch.randn(5, 48)
    off = ctrl.switched_action(obs, 0.0); base = ctrl.base_action(obs)
    assert torch.equal(off, base)                                  # gate-off == pi_0 exactly
    on = ctrl.switched_action(obs, 1.0)
    assert not torch.equal(on, base)                               # gate-on uses the (now different) pi_late


def test_controller_requires_frozen_pi0():
    pi0 = load_frozen_clip_actor(PI0, freeze=False)                # NOT frozen
    late = make_late_actor_from_pi0(pi0)
    with pytest.raises(AssertionError):
        PhaseSwitchedController(pi0, late)


# ── n-step return masking ──
def _traj(rewards, term_at=None, trunc_at=None):
    tr = []
    for i, r in enumerate(rewards):
        tr.append({"obs": np.zeros(48, np.float32), "action": np.zeros(4, np.float32), "reward": float(r),
                   "obs_next": np.full(48, i + 1, np.float32), "terminated": i == term_at, "truncated": i == trunc_at})
    return tr


def test_nstep_no_termination():
    G, boot, mask, gp = nstep_return(_traj([1, 1, 1, 1, 1]), 0, 4, 1.0)
    assert G == 4.0 and mask == 1 and gp == 1.0 and boot[0] == 4     # bootstrap at obs_next of step 3


def test_nstep_terminated_masks_bootstrap():
    G, boot, mask, gp = nstep_return(_traj([1, 1, 1, 1, 1], term_at=2), 0, 4, 1.0)
    assert G == 3.0 and mask == 0                                   # stop at termination, no bootstrap
    assert boot[0] == 3


def test_nstep_truncated_keeps_bootstrap():
    G, boot, mask, gp = nstep_return(_traj([1, 1, 1, 1, 1], trunc_at=2), 0, 4, 1.0)
    assert G == 3.0 and mask == 1                                   # truncation bootstraps (artificial cutoff)


def test_nstep_discount_and_runoff():
    G, boot, mask, gp = nstep_return(_traj([1, 1], ), 0, 4, 0.5)    # only 2 stored → runs off
    assert abs(G - (1 + 0.5)) < 1e-9 and mask == 1 and abs(gp - 0.25) < 1e-9


def test_replay_buffer_sample_nstep_shapes():
    buf = LateReplayBuffer()
    buf.add_trajectory(_traj([1, 1, 1, 1, 1, 1]))
    buf.add_trajectory(_traj([2, 2, 2], term_at=2))
    obs, act, rew, boot, mask, gp = buf.sample_nstep(16, n=4, gamma=0.99, rng=np.random.default_rng(0))
    assert obs.shape == (16, 48) and act.shape == (16, 4) and rew.shape == (16,)
    assert boot.shape == (16, 48) and set(np.unique(mask)) <= {0.0, 1.0}


def test_replay_buffer_rejects_merged_done():
    buf = LateReplayBuffer()
    with pytest.raises(AssertionError):
        buf.add_trajectory([{"obs": np.zeros(48), "action": np.zeros(4), "reward": 0.0,
                             "obs_next": np.zeros(48), "done": True}])   # 'done' merged, missing terminated/truncated


# ── coherent noise ──
def test_coherent_noise_held_2_to_4_steps():
    cn = CoherentNoise(action_dim=4, std=0.3, hold_min=2, hold_max=4, seed=1)
    seq = [cn.sample() for _ in range(60)]
    # count run lengths of identical consecutive noise vectors
    runs, cur = [], 1
    for i in range(1, len(seq)):
        if np.array_equal(seq[i], seq[i - 1]):
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    assert all(2 <= r <= 4 for r in runs[:-1])                     # every completed hold is 2–4 steps
    assert len(set(map(tuple, seq))) > 1                           # not a single constant (it resamples)


def test_coherent_noise_reject_bad_hold():
    with pytest.raises(AssertionError):
        CoherentNoise(hold_min=1, hold_max=4)                      # per-step (hold 1) is forbidden


# ── full-action target smoothing ──
def test_td3_target_action_bounds_full_action_units():
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    cfg = TD3Config()
    obs = torch.randn(64, 48)
    a = td3_target_action(pi0, obs, smoothing_std=cfg.smoothing_std, smoothing_clip=cfg.smoothing_clip,
                          generator=torch.Generator().manual_seed(0))
    assert a.shape == (64, 4)
    assert a.abs().max().item() <= ACTION_SCALE + 1e-6             # clipped to full-action bound
    # smoothing is in full-action units (std ~0.8), not residual units (would be ~0.05)
    assert cfg.smoothing_std > 0.5 and cfg.smoothing_clip > 1.0


def test_late_twin_critic_independent_heads():
    c = LateTwinCritic()
    o, a = torch.randn(5, 48), torch.randn(5, 4)
    q1, q2 = c(o, a)
    assert q1.shape == (5,) and not torch.allclose(q1, q2)
    assert torch.allclose(c.min_q(o, a), torch.min(q1, q2))


# ── deterministic replay-to-handoff (env-touching) ──
@pytest.mark.slow
def test_replay_to_handoff_is_deterministic_and_exact():
    from hymeko_rl.coin_delivery.coin_late_start import (
        build_late_start_bank,
        reconstruct_handoff,
        verify_reconstruction,
    )
    pi0 = load_frozen_clip_actor(PI0, freeze=True)
    bank = build_late_start_bank(pi0, range(6100, 6120), per_family=1)
    assert bank, "no late-starts captured"
    for ls in bank[:6]:
        v = verify_reconstruction(pi0, ls)
        assert v["obs_ok"] and v["base_ok"] and v["causal_ok"] and v["gate_ok"] and v["gate_active"]
    # reconstruct twice → identical live observation (replay, not snapshot restore)
    ls = bank[0]
    _rl1, _g1, _h1, r1 = reconstruct_handoff(pi0, ls)
    _rl2, _g2, _h2, r2 = reconstruct_handoff(pi0, ls)
    assert np.array_equal(r1.obs, r2.obs) and np.array_equal(r1.base, r2.base)
