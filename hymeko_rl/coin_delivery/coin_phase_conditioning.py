"""Smallest-compatible PHASE conditioning (§5) for the phase-switched late controller — NO training here.

The completed Stage-1/1b used ``obs_48`` only: ``pi_late`` and both critics were phase-BLIND, i.e. one late actor for
ALL late phases (a binary early/late controller, NOT a phase-conditioned multi-state baseline). This module adds a
phase one-hot to the actor input and the critic state:

    actor_input  = obs_48 ++ phase_onehot
    critic_state = obs_48 ++ phase_onehot   (critic also takes the action)

Every NEW phase-input weight is ZERO-initialized, so at update 0 the phase contributes nothing and the actor reproduces
``pi_0`` exactly for every phase (and the critic's Q is phase-independent at init).
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_late_start import LATE_FAMILIES, _classify
from hymeko_rl.coin_delivery.coin_td3_contracts import LateTwinCritic
from hymeko_rl.coin_delivery.rl_clip_actor import ClipDeterministicActor, make_backbone

ACTION_SCALE = 4.0
PHASES = LATE_FAMILIES                       # ("transport","target_entry","overshoot","braking","settling_dwell","contact_retention")
N_PHASE = len(PHASES)


class PhaseDetector:
    """DYNAMIC per-transition phase from the CURRENT env state + running trajectory context (deterministic). Call
    :meth:`phase_of` EXACTLY ONCE per state as the trajectory unfolds — it advances the context — so
    ``phase_tp1(k) == phase_t(k+1)`` by construction (the sequence is reconstructible from a deterministic replay).

    This is the CURRENT phase, NOT the static ``LateStart.family`` episode-start label.
    """

    def __init__(self) -> None:
        self.prev_dtz = None
        self.min_dtz = None
        self.prev_speed = None

    def phase_of(self, rl) -> str:
        m = rl.inner._planar_metrics
        dtz = float(m.disk_to_zone); speed = float(rl._speed())
        lc, rc, strict = bool(m.left_contact), bool(m.right_contact), int(rl._strict)
        pdtz = dtz if self.prev_dtz is None else self.prev_dtz
        mdtz = dtz if self.min_dtz is None else self.min_dtz
        pspd = speed if self.prev_speed is None else self.prev_speed
        phase = _classify(0, dtz, pdtz, mdtz, speed, pspd, lc, rc, strict)
        self.min_dtz = min(mdtz, dtz); self.prev_dtz = dtz; self.prev_speed = speed
        return phase


def phase_onehot(family: str) -> np.ndarray:
    v = np.zeros(N_PHASE, np.float32)
    if family in PHASES:
        v[PHASES.index(family)] = 1.0
    return v


def augment(obs: np.ndarray, family: str) -> np.ndarray:
    return np.concatenate([np.asarray(obs, np.float32), phase_onehot(family)]).astype(np.float32)


def make_phase_actor_from_pi0(pi0, *, trainable: bool = True, n_cond: int = N_PHASE) -> ClipDeterministicActor:
    """``pi_late`` over ``obs_48 ++ conditioning`` (dim 48+``n_cond``), conditioning-input weights ZERO ⇒ update-0 == pi_0
    for ANY conditioning vector (``n_cond`` = phase one-hot width, or control+contact+event features)."""
    feat = pi0.head.in_features
    backbone = make_backbone(48 + n_cond, int(feat))
    with torch.no_grad():
        backbone[0].weight.zero_()
        backbone[0].weight[:, :48].copy_(pi0.backbone[0].weight)        # obs columns = pi_0; conditioning columns = 0
        backbone[0].bias.copy_(pi0.backbone[0].bias)
        backbone[2].load_state_dict(pi0.backbone[2].state_dict())
    late = ClipDeterministicActor(backbone, int(feat), int(pi0.action_dim), float(pi0.action_scale))
    late.head.load_state_dict(pi0.head.state_dict())
    for p in late.parameters():
        p.requires_grad_(bool(trainable))
    (late.train() if trainable else late.eval())
    return late


def make_phase_critic(n_cond: int = N_PHASE) -> LateTwinCritic:
    """Twin critic over ``(obs_48 ++ conditioning, action_4)``; conditioning-input columns of the first layer ZERO-init."""
    critic = LateTwinCritic(obs_dim=48 + n_cond)
    with torch.no_grad():
        for q in (critic.q1, critic.q2):
            q[0].weight[:, 48:48 + n_cond].zero_()                      # conditioning columns = 0 at init
    return critic


def assert_phase_actor_is_pi0_at_update0(pi0, phase_actor, *, tol: float = 0.0) -> None:
    """# Invariant: for EVERY phase, phase_actor(obs ++ phase_onehot) == pi_0(obs) at init (zero phase weights)."""
    probe = torch.randn(16, 48, generator=torch.Generator().manual_seed(7))
    base = torch.clamp(pi0.action_mean(probe), -ACTION_SCALE, ACTION_SCALE)
    for fam in PHASES:
        oh = torch.tensor(np.tile(phase_onehot(fam), (16, 1)))
        out = torch.clamp(phase_actor.action_mean(torch.cat([probe, oh], -1)), -ACTION_SCALE, ACTION_SCALE)
        d = (out - base).abs().max().item()
        if d > tol:
            raise AssertionError(f"phase {fam!r}: update-0 actor differs from pi_0 by {d} (> {tol})")
