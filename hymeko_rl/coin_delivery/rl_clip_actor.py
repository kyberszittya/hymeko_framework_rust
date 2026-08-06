"""Clip-squashed actors for the RL entry campaign — a squash that is IDENTITY inside the action range and clips at the
bounds (exactly the env's own ±action_scale clipping). Unlike tanh, a linear-in-range squash lets the actor reproduce
the raw handoff-feedback BC EXACTLY by direct weight-load, which the chaotic contact task requires (measured
2026-07-22: tanh ≤2/9, clip 3/9 = BC). Both SAC (`ClipGaussianActor`) and TD3 (`ClipDeterministicActor`) share ONE
backbone+head tensor and eval to the identical clipped action.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

_LOG_STD_MIN, _LOG_STD_MAX = -6.0, 2.0


def make_backbone(obs_dim: int = 48, feat: int = 256) -> nn.Sequential:
    return nn.Sequential(nn.Linear(obs_dim, feat), nn.ReLU(), nn.Linear(feat, feat), nn.ReLU())


def load_bc_weights(backbone: nn.Sequential, head: nn.Linear, bc) -> None:
    """Copy the FullActionBC (48→256→256→4) into (backbone 48→256→256) + (head 256→4). Exact — no distillation."""
    backbone[0].load_state_dict(bc.net[0].state_dict())
    backbone[2].load_state_dict(bc.net[2].state_dict())
    head.load_state_dict(bc.net[4].state_dict())


class ClipDeterministicActor(nn.Module):
    """TD3 actor: ``a = clip(head(backbone(s)), −scale, scale)`` — identity in-range, so it reproduces the raw BC."""

    def __init__(self, backbone: nn.Module, feat_dim: int, action_dim: int, action_scale: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, action_dim)
        self.action_scale = float(action_scale)
        self._action_dim = int(action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.head(self.backbone(obs)), -self.action_scale, self.action_scale)

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def head_parameters(self):
        return list(self.head.parameters())


class ClipGaussianActor(nn.Module):
    """SAC actor: pre = μ + σ⊙ε, ``a = clip(pre, −scale, scale)``. Deterministic eval = ``clip(μ)`` (= the BC in-range,
    so exact reproduction). ``sample`` returns (action, log_prob) using the pre-clip Gaussian density — exact while the
    samples stay in-range (small init σ + in-range BC mean); the clip only acts as a safety bound near ±scale."""

    def __init__(self, backbone: nn.Module, feat_dim: int, action_dim: int, action_scale: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.mu = nn.Linear(feat_dim, action_dim)
        self.log_std = nn.Linear(feat_dim, action_dim)
        self.action_scale = float(action_scale)
        self._action_dim = int(action_dim)

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.mu(self.backbone(obs)), -self.action_scale, self.action_scale)

    def sample(self, obs: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        h = self.backbone(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre = mu + std * eps
        action = torch.clamp(pre, -self.action_scale, self.action_scale)
        log_prob = (-0.5 * (eps ** 2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)   # pre-clip Gaussian log-density
        return action, log_prob

    @property
    def action_dim(self) -> int:
        return self._action_dim


class ActorEvalWrap:
    """Adapt an actor's deterministic mean to the ``.act(flat_obs)->action`` eval contract."""

    def __init__(self, actor):
        self.actor = actor

    @torch.no_grad()
    def act(self, obs):
        return self.actor.action_mean(torch.as_tensor(np.asarray(obs)[None], dtype=torch.float32))[0].numpy()


def build_shared_sac_td3(bc, *, obs_dim: int = 48, feat: int = 256, action_dim: int = 4, action_scale: float = 4.0):
    """Build a SAC + a TD3 actor that SHARE one backbone+head tensor loaded directly from ``bc`` (exact reproduction).
    SAC's μ and TD3's head are the same parameters (log_std init to ≈−6 for a near-deterministic start)."""
    backbone = make_backbone(obs_dim, feat)
    td3 = ClipDeterministicActor(backbone, feat, action_dim, action_scale)
    load_bc_weights(td3.backbone, td3.head, bc)
    sac = ClipGaussianActor(td3.backbone, feat, action_dim, action_scale)   # SHARE the backbone tensor
    sac.mu.load_state_dict(td3.head.state_dict())                           # SHARE the head as μ
    with torch.no_grad():
        sac.log_std.weight.zero_()
        sac.log_std.bias.fill_(-6.0)
    return sac, td3


def load_frozen_clip_actor(ckpt_path: str, *, freeze: bool = True) -> "ClipDeterministicActor":
    """Reconstruct the immutable frozen ``pi_0`` from a persisted ``shared_clip_actor_init.pt`` checkpoint.

    The checkpoint is ``{backbone, head, action_scale, obs_dim, feat, action_dim, squash}`` (as written by the
    RL-init reproduction, file-SHA prefix ``1902454c``). Parameters are byte-identical to
    :func:`build_shared_sac_td3` on the same BC (verified maxdiff 0.0), but the persisted FILE is the stable identity
    because ``torch.save`` is not byte-reproducible across independent saves.

    # Preconditions: ``ckpt_path`` exists and has the expected keys. # Postconditions: returns a
    :class:`ClipDeterministicActor`; if ``freeze`` every parameter has ``requires_grad=False`` and the module is in
    eval mode. # Invariants: the returned actor's outputs equal the audited frozen actor's.
    """
    ck = torch.load(ckpt_path)
    backbone = make_backbone(int(ck.get("obs_dim", 48)), int(ck.get("feat", 256)))
    backbone.load_state_dict(ck["backbone"])
    actor = ClipDeterministicActor(backbone, int(ck.get("feat", 256)), int(ck.get("action_dim", 4)),
                                   float(ck.get("action_scale", 4.0)))
    actor.head.load_state_dict(ck["head"])
    if freeze:
        for p in actor.parameters():
            p.requires_grad_(False)
        actor.eval()
    return actor
