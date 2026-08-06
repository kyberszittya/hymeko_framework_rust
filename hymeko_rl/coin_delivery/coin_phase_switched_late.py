"""PHASE_SWITCHED_LEARNED_LATE_CONTROLLER_V1 — a frozen ``pi_0`` for gate-off steps and a FULL-action learned
``pi_late`` for gate-on steps (NOT an additive residual; the ±0.25 residual route is retired).

    action_t = clip(pi_0(obs_t),   -4, 4)     if gate_t == 0     # frozen, structurally controls every gate-off step
    action_t = clip(pi_late(obs_t), -4, 4)    if gate_t == 1     # full-action learned late-phase actor

Structural guarantees:
- ``pi_0`` is immutable (no grad, absent from every optimizer/Polyak update); gate-off actions are bit-identical to it,
  independent of ``pi_late``.
- ``pi_late`` is initialized as an EXACT copy of ``pi_0`` (same clip-squashed backbone+head), so at update 0 the composed
  controller reproduces ``pi_0`` on every step (gate-on == gate-off == pi_0) and therefore the frozen neutral-reset
  result: headline 3/9, validation 2/30, grasp 9/9, delivered {1011, 1447, 1568}.
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.coin_delivery.coin_residual_controller import assert_base_absent_from_optimizer, assert_frozen_base
from hymeko_rl.coin_delivery.rl_clip_actor import ClipDeterministicActor, make_backbone

ACTION_SCALE = 4.0


def make_late_actor_from_pi0(pi0, *, trainable: bool = True) -> ClipDeterministicActor:
    """``pi_late`` = an EXACT copy of ``pi_0`` (same clip-squashed architecture and weights), trainable by default.

    # Preconditions: ``pi0`` is a :class:`ClipDeterministicActor`. # Postconditions: returned actor's parameters equal
    ``pi_0``'s byte-for-byte at init (``update-0`` reproduces ``pi_0``); ``requires_grad`` is set per ``trainable``.
    """
    obs_dim = pi0.backbone[0].in_features
    feat = pi0.head.in_features
    backbone = make_backbone(int(obs_dim), int(feat))
    late = ClipDeterministicActor(backbone, int(feat), int(pi0.action_dim), float(pi0.action_scale))
    late.load_state_dict(pi0.state_dict())                       # exact copy
    for p in late.parameters():
        p.requires_grad_(bool(trainable))
    (late.train() if trainable else late.eval())
    return late


class PhaseSwitchedController:
    """Frozen ``pi_0`` (gate-off) + learned full-action ``pi_late`` (gate-on). ``act`` is the deployed eval contract."""

    def __init__(self, pi0, pi_late: ClipDeterministicActor, action_scale: float = ACTION_SCALE) -> None:
        assert_frozen_base(pi0)                                  # invariant: pi_0 is frozen
        self.pi0 = pi0
        self.pi_late = pi_late
        self.action_scale = float(action_scale)

    def base_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.clamp(self.pi0.action_mean(obs), -self.action_scale, self.action_scale)

    def late_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.pi_late.action_mean(obs), -self.action_scale, self.action_scale)

    def switched_action(self, obs: torch.Tensor, gate_mult: float) -> torch.Tensor:
        """# Postcondition: ``gate_mult==0`` ⇒ returns ``base_action`` bit-identically (pi_late is not consulted)."""
        if float(gate_mult) == 0.0:
            return self.base_action(obs)
        with torch.no_grad():
            return self.late_action(obs)

    @torch.no_grad()
    def act(self, obs_flat: np.ndarray, gate_mult: float) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs_flat)[None], dtype=torch.float32)
        return self.switched_action(obs, gate_mult)[0].numpy()


def assert_late_is_pi0_copy(pi0, pi_late, *, tol: float = 0.0) -> None:
    """# Invariant (update-0): every ``pi_late`` parameter equals ``pi_0``'s (within ``tol``). Raises otherwise."""
    p0 = dict(pi0.named_parameters()); pl = dict(pi_late.named_parameters())
    if set(p0) != set(pl):
        raise AssertionError("pi_late and pi_0 have different parameter names (not the same architecture)")
    for n in p0:
        d = (p0[n].detach() - pl[n].detach()).abs().max().item()
        if d > tol:
            raise AssertionError(f"pi_late parameter {n!r} differs from pi_0 by {d} (> {tol})")


__all__ = ["make_late_actor_from_pi0", "PhaseSwitchedController", "assert_late_is_pi0_copy",
           "assert_frozen_base", "assert_base_absent_from_optimizer"]
