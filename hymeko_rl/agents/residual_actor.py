"""Bounded, phase-gated neural residual actor over a FROZEN base policy.

    a(o) = clip( pi_base(o) + gate * epsilon * action_scale * tanh(r_phi(o)),  action_low, action_high )

The base policy (e.g. the frozen MLP+DAgger actor) is never updated; only the residual net ``r_phi`` trains. The
residual net is zero-initialised so at init the action EXACTLY equals the base (a ≡ pi_base). ``tanh`` + ``epsilon``
bound the correction to a small band (‖residual‖ ≤ epsilon·action_scale per dim, so the normalised magnitude is
≤ epsilon); a phase ``gate`` ∈ {0,1} confines the correction to monitor-relevant phases (CONTACT/PUSH), leaving the
base untouched elsewhere (gate=0 during APPROACH). This is the actor-level, neural-base counterpart of the
env-level, scripted-base :class:`~hymeko_rl.env.residual.ResidualControllerEnv` — a different primitive for a
different base, adding the tanh bound + phase gate that the env wrapper does not have.

Motivation: the CQL actor smoke (report 2026-07-07-cql-actor-smoke) fixed the critic ranking but still collapsed
via whole-policy drift off the DAgger manifold. Bounding + gating the correction structurally caps that drift.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def build_residual_net(flat_dim: int, action_dim: int, *, hidden: int = 64) -> nn.Module:
    """A small MLP residual head, zero-initialised (last layer) so its output — hence the correction — is 0 at init."""
    net = nn.Sequential(nn.Linear(flat_dim, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
    with torch.no_grad():
        net[-1].weight.zero_()
        net[-1].bias.zero_()
    return net


class ResidualActor(nn.Module):
    """Frozen base + bounded, phase-gated trainable residual. Only ``residual`` has gradients."""

    def __init__(self, base: nn.Module, residual: nn.Module, *, epsilon: float,
                 action_lo: Any, action_hi: Any) -> None:
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive; got {epsilon}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.residual = residual
        self.epsilon = float(epsilon)
        self.register_buffer("lo", torch.as_tensor(action_lo, dtype=torch.float32))
        self.register_buffer("hi", torch.as_tensor(action_hi, dtype=torch.float32))

    # base actor API passthrough (train_offpolicy / greedy_action_fn read these)
    @property
    def action_dim(self) -> int:
        return int(self.base.action_dim)

    @property
    def action_scale(self) -> float:
        return float(self.base.action_scale)

    def _base_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.base.action_mean(obs) if hasattr(self.base, "action_mean") else self.base(obs)

    def raw_residual(self, obs: torch.Tensor) -> torch.Tensor:
        """The UNGATED bounded correction ``epsilon · action_scale · tanh(r_phi(o))`` (‖·‖ ≤ epsilon·scale)."""
        pre = self.residual(obs.reshape(obs.shape[0], -1))
        return self.epsilon * self.action_scale * torch.tanh(pre)

    def saturation(self, obs: torch.Tensor) -> float:
        """Mean ``|tanh(r_phi)|`` — near 1 means the residual is pinned at its bound (saturating)."""
        with torch.no_grad():
            pre = self.residual(obs.reshape(obs.shape[0], -1))
            return float(torch.tanh(pre).abs().mean())

    def action_mean(self, obs: torch.Tensor, gate: "torch.Tensor | None" = None) -> torch.Tensor:
        """``clip(base(o) + gate · residual(o), lo, hi)``. ``gate`` broadcasts over action dims; ``None`` → all-on."""
        base_a = self._base_action(obs)
        r = self.raw_residual(obs)
        if gate is not None:
            r = r * gate.reshape(-1, 1)
        return torch.clamp(base_a + r, self.lo, self.hi)

    def forward(self, obs: torch.Tensor, gate: "torch.Tensor | None" = None) -> torch.Tensor:
        return self.action_mean(obs, gate)


def contact_gate(z: torch.Tensor) -> torch.Tensor:
    """Phase gate from the privileged state ``z`` = [left_contact, right_contact, phase0..2]: 1 in CONTACT/PUSH
    (either fingertip touching the coin), 0 in APPROACH (no contact → residual defaults to zero)."""
    return ((z[:, 0] > 0.5) | (z[:, 1] > 0.5)).float()
