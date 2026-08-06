"""PHASE_GATED_LEARNED_RESIDUAL controller — a frozen learned base ``pi_0`` plus a gated, bounded, zero-initialized
learned residual (§1-§5).

    base_t      = clip(pi_0(obs_t), -4, 4)                       # frozen, requires_grad=False
    residual_t  = residual_transform(residual_actor(obs_t))      # bounded to [-0.25, 0.25] per component
    composite_t = clip(base_t + gate_t * residual_t, -4, 4)      # gate_t in {0,1} from STABLE_OBJECT_ENGAGEMENT_V1

Structural guarantees (the reason for this architecture):
- ``pi_0`` is immutable: no gradients, absent from every optimizer / Polyak update.
- the residual is **zero at init** (last layer zeroed ⇒ ``residual_transform(0)=0``) ⇒ update-0 composite == pi_0.
- when ``gate_t == 0`` the composite equals the base **bit-identically**, independent of the residual network, and
  the residual receives **zero** effective executed-action gradient.

The residual actor sees only the canonical deployable observation (48-d ``node_features``); it is never given phase
labels, target distance, success flags, planner state, trajectory id, or future information.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from torch import nn

RESIDUAL_BOUND = 0.25


class BoundedResidualTransform:
    """``delta_exec = bound * tanh(delta_raw)`` — smooth, symmetric, in ``[-bound, bound]``, identity-slope ``bound``
    at 0 (so a zero raw output ⇒ zero executed residual, and the residual can still learn: d/draw|0 = bound). This is
    an explicit bounded transform — NOT an unbounded residual relying on env clipping (§5)."""

    def __init__(self, bound: float = RESIDUAL_BOUND) -> None:
        self.bound = float(bound)

    def __call__(self, raw: torch.Tensor) -> torch.Tensor:
        return self.bound * torch.tanh(raw)

    def grad_at(self, raw: torch.Tensor) -> torch.Tensor:
        """d delta_exec / d raw = bound * (1 - tanh^2(raw))."""
        return self.bound * (1.0 - torch.tanh(raw) ** 2)


class ZeroInitResidualActor(nn.Module):
    """``node_features`` (48) → bounded residual (4). Last layer zero-initialized ⇒ executed residual == 0 at init."""

    def __init__(self, obs_dim: int = 48, action_dim: int = 4, hidden: int = 256, bound: float = RESIDUAL_BOUND):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)
        self.transform = BoundedResidualTransform(bound)
        self.obs_dim, self.action_dim, self.bound = int(obs_dim), int(action_dim), float(bound)

    def raw(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def residual_exec(self, obs: torch.Tensor) -> torch.Tensor:
        return self.transform(self.net(obs))

    def contract(self) -> dict:
        return {"residual_actor": "ZeroInitResidualActor", "arch": "48->256->256->4 MLP ReLU, last layer zero-init",
                "transform": "delta_exec = 0.25 * tanh(raw)", "bound": self.bound, "obs_dim": self.obs_dim,
                "action_dim": self.action_dim, "init": "executed residual == 0 at update 0",
                "observation": "canonical node_features flat 48 only (no phase/target/success/planner/traj/future)"}

    def contract_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.contract(), sort_keys=True).encode()).hexdigest()


class CompositeResidualController:
    """Frozen base ``pi_0`` + gated bounded residual. ``composite_action`` is the deployed action; the gate multiplier
    is supplied per step by :class:`StableEngagementGate` (which the caller advances from deployable env signals)."""

    def __init__(self, pi0, residual: ZeroInitResidualActor, action_scale: float = 4.0) -> None:
        assert_frozen_base(pi0)
        self.pi0 = pi0
        self.residual = residual
        self.action_scale = float(action_scale)

    def base_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.clamp(self.pi0.action_mean(obs), -self.action_scale, self.action_scale)

    def composite_action(self, obs: torch.Tensor, gate_mult: float) -> torch.Tensor:
        """# Postcondition: gate_mult==0 ⇒ returns base_action bit-identically (residual term is exactly 0)."""
        base = self.base_action(obs)
        residual = self.residual.residual_exec(obs)
        return torch.clamp(base + gate_mult * residual, -self.action_scale, self.action_scale)

    @torch.no_grad()
    def act(self, obs_flat: np.ndarray, gate_mult: float) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs_flat)[None], dtype=torch.float32)
        return self.composite_action(obs, gate_mult)[0].numpy()


def assert_frozen_base(pi0) -> None:
    """# Invariant: every pi_0 parameter is frozen. Raises if any requires grad."""
    for n, p in pi0.named_parameters():
        if p.requires_grad:
            raise AssertionError(f"frozen base pi_0 parameter {n!r} has requires_grad=True")


def assert_base_absent_from_optimizer(pi0, optimizer: torch.optim.Optimizer) -> None:
    """# Invariant: no pi_0 tensor is in any optimizer param group (structural-preservation guard, §2)."""
    base_ids = {id(p) for p in pi0.parameters()}
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in base_ids:
                raise AssertionError("frozen base pi_0 parameter found in an optimizer param group")
