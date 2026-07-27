"""R0 — the bounded residual over the frozen KINETIC clone (the wrapper the reward-driven RL trains).

The learned KINETIC action becomes

    u_t = clip( u_clone(o_{≤t}) + α · tanh δ_ψ(o_{≤t}),  u_min, u_max )

where ``u_clone`` is the FROZEN K2 clone and ``δ_ψ`` a small learned residual (the TD3/SAC actor), bounded to ± α. The residual
SHAPES the clone; it does not replace it. Mandatory update-zero identity: with a zero-initialised residual (``tanh δ_ψ ≡ 0``)
the deployed controller reproduces the K2 clone BIT-FOR-BIT — same APPROACH, same KINETIC transport, same release/coast/K6,
same 46.2 mm landing. The residual only ever modifies the KINETIC segment; every torque/slew/joint limit and the release/coast/
K6 downstream stay the frozen scaffold's; no teacher is in the loop.

This module is the deploy-side wrapper (policy net + actor + controller) reused by both the R0 identity gate and the R1/R2
TD3-vs-SAC training; the training loop itself lives in the R1 experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.theta_option.kinetic_clone import (
    ACT_DIM, OBS_DIM, CloneActor, KineticCloneController, NormStats)


@dataclass(frozen=True)
class ResidualBounds:
    """The residual scale α (deliberately small — the residual refines a working clone). # Invariants: α ≥ 0; the executed
    correction is within ± α, so the clone remains the dominant term."""

    alpha: float = 0.15


class ResidualPolicy(nn.Module):
    """The deterministic residual actor δ_ψ: MLP (OBS_DIM → hidden → hidden → ACT_DIM), tanh-bounded. With ``zero_init`` the
    FINAL layer is zeroed, so ``forward`` returns exactly 0 for any input — the update-zero identity. # Postconditions: output
    in (−1, 1)^ACT_DIM."""

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden: int = 64, zero_init: bool = True) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        if zero_init:
            nn.init.zeros_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mu(self.body(x)))


class ResidualActor:
    """Streaming deploy wrapper for a residual policy: frozen (clone) normalisation + deterministic eval. ``act(obs)`` returns
    the residual output ``tanh δ_ψ ∈ [−1,1]^ACT_DIM`` for one step. # Invariants: eval + no_grad ⇒ deterministic; reads only the
    41-D observation; at a zero-initialised policy returns exactly 0."""

    def __init__(self, policy: nn.Module, norm: NormStats) -> None:
        self.policy = policy.eval()
        self.norm = norm

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = self.norm.apply(obs).astype(np.float32)
        with torch.no_grad():
            r = self.policy(torch.from_numpy(x).view(1, -1))
        return r.view(-1).numpy().astype(np.float64)


class KineticResidualController(KineticCloneController):
    """Frozen APPROACH + frozen KINETIC transitions + frozen K2 clone, with a bounded residual added to ONLY the KINETIC
    transport action: ``u = clip(u_clone + α · tanh δ_ψ, −1, 1)``. # Postconditions: at a zero residual, identical to the K2
    clone (update-zero identity); |Δτ| ≤ slew; no teacher in the loop; release/coast/K6 downstream unchanged."""

    def __init__(self, snap: Any, clone: CloneActor, residual: ResidualActor, bounds: ResidualBounds = ResidualBounds(),
                 **kw: Any) -> None:
        self.residual = residual
        self.bounds = bounds
        super().__init__(snap, clone, **kw)

    def reset(self) -> None:
        super().reset()
        if hasattr(self.residual, "reset"):
            self.residual.reset()
        self.residual_trace: list[dict[str, Any]] = []

    def _transport_action(self, rl: Any, obs: np.ndarray) -> np.ndarray:
        u_clone = np.clip(np.asarray(self.actor.act(obs), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        r = np.clip(np.asarray(self.residual.act(obs), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        u = np.clip(u_clone + self.bounds.alpha * r, -1.0, 1.0)
        self.residual_trace.append({"u_clone": [round(float(x), 5) for x in u_clone],
                                    "residual": [round(float(x), 5) for x in r], "u": [round(float(x), 5) for x in u]})
        return u
