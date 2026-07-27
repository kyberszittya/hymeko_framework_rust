"""R2 — the per-step STATE-DEPENDENT bounded residual over the frozen KINETIC clone.

R1 showed a constant residual only adds coast momentum. R2 learns to actively FOLLOW the coin in light contact: the residual
must vary frame-by-frame with the contact / slip / geometry, holding contact below ~55 mm to a close-and-moving release. No
second recurrent net is built — the FROZEN clone already carries the temporal state; the residual conditions on

    δ_ψ(o_t, h_clone,t, δu_{t-1})   →   augmented state  [ norm(41-D obs) · clone GRU hidden (64) · previous residual (4) ]

a small MLP → 4-D bounded residual, applied as ``u = clip(u_clone + α·tanh δ_ψ, −1, 1)``. The clone weights stay frozen; every
torque/slew/joint limit and the release/coast/K6 downstream are the frozen scaffold's; no teacher is in the loop. The controller
records the per-step (augmented state, residual action) so a per-step off-policy RL (TD3/SAC) can train ``δ_ψ`` directly.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from hymeko_rl.coin_delivery.theta_option.kinetic_clone import ACT_DIM, CloneActor, KineticCloneController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds

CLONE_HIDDEN = 64
AUG_DIM = 41 + CLONE_HIDDEN + ACT_DIM          # 109 — [norm obs · clone hidden · prev residual]


class TemporalResidualPolicy(nn.Module):
    """Deterministic per-step residual δ_ψ over the augmented state (MLP → tanh). ``zero_init`` zeroes the final layer so the
    residual is exactly 0 everywhere at init (the update-zero identity). # Postconditions: output in (−1, 1)^ACT_DIM."""

    def __init__(self, aug_dim: int = AUG_DIM, act_dim: int = ACT_DIM, hidden: int = 64, zero_init: bool = False) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(aug_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        if zero_init:
            nn.init.zeros_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mu(self.body(x)))

    def mean_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class KineticTemporalResidualController(KineticCloneController):
    """Frozen APPROACH + frozen KINETIC transitions + frozen clone, with a PER-STEP state-dependent residual over the KINETIC
    transport action. ``residual_fn(aug) -> [−1,1]^ACT_DIM`` receives the augmented state each step (may add exploration). Records
    ``aug_trace`` = per-step (augmented state, residual action) for RL. # Postconditions: at a zero residual, identical to the
    K2 clone; |Δτ| ≤ slew; no teacher; release/coast/K6 downstream unchanged."""

    def __init__(self, snap: Any, clone: CloneActor, residual_fn: Callable[[np.ndarray], np.ndarray],
                 bounds: ResidualBounds = ResidualBounds(), *, start_kinetic: "dict | None" = None, **kw: Any) -> None:
        self.residual_fn = residual_fn
        self.bounds = bounds
        self._start_kinetic = start_kinetic     # {clone_hidden, prev_res} for a segment-local frontier restart (else None)
        super().__init__(snap, clone, **kw)

    def reset(self) -> None:
        super().reset()
        self._prev_res = np.zeros(ACT_DIM, np.float64)
        self.aug_trace: list[tuple[np.ndarray, np.ndarray]] = []
        self.residual_trace: list[dict[str, Any]] = []
        if self._start_kinetic is not None:     # segment-local restart: begin IN the KINETIC phase with the restored state
            from hymeko_rl.coin_delivery.theta_option.hybrid_approach import KINETIC as _KIN
            self.phase = _KIN
            self._kinetic_steps = 0
            self.actor.set_hidden(self._start_kinetic.get("clone_hidden"))
            self._prev_res = np.asarray(self._start_kinetic.get("prev_res", np.zeros(ACT_DIM)), np.float64).copy()

    def _augmented_state(self, obs: np.ndarray) -> np.ndarray:
        """[ norm(41-D obs) · frozen clone GRU hidden · previous residual ] — the causal state δ_ψ conditions on. The clone
        hidden is read AFTER the clone processed this obs (h_clone,t)."""
        return np.concatenate([self.actor.norm.apply(obs), self.actor.hidden_vec(), self._prev_res]).astype(np.float64)

    def _transport_action(self, rl: Any, obs: np.ndarray) -> np.ndarray:
        u_clone = np.clip(np.asarray(self.actor.act(obs), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)   # sets the clone hidden
        aug = self._augmented_state(obs)
        r = np.clip(np.asarray(self.residual_fn(aug), np.float64).ravel()[:ACT_DIM], -1.0, 1.0)
        u = np.clip(u_clone + self.bounds.alpha * r, -1.0, 1.0)
        self.aug_trace.append((aug, r.copy()))
        self.residual_trace.append({"residual": [round(float(x), 5) for x in r], "u": [round(float(x), 5) for x in u]})
        self._prev_res = r
        return u


def deterministic_residual(policy: nn.Module) -> Callable[[np.ndarray], np.ndarray]:
    """A residual_fn using the policy's MEAN action (deploy / evaluation)."""
    policy.eval()

    def fn(aug: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return policy.mean_action(torch.as_tensor(aug, dtype=torch.float32).view(1, -1))[0].numpy()
    return fn


def exploring_residual(policy: nn.Module, algo: str, expl_noise: float, rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray]:
    """A residual_fn with exploration: SAC samples the squashed Gaussian; TD3 adds clipped Gaussian noise to the mean. Records
    the ACTUAL action taken (the behaviour action stored in replay)."""
    def fn(aug: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(aug, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            if algo == "sac":
                return policy.sample(x)[0][0].numpy()
            a = policy.mean_action(x)[0].numpy()
        return np.clip(a + rng.normal(0, expl_noise, ACT_DIM).astype(np.float32), -1.0, 1.0)
    return fn
