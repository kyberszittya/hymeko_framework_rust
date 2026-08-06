"""SB3-style flat-observation critic — early obs+action concatenation, for MetaWorld / flat-vector SAC.

The default :class:`hymeko_rl.train.ddpg.QCritic` fuses the action **late** (obs -> backbone -> feat, then the
action is concatenated with a single obs+action layer). On flat manipulation observations that under-discriminates
the action — ``dQ/da`` is weak — so the actor cannot do the fine control needed to *hold* a contact, producing a
**reach-then-regress** failure. Demonstrated on Coffee-Push S1 (2026-07-18 cross-impl audit): our SAC 0/4 stable
contact vs SB3 4/4; swapping in this early-concat critic restored reach-and-hold (0/4 -> 2/4). This critic
concatenates obs+action at the **input** into a ``[hidden, hidden]`` MLP (the SB3 convention), restoring the
action gradient.

Preconditions: flat (1-D per item) observations of width ``obs_dim``; use the structural ``QCritic`` for the 2-D
hypergraph obs. Pairs with ``reward_norm=False`` (the calibration fix; reward normalisation inflates the critic on
dense rewards — Q-vs-MC bias +60 vs +5).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EarlyConcatCritic(nn.Module):
    """SB3-style ``Q(s,a)``: concatenate obs+action at the input, then a ``[hidden, hidden]`` MLP -> 1.

    # Preconditions ``obs`` ``(B, obs_dim)``, ``action`` ``(B, action_dim)``. # Postconditions returns ``(B,)``
      (matching :class:`QCritic`)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        if obs_dim < 1 or action_dim < 1:
            raise ValueError(f"obs_dim/action_dim must be >= 1; got {obs_dim}/{action_dim}")
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q: torch.Tensor = self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)
        return q


def build_flat_sac(obs_dim: int, action_dim: int, action_scale: float, *, hidden: int = 256,
                   n_critics: int = 2, device: "torch.device | str" = "cpu",
                   ) -> "tuple[nn.Module, list[EarlyConcatCritic]]":
    """Flat-obs SAC: the repo's pooled ``mlp`` actor + **early-concat** critics (the Coffee-Push-corrected stack).

    Returns ``(actor, [EarlyConcatCritic, ...])`` — drop-in for :func:`train_sac`, which only calls ``c(s, a)``."""
    from hymeko_rl.train.sac import build_sac
    actor, _ = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim, action_dim=action_dim,
                         action_scale=action_scale, hidden=hidden, device=device)
    critics = [EarlyConcatCritic(obs_dim, action_dim, hidden).to(device) for _ in range(max(1, n_critics))]
    return actor, critics
