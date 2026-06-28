"""Rate-asymmetric dual-loop controller — Kato's reflexive/deliberative split, made compute-honest.

Measured (2026-06-25): the HSiKAN backbone is ~2.4 ms/step (~420 Hz) on one CPU thread; the MLP is ~0.16 ms
(~6400 Hz) — a ~15x latency asymmetry. So the two discriminators cannot (and need not) run at the same rate.
This module runs them at two rates:

  - the **reflex** (fast, MLP) every control step — reactivity + a hard real-time guarantee;
  - the **deliberation** (slow, HSiKAN) every ``N`` steps, its output *held* as a slowly-varying conditioning
    signal the reflex consumes between deliberations.

``action = head( reflex(obs) ⊕ held_deliberation_context )``. The biological motor hierarchy (slow cortical
deliberation over fast spinal/cerebellar reflex). The prototype to de-risk the dual-discriminator plan
(``docs/plans/2026-06-24-kato-collab-dual-discriminator``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from hymeko_rl.policy import hsikan_backbone, mlp_backbone

if TYPE_CHECKING:
    from hymeko_rl.hypergraph_state import HypergraphState


class DualRateController(nn.Module):
    """Fast reflex backbone + slow deliberative backbone, fused by a shared actor/critic head.

    The deliberation context is passed in explicitly (computed by :meth:`deliberate`), so the module is
    stateless — the *cadence* (how often deliberation is recomputed) is the caller's loop, driven by
    :class:`RateAsymmetricLoop` for inference or every-step for training.

    # Preconditions ``reflex_feat, delib_feat, action_dim >= 1``; reflex reads the flat obs, deliberative the
      per-vertex graph obs (both from the same observation).
    # Postconditions ``action_mean(obs, context)`` returns ``(B, action_dim)``; ``deliberate(obs)`` returns the
      held context ``(B, delib_feat)``.
    """

    def __init__(self, reflex: nn.Module, reflex_feat: int, deliberative: nn.Module, delib_feat: int,
                 action_dim: int, *, log_std_init: float = -1.6) -> None:
        super().__init__()
        if reflex_feat < 1 or delib_feat < 1 or action_dim < 1:
            raise ValueError(f"feat/action dims must be >= 1; got {reflex_feat}/{delib_feat}/{action_dim}")
        self.reflex = reflex
        self.deliberative = deliberative
        self.delib_feat = delib_feat
        self.actor_mean = nn.Linear(reflex_feat + delib_feat, action_dim)
        self.critic = nn.Linear(reflex_feat + delib_feat, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def deliberate(self, obs: torch.Tensor) -> torch.Tensor:
        """The slow loop: structural reasoning → conditioning context ``(B, delib_feat)``. Called every N steps."""
        ctx: torch.Tensor = self.deliberative(obs)
        return ctx

    def _ctx(self, obs: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        """Resolve the deliberation context: use the held ``context`` if given (the rate-asymmetric eval path),
        else deliberate now (every-step — the training / ``ActorCritic``-compatible path)."""
        return self.deliberate(obs) if context is None else context

    def _features(self, obs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        reflex: torch.Tensor = self.reflex(obs)
        if context.shape[0] != reflex.shape[0]:
            context = context.expand(reflex.shape[0], -1)
        return torch.cat([reflex, context], dim=-1)

    def action_mean(self, obs: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        mean: torch.Tensor = self.actor_mean(self._features(obs, self._ctx(obs, context)))
        return mean

    def value(self, obs: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        v: torch.Tensor = self.critic(self._features(obs, self._ctx(obs, context))).squeeze(-1)
        return v

    def _dist(self, obs: torch.Tensor, context: torch.Tensor) -> Any:
        mean = self.action_mean(obs, context)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    @torch.no_grad()
    def act(self, obs: torch.Tensor,
            context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action; returns ``(action, log_prob, value)`` (rollout)."""
        ctx = self._ctx(obs, context)
        dist = self._dist(obs, ctx)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs, ctx)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor,
                 context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx = self._ctx(obs, context)
        dist = self._dist(obs, ctx)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(obs, ctx)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RateAsymmetricLoop:
    """Drives a :class:`DualRateController` at two rates: recompute the deliberation context every
    ``deliberate_every`` steps, hold it otherwise; the reflex runs every step. The inference/deployment path.

    # Preconditions ``deliberate_every >= 1``.
    # Postconditions ``act_mean(obs)`` returns the deterministic action each step; the deliberation context is
      recomputed exactly on steps ``0, N, 2N, …`` (count via :attr:`n_deliberations`).
    """

    def __init__(self, controller: DualRateController, deliberate_every: int = 4) -> None:
        if deliberate_every < 1:
            raise ValueError(f"deliberate_every must be >= 1; got {deliberate_every}")
        self.controller = controller
        self.deliberate_every = deliberate_every
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._context: torch.Tensor | None = None
        self.n_deliberations = 0

    @torch.no_grad()
    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        if self._context is None or self._step % self.deliberate_every == 0:
            self._context = self.controller.deliberate(obs)
            self.n_deliberations += 1
        self._step += 1
        return self.controller.action_mean(obs, self._context)


def build_dual_rate(obs_feat: int, action_dim: int, *, hg_state: "HypergraphState", n_vertices: int,
                    reflex_hidden: int = 64, delib_hidden: int = 64, delib_n_layers: int = 2,
                    incidence: str = "fixed", skip: str = "none", log_std_init: float = -1.6,
                    ) -> DualRateController:
    """Build the prototype: MLP reflex (flat obs) + HSiKAN deliberation (graph obs) + fused heads.

    Both read the same per-vertex observation ``(B, n_vertices, obs_feat)``; the reflex flattens it, the
    deliberative reads it as the kinematic hypergraph. # Preconditions ``obs_feat, action_dim, n_vertices >= 1``.
    """
    reflex, reflex_feat = mlp_backbone(n_vertices * obs_feat, hidden=reflex_hidden)
    deliberative, delib_feat = hsikan_backbone(obs_feat, hg_state=hg_state, hidden=delib_hidden,
                                               n_layers=delib_n_layers, incidence=incidence, skip=skip)
    return DualRateController(reflex, reflex_feat, deliberative, delib_feat, action_dim,
                             log_std_init=log_std_init)
