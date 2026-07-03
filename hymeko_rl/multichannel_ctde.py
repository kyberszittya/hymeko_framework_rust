"""3-agent collaborative CTDE with a MultiTreeChannel coordination channel.

Two actor graphs (one per arm) + one centralized critic graph, all over the joint robot hypergraph, cross-coupled
by a :class:`~hymeko_rl.tree_channel.MultiTreeChannel`: the arms coordinate **through the shared coin/zone
couplings**, and the critic informs both. This is the *learned, state-adaptive* coordination the fixed-backbone
CTDE lacked — the one that delivered all-or-nothing (single ≥ collab; the variance was the failure). Drop-in for
the single-agent BC / PPO / eval machinery: exposes ``action_mean`` / ``act`` / ``value`` / ``evaluate`` /
``n_parameters`` exactly like :class:`~hymeko_rl.collaborative.CTDEActorCritic`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from hymeko_rl.collaborative import arm_action_partition
from hymeko_rl.policy import _BACKBONES
from hymeko_rl.tree_channel import MultiTreeChannel, adjacency_from_hg


class MultiChannelCTDE(nn.Module):
    """``n_actors`` actor graphs + 1 critic graph, coupled by a MultiTreeChannel on the joint hypergraph.

    # Preconditions ``len(actor_backbones) == len(agent_action_dims)``; each backbone exposes
      ``node_activations(obs) -> (B, N, hidden)``. # Postconditions ``action_mean`` → ``(B, sum(dims))``;
      ``value`` → ``(B,)`` (centralized, sees the whole hypergraph after the channel)."""

    def __init__(self, actor_backbones: "list[nn.Module]", critic_backbone: nn.Module, hidden: int,
                 adj: torch.Tensor, agent_action_dims: "tuple[int, ...]", *, log_std_init: float = -1.6) -> None:
        super().__init__()
        if len(actor_backbones) != len(agent_action_dims) or not agent_action_dims:
            raise ValueError("one actor backbone per action-dim group; dims non-empty")
        self.actor_backbones = nn.ModuleList(actor_backbones)
        self.critic_backbone = critic_backbone
        self.n_actors = len(agent_action_dims)
        self.agent_action_dims = tuple(agent_action_dims)
        self.channel = MultiTreeChannel(hidden, adj, n_agents=self.n_actors + 1)
        self.action_heads = nn.ModuleList([nn.Linear(hidden, d) for d in agent_action_dims])
        self.critic = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((sum(agent_action_dims),), float(log_std_init)))

    def _pooled(self, obs: torch.Tensor) -> "list[torch.Tensor]":
        feats: list[torch.Tensor] = [bb.node_activations(obs) for bb in self.actor_backbones]   # type: ignore[operator]
        feats.append(self.critic_backbone.node_activations(obs))                                # type: ignore[operator]
        return [o.mean(dim=1) for o in self.channel(feats)]     # coordination channel, then pool -> (B, hidden)

    def _heads(self, obs: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        pooled = self._pooled(obs)
        means = torch.cat([self.action_heads[i](pooled[i]) for i in range(self.n_actors)], dim=-1)
        value: torch.Tensor = self.critic(pooled[-1]).squeeze(-1)
        return means, value

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self._heads(obs)[0]

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self._heads(obs)[1]

    def _dist(self, means: torch.Tensor) -> Any:
        return torch.distributions.Normal(means, self.log_std.exp().expand_as(means))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        means, value = self._heads(obs)
        dist = self._dist(means)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), value

    def evaluate(self, obs: torch.Tensor,
                 action: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        means, value = self._heads(obs)
        dist = self._dist(means)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeterministicMultiChannelActor(nn.Module):
    """Deterministic (TD3) channel-coupled CTDE actor: ``n_actors`` per-arm backbones coordinated by a
    :class:`~hymeko_rl.tree_channel.MultiTreeChannel` (the arms talk **through the shared coin/zone couplings**
    on the kinematic hypergraph), then per-arm deterministic heads → ``action_scale·tanh(concat)``.

    Unlike :class:`MultiChannelCTDE` this has **no** ``log_std``/``value`` (off-policy uses a *separate*
    :class:`~hymeko_rl.ddpg.QCritic`); the coordination structure under test lives entirely in the actor's
    reasoning (the channel between the two per-arm backbones) — per-arm output *heads* over one backbone would be
    equivalent to a single head, so the channel is what makes this non-vacuously collaborative.

    Satisfies the :func:`~hymeko_rl.ddpg.train_offpolicy` actor contract by duck-typing: ``forward``/
    ``action_mean`` (alias, for BC + greedy eval), ``action_scale``, ``action_dim``. It exposes **no** single
    ``backbone`` attribute, so the trainer's shared-trunk detection correctly takes the non-shared path.

    # Preconditions ``len(actor_backbones) == len(agent_action_dims) >= 1``; each backbone exposes
      ``node_activations(obs) -> (B, N, hidden)``; ``adj`` is ``(N, N)``.
    # Postconditions ``forward(obs) -> (B, sum(agent_action_dims))`` bounded in ``[-action_scale, action_scale]``.
    """

    def __init__(self, actor_backbones: "list[nn.Module]", hidden: int, adj: torch.Tensor,
                 agent_action_dims: "tuple[int, ...]", action_scale: float) -> None:
        super().__init__()
        if len(actor_backbones) != len(agent_action_dims) or not agent_action_dims:
            raise ValueError("one actor backbone per action-dim group; dims non-empty")
        if any(d < 1 for d in agent_action_dims):
            raise ValueError(f"all agent action dims must be >= 1; got {agent_action_dims}")
        self.actor_backbones = nn.ModuleList(actor_backbones)
        self.n_actors = len(agent_action_dims)
        self.agent_action_dims = tuple(agent_action_dims)
        self.channel = MultiTreeChannel(hidden, adj, n_agents=self.n_actors)
        self.action_heads = nn.ModuleList([nn.Linear(hidden, d) for d in agent_action_dims])
        self.action_scale = float(action_scale)

    @property
    def action_dim(self) -> int:
        """The joint action width — the off-policy actor contract (replaces ``head.out_features``)."""
        return sum(self.agent_action_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = [bb.node_activations(obs) for bb in self.actor_backbones]   # type: ignore[operator]  # (B,N,hidden)
        pooled = [o.mean(dim=1) for o in self.channel(feats)]              # coordinate, then pool → (B, hidden)
        means = torch.cat([self.action_heads[i](pooled[i]) for i in range(self.n_actors)], dim=-1)
        out: torch.Tensor = self.action_scale * torch.tanh(means)
        return out

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Alias of :meth:`forward` — the deterministic policy IS its mean (for BC + greedy eval reuse)."""
        return self.forward(obs)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_collaborative_offpolicy(env: Any, *, kind: str = "hsikan", hidden: int = 64, n_layers: int = 2,
                                  incidence: str = "fixed", skip: str = "highway", n_critics: int = 2,
                                  critic_layernorm: bool = True,
                                  ) -> "tuple[DeterministicMultiChannelActor, list[nn.Module]]":
    """Build the off-policy (TD3) collaborative coin-toss: a :class:`DeterministicMultiChannelActor` (per-arm
    backbones + coordination channel) + ``n_critics`` centralized :class:`~hymeko_rl.ddpg.QCritic` twins over the
    joint action. LayerNorm critics default ON (the round-coin late Q-overestimation collapse — anti-overestimation
    matters). Drop-in for :func:`~hymeko_rl.ddpg.train_offpolicy` / the BC→TD3+BC pipeline.

    # Preconditions ``kind in _BACKBONES``; ``env`` exposes ``hg`` + arm-separable actuators; ``n_critics >= 1``.
    """
    from gymnasium.spaces import Box

    from hymeko_rl.ddpg import QCritic
    if kind not in _BACKBONES:
        raise ValueError(f"unknown backbone kind {kind!r}; expected one of {sorted(_BACKBONES)}")
    if n_critics < 1:
        raise ValueError(f"n_critics must be >= 1; got {n_critics}")
    feat = int(env.observation_space.shape[1])
    dims = tuple(s.stop - s.start for s in arm_action_partition(env))
    adj = adjacency_from_hg(env.hg)
    space = env.action_space
    assert isinstance(space, Box)
    scale = float(np.max(np.abs(space.high)))

    def _bb() -> "tuple[nn.Module, int]":
        mod, fdim = _BACKBONES[kind](feat, hg_state=env.hg, hidden=hidden, n_layers=n_layers,
                                     incidence=incidence, skip=skip)
        return mod, int(fdim)

    actors = [_bb()[0] for _ in dims]
    action_dim = sum(dims)
    actor = DeterministicMultiChannelActor(actors, hidden, adj, dims, scale)
    critics: list[nn.Module] = []
    for _ in range(n_critics):                              # centralized twins: each sees the FULL joint action
        cb, cfeat = _bb()
        critics.append(QCritic(cb, cfeat, action_dim, layernorm=critic_layernorm))
    return actor, critics


def build_multichannel_collaborative(env: Any, *, kind: str = "hsikan", hidden: int = 64, n_layers: int = 2,
                                     incidence: str = "fixed", skip: str = "highway",
                                     log_std_init: float = -1.6) -> MultiChannelCTDE:
    """Build the 3-agent (per-arm actors + critic) MultiTreeChannel CTDE for ``env`` — the coin-toss coordination
    architecture. ``kind`` selects the shared per-agent backbone via the policy registry (``sa_hsikan`` is the
    cheaper Bᴸ-collapse — same per-node interface, ~half the per-step cost, so more PPO refine fits the wall).
    HSiKAN-only kwargs (``n_layers``/``incidence``/``skip``) are ignored by structural backbones.
    # Preconditions ``kind in _BACKBONES``; ``env`` exposes ``hg`` + arm-separable actuators
    (:func:`arm_action_partition`)."""
    if kind not in _BACKBONES:
        raise ValueError(f"unknown backbone kind {kind!r}; expected one of {sorted(_BACKBONES)}")
    feat = int(env.observation_space.shape[1])
    dims = tuple(s.stop - s.start for s in arm_action_partition(env))
    adj = adjacency_from_hg(env.hg)

    def _bb() -> nn.Module:
        mod: nn.Module = _BACKBONES[kind](feat, hg_state=env.hg, hidden=hidden, n_layers=n_layers,
                                          incidence=incidence, skip=skip)[0]
        return mod

    actors = [_bb() for _ in dims]
    return MultiChannelCTDE(actors, _bb(), hidden, adj, dims, log_std_init=log_std_init)
