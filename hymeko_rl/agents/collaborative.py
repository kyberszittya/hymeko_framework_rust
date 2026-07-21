"""Collaborative (cooperative multi-agent) reframe of Galambos — Kato's k-agent direction, minimal form.

Each **arm is an agent** controlling its own joints; the two agents cooperate to deliver the coin to the zone
and **share the team reward** (the canonical cooperative-MARL setup). Two reused seams make this nearly free:

  - the underlying action vector is already arm-separable (``[j1_left, j2_left, j1_right, j2_right]``), so a
    multi-agent *view* over :class:`PlanarGraspEnv` only has to split/merge it — no new physics, no env clone;
  - one **shared HSiKAN backbone** does the structural reasoning over the *whole* hypergraph (both arms + the
    coin/zone if ``task_graph``), feeding **per-arm action heads** (decentralized actors with shared reasoning)
    and **one centralized critic** — i.e. CTDE (centralized training, decentralized execution).

This realizes `project-actor-critic-shared-reasoning` (share the HSiKAN reasoning, per-agent heads) and is the
prototype for the dual-discriminator plan's collaborative-k-agent item
(``docs/plans/2026-06-24-kato-collab-dual-discriminator``). The policy's ``action_mean(obs)`` returns the
concatenated full action, so it stays drop-in for the existing BC / eval machinery.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np
import torch
import torch.nn as nn

from hymeko_rl.agents.policy import hsikan_backbone, mlp_backbone

if TYPE_CHECKING:
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv


def arm_action_partition(env: "PlanarGraspEnv") -> tuple[slice, ...]:
    """Partition the flat actuator vector into one contiguous slice per arm, by actuator-name suffix
    (``*_left`` / ``*_right``). For the Galambos two-arm layout this is ``(slice(0,2), slice(2,4))``.

    # Preconditions actuators are grouped contiguously per arm (true for the emitted/hand-authored Galambos).
    # Postconditions the returned slices tile ``range(env.n_actions)`` with no overlap.
    """
    names = [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(env.n_actions)]
    side = ["left" if n.endswith("left") else "right" if n.endswith("right") else "?" for n in names]
    parts: list[slice] = []
    i = 0
    while i < len(side):
        j = i
        while j < len(side) and side[j] == side[i]:
            j += 1
        parts.append(slice(i, j))
        i = j
    return tuple(parts)


class CollaborativeGalambos:
    """Cooperative multi-agent view of :class:`PlanarGraspEnv`: one agent per arm, shared structural
    observation (the full hypergraph — CTDE), shared team reward. ``step`` takes one action per agent and
    merges them into the underlying control vector.

    # Preconditions ``env`` is a constructed ``PlanarGraspEnv``. # Postconditions ``reset``/``step`` return one
      observation and one reward *per agent* (here identical — shared obs, team reward); ``info`` is the env's.
    # Invariants merging the per-agent actions reproduces exactly the single-agent control vector.
    """

    def __init__(self, env: "PlanarGraspEnv") -> None:
        self.env = env
        self.partition = arm_action_partition(env)
        self.n_agents = len(self.partition)
        self.agent_action_dims = tuple(s.stop - s.start for s in self.partition)

    def reset(self, *, seed: int | None = None) -> tuple[list[np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        return [obs] * self.n_agents, info

    def merge(self, actions: Sequence[np.ndarray]) -> np.ndarray:
        """Assemble the per-agent actions into the single control vector (the inverse of the partition)."""
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} per-agent actions; got {len(actions)}")
        merged = np.zeros(self.env.n_actions, dtype=np.float32)
        for sl, a in zip(self.partition, actions):
            merged[sl] = np.asarray(a, dtype=np.float32)
        return merged

    def step(self, actions: Sequence[np.ndarray],
             ) -> tuple[list[np.ndarray], list[float], bool, bool, dict[str, Any]]:
        obs, reward, term, trunc, info = self.env.step(self.merge(actions))
        return [obs] * self.n_agents, [float(reward)] * self.n_agents, term, trunc, info


class CTDEActorCritic(nn.Module):
    """Cooperative CTDE policy: one shared backbone (the structural reasoning) → per-agent action heads
    (decentralized actors) + one centralized critic. Each agent owns a disjoint action slice; the critic and
    the shared backbone are the *centralized* training signal.

    ``action_mean(obs)`` concatenates the per-agent means into the full action vector, so the policy is a
    drop-in for the single-agent BC / PPO / eval machinery (``behaviour_clone``, ``eval_delivery``).

    # Preconditions ``backbone`` maps obs → ``(B, feat_dim)``; ``agent_action_dims`` are ``>= 1``.
    # Postconditions ``action_mean`` → ``(B, sum(agent_action_dims))``; ``value`` → ``(B,)`` (centralized).
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, agent_action_dims: tuple[int, ...], *,
                 log_std_init: float = -1.6) -> None:
        super().__init__()
        if feat_dim < 1 or not agent_action_dims or any(d < 1 for d in agent_action_dims):
            raise ValueError(f"feat_dim>=1 and all agent dims>=1 required; got {feat_dim}, {agent_action_dims}")
        self.backbone = backbone
        self.agent_action_dims = agent_action_dims
        self.action_heads = nn.ModuleList([nn.Linear(feat_dim, d) for d in agent_action_dims])
        self.critic = nn.Linear(feat_dim, 1)
        self.log_std = nn.Parameter(torch.full((sum(agent_action_dims),), float(log_std_init)))

    def agent_means(self, obs: torch.Tensor) -> list[torch.Tensor]:
        """Per-agent action means (decentralized actors over the shared features)."""
        feat = self.backbone(obs)
        return [head(feat) for head in self.action_heads]

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(self.agent_means(obs), dim=-1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """The centralized critic's state value ``(B,)`` (sees the whole shared observation)."""
        v: torch.Tensor = self.critic(self.backbone(obs)).squeeze(-1)
        return v

    def _dist(self, obs: torch.Tensor) -> Any:
        mean = self.action_mean(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate(self, obs: torch.Tensor,
                 action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._dist(obs)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(obs)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_collaborative(env: "PlanarGraspEnv", *, kind: str = "hsikan", hidden: int = 64, n_layers: int = 2,
                        incidence: str = "fixed", skip: str = "highway",
                        log_std_init: float = -1.6) -> CTDEActorCritic:
    """Build the cooperative CTDE policy for ``env``: a shared ``kind`` backbone over the hypergraph + one
    action head per arm (from :func:`arm_action_partition`) + a centralized critic.

    # Preconditions ``kind in {"hsikan", "mlp"}``; ``env`` exposes ``hg`` + ``observation_space``.
    """
    feat = int(env.observation_space.shape[1])   # type: ignore[index]
    dims = tuple(s.stop - s.start for s in arm_action_partition(env))
    if kind == "mlp":
        backbone, fdim = mlp_backbone(env.hg.n_vertices * feat, hidden=hidden)
    elif kind == "hsikan":
        backbone, fdim = hsikan_backbone(feat, hg_state=env.hg, hidden=hidden, n_layers=n_layers,
                                         incidence=incidence, skip=skip)
    else:
        raise ValueError(f"unknown kind {kind!r}; expected 'hsikan' or 'mlp'")
    return CTDEActorCritic(backbone, fdim, dims, log_std_init=log_std_init)
