"""Structural (star-expansion) entropy H★ of an HSiKAN policy's hypergraph activation — the "novel bet".

H★ measures how *diffusely* the policy reasons over the robot's kinematic structure: the Shannon entropy of the
per-vertex activation-energy distribution after HSiKAN message passing (before the mean-pool). High H★ ⇒ every
link is engaged (diffuse reasoning); low H★ ⇒ a few links dominate (concentrated). It is the **third** signal of
the exploration seat (alongside SAC's policy-entropy ``α·H(π)`` and critic-ensemble disagreement ``Var_k Q``) —
distinct in that it reads the *structure* of the policy's computation, not its action distribution.

It is differentiable in the activations, so it can augment the RL objective exactly where ``ent_coef·H(π)`` sits
(``StrategySpec.ent_coef``). The discriminating question (a real-topology A/B, never cart-pole): does H★ feedback
beat action-entropy *and* critic-disagreement on exploration? See
``docs/plans/2026-06-23-rl-scenario-ablation-entropy/`` (Stage 3).

The activation source is :meth:`hymeko_rl.agents.policy.HSiKANBackbone.node_activations`. The current ``hg_state`` is
the signed vertex×vertex kinematic adjacency; once the engine's transitive-import resolver exposes the canonical
``.hymeko`` star-expansion (with hyperedge-hub nodes), H★ extends to the full expanded node set unchanged in form
(see memory ``project-engine-transitive-imports``).
"""
from __future__ import annotations

import math
from typing import Protocol

import torch


class StructuralBackbone(Protocol):
    """A backbone that exposes per-vertex activations — i.e. the HSiKAN/Gömb hypergraph backbone.

    The MLP baseline deliberately does NOT satisfy this: H★ is defined only over the structural backbone."""

    def node_activations(self, x: torch.Tensor) -> torch.Tensor: ...


def star_expansion_entropy(node_activations: torch.Tensor, *, eps: float = 1e-8,
                           normalize: bool = True) -> torch.Tensor:
    """H★ from per-vertex HSiKAN activations: entropy of the activation-energy distribution over the vertices.

    The per-vertex energy is the L2 norm of each vertex's hidden vector; normalized over the vertices it is a
    probability distribution whose Shannon entropy is H★.

    # Preconditions ``node_activations`` is ``(B, N, H)`` (per-vertex hidden, pre-pool) with ``N >= 1``.
    # Postconditions returns ``(B,)`` non-negative entropies; if ``normalize`` and ``N > 1`` they lie in
      ``[0, 1]`` (``1`` = uniform energy over the N vertices, ``0`` = one vertex carries all energy).
    # Errors ``ValueError`` if the input is not rank-3.
    """
    if node_activations.dim() != 3:
        raise ValueError(f"expected (B, N, H) per-vertex activations; got {tuple(node_activations.shape)}")
    n = node_activations.shape[1]
    energy = node_activations.norm(dim=-1)                       # (B, N) per-vertex activation energy >= 0
    p = energy / (energy.sum(dim=-1, keepdim=True) + eps)        # distribution over the star-expansion vertices
    entropy: torch.Tensor = -(p * (p + eps).log()).sum(dim=-1)   # (B,) Shannon entropy (nats)
    if normalize and n > 1:
        entropy = entropy / math.log(n)                         # → [0, 1] (fraction of the max log N)
    return entropy


def mean_star_entropy(backbone: StructuralBackbone, obs: torch.Tensor, *, normalize: bool = True,
                      ) -> torch.Tensor:
    """Convenience: mean H★ over a batch, read straight from an HSiKAN ``backbone`` and an obs batch.

    # Preconditions ``backbone`` exposes ``node_activations(obs) -> (B, N, H)`` (i.e. an
      :class:`~hymeko_rl.agents.policy.HSiKANBackbone`); ``obs`` is ``(B, N, in_feat)``.
    # Postconditions returns a scalar tensor (mean over the batch), differentiable in the backbone params.
    """
    activations = backbone.node_activations(obs)
    mean_entropy: torch.Tensor = star_expansion_entropy(activations, normalize=normalize).mean()
    return mean_entropy
