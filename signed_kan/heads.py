"""Changeable heads for the one general HSiKAN body.

The body (:class:`~signed_kan.backbone.SignedKANBackbone`: CR signed message passing + highway + weighted
incidence) is shared; only the **head** changes per convention:

- **RL convention** — the pooled graph embedding feeds an actor/critic head (lives in ``hymeko_rl.ActorCritic``).
- **Signed-graph convention** — per-node representations + target edges feed :class:`EdgeSignHead` for link-sign
  prediction.

This is the unification: one CR HSiKAN, two heads — rather than two forked layers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class GraphHead(nn.Module, ABC):
    """A readout from the body's per-node representations to a task output."""

    @abstractmethod
    def forward(self, node_reps: torch.Tensor, edges: torch.Tensor) -> torch.Tensor: ...


class EdgeSignHead(GraphHead):
    """Link-sign prediction head: combine the two endpoint representations of each target edge → sign logit.

    The standard signed-GCN readout (concatenate the endpoint reps, then a linear classifier) — the
    signed-graph convention's head of the general HSiKAN.

    # Preconditions ``hidden >= 1``, ``n_classes >= 1``.
    # Postconditions ``forward(node_reps (N, hidden), edges (E, 2)) -> logits (E, n_classes)``.
    """

    def __init__(self, hidden: int, *, n_classes: int = 1) -> None:
        super().__init__()
        if hidden < 1 or n_classes < 1:
            raise ValueError(f"hidden/n_classes must be >= 1; got {hidden}/{n_classes}")
        self.classifier = nn.Linear(2 * hidden, n_classes)

    def forward(self, node_reps: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if node_reps.dim() != 2:
            raise ValueError(f"node_reps must be (N, hidden); got {tuple(node_reps.shape)}")
        if edges.dim() != 2 or edges.shape[1] != 2:
            raise ValueError(f"edges must be (E, 2); got {tuple(edges.shape)}")
        u = node_reps[edges[:, 0]]
        v = node_reps[edges[:, 1]]
        logits: torch.Tensor = self.classifier(torch.cat([u, v], dim=-1))
        return logits
