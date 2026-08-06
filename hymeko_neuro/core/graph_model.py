"""The general HSiKAN under the SIGNED-GRAPH convention.

Same body as the RL line — :class:`~hymeko_neuro.core.backbone.SignedKANBackbone` (CR signed message passing + highway +
weighted incidence) — but with the **signed-graph** input adapter (transductive node embeddings) and the
**edge-sign** head (:class:`~hymeko_neuro.core.heads.EdgeSignHead`). The only differences from the RL model are the
*input adapter* (embeddings vs dynamic features) and the *head* (edge-sign vs actor/critic); the CR body is
identical. This is the "one general HSiKAN, changeable head" unification — and it lets a signed-graph task run
on the **CR** spline that defines HSiKAN (rather than the legacy triad+B-spline layer).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import SignedKANBackbone
from .backends import AggregationBackend
from .heads import EdgeSignHead


class SignedGraphHSiKAN(nn.Module):
    """Transductive node embeddings → CR signed message passing → edge-sign logits.

    # Preconditions ``n_nodes >= 1``; ``a_pos``/``a_neg`` are ``(n_nodes, n_nodes)`` (dense for small graphs,
      or pass a sparse-capable ``backend`` for large ones); ``hidden, n_layers >= 1``.
    # Postconditions ``forward(edges (E, 2)) -> logits (E, n_classes)``.
    """

    def __init__(self, n_nodes: int, a_pos: torch.Tensor, a_neg: torch.Tensor, *, hidden: int = 32,
                 n_layers: int = 2, incidence: str = "fixed", activation: str = "cr", skip: str = "none",
                 n_classes: int = 1, backend: AggregationBackend | None = None) -> None:
        super().__init__()
        if n_nodes < 1:
            raise ValueError(f"n_nodes must be >= 1; got {n_nodes}")
        self.embed = nn.Embedding(n_nodes, hidden)             # transductive features (the input adapter)
        self.backbone = SignedKANBackbone(hidden, hidden, n_layers, a_pos, a_neg, incidence=incidence,
                                          activation=activation, skip=skip, backend=backend)
        self.head = EdgeSignHead(hidden, n_classes=n_classes)

    def node_representations(self) -> torch.Tensor:
        """Per-node reps after the CR signed message passing — ``(n_nodes, hidden)``."""
        x = self.embed.weight.unsqueeze(0)                    # (1, N, hidden)
        reps: torch.Tensor = self.backbone.node_activations(x).squeeze(0)
        return reps

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.head(self.node_representations(), edges)
        return logits
