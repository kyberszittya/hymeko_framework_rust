"""The signed-KAN backbone: incidence handling + a stack of :class:`~hymeko_neuro.core.layer.SignedKANLayer` + pooling.

This is the consumer-facing module — what ``hymeko_rl``'s ``HSiKANBackbone`` becomes (phase 2). It owns the signed
incidence ``A±`` (the three modes below) and the inter-layer stack/pool; the per-layer aggregation goes through the
pluggable backend. The consumer supplies the signed adjacency (``a_pos``/``a_neg``) — e.g. the RL line builds it from
the kinematic hypergraph — keeping the core decoupled from any one graph source.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backends import AggregationBackend
from .layer import SignedKANLayer

INCIDENCE_MODES = ("fixed", "learned", "weighted")
POOL_MODES = ("mean", "sum")


class SignedKANBackbone(nn.Module):
    """Stacked signed-KAN layers over a signed incidence → pooled graph embedding.

    Input is per-vertex features ``(B, N, in_feat)``; output is ``(B, hidden)`` (pooled). The incidence mode sets
    how ``A±`` carries its arc weights:

    - ``"fixed"`` — buffers: the binary kinematic structure (sign in {+,-}, magnitude row-normalised).
    - ``"learned"`` — full ``A±`` trainable (any real value on any vertex pair; sparsity not preserved).
    - ``"weighted"`` — the structural mask is fixed, a per-arc weight is learned (init 1.0 → parity with
      ``"fixed"``): free real weights on the *real* arcs — the signed-hypergraph premise, sparsity preserved.

    # Preconditions ``in_feat, hidden, n_layers >= 1``; ``a_pos``/``a_neg`` are square ``(N, N)`` and same shape;
      ``incidence in INCIDENCE_MODES``; ``pool in POOL_MODES``.
    # Postconditions ``forward`` returns ``(B, hidden)``; ``node_activations`` returns ``(B, N, hidden)``.
    """

    def __init__(self, in_feat: int, hidden: int, n_layers: int,
                 a_pos: torch.Tensor, a_neg: torch.Tensor, *, incidence: str = "fixed",
                 activation: str = "cr", skip: str = "none", grid: int = 5, pool: str = "mean",
                 backend: AggregationBackend | None = None) -> None:
        super().__init__()
        if in_feat < 1 or hidden < 1 or n_layers < 1:
            raise ValueError(f"in_feat/hidden/n_layers must be >= 1; got {in_feat}/{hidden}/{n_layers}")
        if incidence not in INCIDENCE_MODES:
            raise ValueError(f"incidence must be one of {INCIDENCE_MODES}; got {incidence!r}")
        if pool not in POOL_MODES:
            raise ValueError(f"pool must be one of {POOL_MODES}; got {pool!r}")
        if a_pos.shape != a_neg.shape or a_pos.dim() != 2 or a_pos.shape[0] != a_pos.shape[1]:
            raise ValueError(f"a_pos/a_neg must be equal square (N, N); got {tuple(a_pos.shape)}/{tuple(a_neg.shape)}")
        self.incidence = incidence
        self.pool = pool
        self.n_vertices = a_pos.shape[0]
        if incidence == "learned":
            self.a_pos = nn.Parameter(a_pos.clone())
            self.a_neg = nn.Parameter(a_neg.clone())
        else:
            self.register_buffer("a_pos", a_pos)
            self.register_buffer("a_neg", a_neg)
        if incidence == "weighted":
            self.w_pos_arc = nn.Parameter(torch.ones_like(a_pos))
            self.w_neg_arc = nn.Parameter(torch.ones_like(a_neg))
        dims = [in_feat] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            [SignedKANLayer(dims[i], dims[i + 1], activation=activation, skip=skip, grid=grid, backend=backend)
             for i in range(n_layers)])

    def _effective_adj(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The signed adjacency the forward pass uses. ``"weighted"`` scales the fixed structural mask by the
        learned per-arc weights (real weights on real arcs only); otherwise the stored ``a_pos``/``a_neg``."""
        if self.incidence == "weighted":
            return self.a_pos * self.w_pos_arc, self.a_neg * self.w_neg_arc
        return self.a_pos, self.a_neg

    def node_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Per-vertex activations after the layers, before pooling (the structural representation). ``(B, N, hidden)``."""
        if x.dim() != 3 or x.shape[1] != self.n_vertices:
            raise ValueError(f"expected (B, {self.n_vertices}, in_feat); got {tuple(x.shape)}")
        a_pos, a_neg = self._effective_adj()
        for layer in self.layers:
            x = layer(x, a_pos, a_neg)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.node_activations(x)
        pooled: torch.Tensor = h.mean(dim=1) if self.pool == "mean" else h.sum(dim=1)
        return pooled
