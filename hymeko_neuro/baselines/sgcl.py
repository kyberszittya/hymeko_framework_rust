"""SGCL — Signed Graph Contrastive Learning (Shu et al. 2022), in-protocol reimpl.

Distinction from SGCN: the encoder is the same Derr-style B/U signed GCN (reused
verbatim from :mod:`sgcn_model`, no duplication), but training adds a **sign-aware
contrastive** objective on a projection head — positive-edge endpoints are pulled
together in projection space, negative-edge endpoints pushed apart. This is the
SGCL-vs-SGCN architectural delta, expressed through the shared loop's
``aux_loss`` hook (weighted by ``HParams.aux_weight``).

Strict + audit clean by construction: the contrastive term consumes only the
training edges/signs handed to the loop; nothing test-side enters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import GraphMeta, HParams, SignedLinkBaseline, SignedLinkModule, register
from .sgcn_model import SGCN, build_signed_adj


class SGCLModel(SignedLinkModule):
    """SGCN encoder + projection head; BCE + sign-aware contrastive aux loss."""

    def __init__(self, n_nodes: int, hidden_dim: int = 32, n_layers: int = 2,
                 proj_dim: int = 32, temperature: float = 0.5):
        super().__init__()
        self.encoder = SGCN(n_nodes, hidden_dim=hidden_dim, n_layers=n_layers)
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, A_pos, A_neg) -> torch.Tensor:
        return self.encoder.encode_nodes(A_pos, A_neg)  # (n, 2*hidden)

    def aux_loss(self, z, edges_t, signs_t) -> torch.Tensor:
        """Sign-aware contrastive: cosine sim of projected endpoints should be
        high for positive edges, low for negative ones."""
        p = F.normalize(self.proj(z), dim=-1)
        u, v = edges_t[:, 0], edges_t[:, 1]
        sim = (p[u] * p[v]).sum(-1) / self.temperature
        target = (signs_t > 0).float()
        return F.binary_cross_entropy_with_logits(sim, target)


@register("sgcl")
class SGCLBaseline(SignedLinkBaseline):
    def build_model(self, meta: GraphMeta, hp: HParams):
        return SGCLModel(meta.n_nodes, hidden_dim=hp.hidden, n_layers=hp.n_layers)

    def build_context(self, edges, signs, n_nodes, device):
        return build_signed_adj(edges, signs, n_nodes, device)

    def default_hparams(self) -> HParams:
        return HParams(hidden=32, n_layers=2, aux_weight=0.5)
