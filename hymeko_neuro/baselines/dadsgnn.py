"""DADSGNN — Depth-Augmented Dual-path Signed GNN (Nat. Sci. Rep. 2025), reimpl.

Distinction from SGCN: instead of reading out only the final layer, DADSGNN keeps
the B/U embedding at **every depth** and fuses them with a learned per-node
attention over depths (jumping-knowledge style) — the "depth-augmented"
propagation. The per-depth signed convolution is the reused
:class:`sgcn_model.SGCNLayer` (no duplication); the depth-attention readout is the
DADSGNN-specific structure.

Context: train-only signed adjacency, reused from :func:`sgcn_model.build_signed_adj`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import GraphMeta, HParams, SignedLinkBaseline, SignedLinkModule, register
from .sgcn_model import SGCNLayer, build_signed_adj


class DADSGNNModel(SignedLinkModule):
    def __init__(self, n_nodes: int, hidden_dim: int = 32, n_layers: int = 3):
        super().__init__()
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.layers = nn.ModuleList([
            SGCNLayer(in_dim=hidden_dim, out_dim=hidden_dim, first=(i == 0))
            for i in range(n_layers)
        ])
        # Per-node attention over depth-wise z = [h_B ; h_U] (each 2*hidden).
        self.depth_attn = nn.Linear(2 * hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, A_pos, A_neg) -> torch.Tensor:
        x = self.node_embed.weight
        h_B, h_U = x, x
        depth_z = []
        for layer in self.layers:
            h_B, h_U = layer(h_B, h_U, A_pos, A_neg)
            depth_z.append(torch.cat([h_B, h_U], dim=-1))   # (n, 2*hidden)
        Z = torch.stack(depth_z, dim=1)                     # (n, L, 2*hidden)
        w = torch.softmax(self.depth_attn(Z).squeeze(-1), dim=1)  # (n, L)
        return (w.unsqueeze(-1) * Z).sum(dim=1)             # (n, 2*hidden)


@register("dadsgnn")
class DADSGNNBaseline(SignedLinkBaseline):
    def build_model(self, meta: GraphMeta, hp: HParams):
        return DADSGNNModel(meta.n_nodes, hidden_dim=hp.hidden, n_layers=hp.n_layers)

    def build_context(self, edges, signs, n_nodes, device):
        return build_signed_adj(edges, signs, n_nodes, device)

    def default_hparams(self) -> HParams:
        return HParams(hidden=32, n_layers=3)
