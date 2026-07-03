"""SE-SGformer (AAAI 2025), in-protocol reimplementation.

Distinction from SGT: SE-SGformer prepends a **structural encoding** to node
representations before the transformer stack. Here the structural encoding is the
per-node signed-degree profile ``log1p([deg_pos, deg_neg])`` (train-only),
projected to the model width and added to the learned embedding. The transformer
blocks are the sign-biased :class:`sgt.SGTBlock` (reused, no duplication).

Context: ``(nbrs, sgns, struct)`` — signed neighbour lists (train-only) plus the
precomputed structural-encoding tensor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .registry import GraphMeta, HParams, SignedLinkBaseline, SignedLinkModule, register
from .sgt import SGTBlock, build_signed_neighbours


def _signed_degree_encoding(edges, signs, n_nodes: int,
                            device: torch.device) -> torch.Tensor:
    """``(n_nodes, 2)`` tensor ``log1p([deg_pos, deg_neg])`` from train edges."""
    deg = np.zeros((n_nodes, 2), dtype=np.float32)  # [:,0]=pos, [:,1]=neg
    # Sign 0 = topology-only (R_topo): contributes to neither signed-degree bin
    # (its sign is withheld); it still enters message passing via the neighbour
    # lists. Only ±1 edges carry a signed degree.
    for sign_val, col in ((1, 0), (-1, 1)):
        e = edges[signs == sign_val]
        np.add.at(deg, (e[:, 0], col), 1.0)
        np.add.at(deg, (e[:, 1], col), 1.0)
    return torch.from_numpy(np.log1p(deg)).to(device)


class SESGformerModel(SignedLinkModule):
    def __init__(self, n_nodes: int, hidden_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.struct_proj = nn.Linear(2, hidden_dim)
        self.blocks = nn.ModuleList(
            [SGTBlock(hidden_dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, nbrs, sgns, struct) -> torch.Tensor:
        h = self.node_embed.weight + self.struct_proj(struct)
        for block in self.blocks:
            h = block(h, nbrs, sgns)
        return self.ln_f(h)


@register("sesgformer")
class SESGformerBaseline(SignedLinkBaseline):
    def build_model(self, meta: GraphMeta, hp: HParams):
        return SESGformerModel(meta.n_nodes, hidden_dim=hp.hidden,
                               n_heads=hp.n_heads, n_layers=hp.n_layers)

    def build_context(self, edges, signs, n_nodes, device):
        nbrs, sgns = build_signed_neighbours(edges, signs, n_nodes)
        struct = _signed_degree_encoding(edges, signs, n_nodes, device)
        return nbrs, sgns, struct

    def default_hparams(self) -> HParams:
        return HParams(hidden=32, n_layers=2, n_heads=4)
