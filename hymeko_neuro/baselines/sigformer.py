"""SiGformer (2023), in-protocol reimplementation.

Distinction from SGT (sign-bias scalar) and SiGAT (bare motif attention):
SiGformer runs **separate positive and negative attention streams** inside a
pre-LayerNorm transformer block, then *gates* their fusion with the self
representation. Reuses :class:`sigat_model.MotifAttention` for each stream (no
duplication); the gate + pre-LN + FFN are the SiGformer-specific structure.

Context: per-node positive/negative neighbour buckets (train-only), reused from
:func:`sigat_model.build_neighbour_lists`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import GraphMeta, HParams, SignedLinkBaseline, SignedLinkModule, register
from .sigat_model import MotifAttention, build_neighbour_lists


class SiGformerBlock(nn.Module):
    """Pre-LN block: gated [pos-attn ; neg-attn ; self] fusion + FFN residual."""

    def __init__(self, dim: int, n_heads: int = 4, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.pos = MotifAttention(dim, n_heads)
        self.neg = MotifAttention(dim, n_heads)
        self.gate = nn.Linear(3 * dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, h, pos_buckets, neg_buckets):
        hn = self.ln1(h)
        fused = self.gate(torch.cat([self.pos(hn, pos_buckets),
                                     self.neg(hn, neg_buckets), hn], dim=-1))
        h = h + fused
        h = h + self.ff(self.ln2(h))
        return h


class SiGformerModel(SignedLinkModule):
    def __init__(self, n_nodes: int, hidden_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.blocks = nn.ModuleList(
            [SiGformerBlock(hidden_dim, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, pos_buckets, neg_buckets) -> torch.Tensor:
        h = self.node_embed.weight
        for block in self.blocks:
            h = block(h, pos_buckets, neg_buckets)
        return self.ln_f(h)


@register("sigformer")
class SiGformerBaseline(SignedLinkBaseline):
    def build_model(self, meta: GraphMeta, hp: HParams):
        return SiGformerModel(meta.n_nodes, hidden_dim=hp.hidden,
                              n_heads=hp.n_heads, n_layers=hp.n_layers)

    def build_context(self, edges, signs, n_nodes, device):
        return build_neighbour_lists(edges, signs, n_nodes)

    def default_hparams(self) -> HParams:
        return HParams(hidden=32, n_layers=2, n_heads=4)
