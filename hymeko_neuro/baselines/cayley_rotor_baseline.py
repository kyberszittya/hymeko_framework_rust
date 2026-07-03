"""Cayley-rotor signed-link baseline — the inductive, leakage-free node feature.

This is DADSGNN's signed message-passing (reused: ``SGCNLayer`` + ``build_signed_adj``)
with its **transductive** ``nn.Embedding(n_nodes, d)`` table replaced by an
**inductive** Cayley-rotor embedding computed from train-only structural node
features. The embedding axis is the only difference, so a head-to-head against
``dadsgnn`` isolates exactly what the rotor buys:

  * inductive: no per-node identity table, so it transfers and does not memorise;
  * leakage-free: the node-ID table is the memorisation channel of the leakage
    audit ([[project-nature-leakage-paper]]); structural features remove it;
  * parameter-light: the table (``n_nodes * hidden`` params) is dropped for a
    tiny shared linear (``struct_dim * d``), item-count-free.

Strict invariant preserved: ``build_context`` receives train-only edges/signs,
and the structural features are computed from those alone.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hymeko_neuro.graph.embeddings.cayley_rotor import CayleyRotorEmbedding

from .registry import (
    GraphMeta,
    HParams,
    SignedLinkBaseline,
    SignedLinkModule,
    register,
)
from .sgcn_model import SGCNLayer, build_signed_adj
# Inductive per-node features live in the StructuralFeature registry (single
# source of truth, §6.5 #1). Re-exported here for existing importers
# (run_hsikan_rotor, run_rotor_head_ablation, tests) that import these names.
from .structural_features import (
    CAP_PER_K,
    CYC_DIM,
    CYC_KS,
    STRUCT_DIM,
    _cycle_features,
    _structural_features,
    build_node_features,
    feature_dim,
)

__all__ = [
    "STRUCT_DIM", "CYC_DIM", "CYC_KS", "CAP_PER_K",
    "_structural_features", "_cycle_features",
    "build_node_features", "feature_dim",
]


class CayleyRotorSignedModel(SignedLinkModule):
    """Inductive rotor features → signed message passing → per-node embedding."""

    def __init__(self, in_features: int, hidden: int = 32, n_layers: int = 2):
        super().__init__()
        n_blocks = max(1, hidden // 3)
        self.emb = CayleyRotorEmbedding(n_blocks=n_blocks, in_features=in_features)
        self.proj = nn.Linear(self.emb.embedding_dim, hidden)
        self.layers = nn.ModuleList([
            SGCNLayer(in_dim=hidden, out_dim=hidden, first=(i == 0))
            for i in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode_nodes(self, feats, A_pos, A_neg) -> torch.Tensor:
        x = self.proj(self.emb(feats))          # inductive rotor features (n, hidden)
        h_B, h_U = x, x
        for layer in self.layers:
            h_B, h_U = layer(h_B, h_U, A_pos, A_neg)
        return torch.cat([h_B, h_U], dim=-1)    # (n, 2*hidden)


@register("cayley_rotor")
class CayleyRotorBaseline(SignedLinkBaseline):
    def build_model(self, meta: GraphMeta, hp: HParams):
        return CayleyRotorSignedModel(
            in_features=STRUCT_DIM, hidden=hp.hidden, n_layers=hp.n_layers,
        )

    def build_context(self, edges, signs, n_nodes, device):
        feats = _structural_features(edges, signs, n_nodes)
        feats_t = torch.from_numpy(feats).to(device)
        a_pos, a_neg = build_signed_adj(edges, signs, n_nodes, device)
        return (feats_t, a_pos, a_neg)

    def default_hparams(self) -> HParams:
        return HParams(hidden=32, n_layers=2)


# Signed k-cycle participation (`_cycle_features`, `CYC_KS`, `CAP_PER_K`,
# `CYC_DIM`) now lives in `structural_features` and is imported above.


class CayleyRotorJumpModel(SignedLinkModule):
    """Rotor (degree features) → signed message passing, with the signed-cycle
    features **jumped past the SGCN** directly to the readout. The decomposition
    (reports/cyc_decompose) showed cycles-as-input are absorbed/redundant under
    message passing; this routes them around it so the readout sees the cycle
    signal undiluted (the highway-jump idea applied to cycles)."""

    def __init__(self, in_features: int, jump_features: int,
                 hidden: int = 32, n_layers: int = 2):
        super().__init__()
        n_blocks = max(1, hidden // 3)
        self.emb = CayleyRotorEmbedding(n_blocks=n_blocks, in_features=in_features)
        self.proj = nn.Linear(self.emb.embedding_dim, hidden)
        self.layers = nn.ModuleList([
            SGCNLayer(in_dim=hidden, out_dim=hidden, first=(i == 0))
            for i in range(n_layers)
        ])
        self.jump = nn.Linear(jump_features, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(6 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def encode_nodes(self, deg, cyc, A_pos, A_neg) -> torch.Tensor:
        x = self.proj(self.emb(deg))
        h_B, h_U = x, x
        for layer in self.layers:
            h_B, h_U = layer(h_B, h_U, A_pos, A_neg)
        j = self.jump(cyc)                       # cycles jumped past the SGCN
        return torch.cat([h_B, h_U, j], dim=-1)  # (n, 3*hidden)


@register("cayley_rotor_jump")
class CayleyRotorJumpBaseline(CayleyRotorBaseline):
    """Cycles jumped to the readout (vs ``cayley_rotor_cyc`` which feeds them in
    at the bottom). Isolates whether jump-propagation un-redundants the cycle
    signal that message-passing otherwise absorbs."""

    def build_model(self, meta: GraphMeta, hp: HParams):
        return CayleyRotorJumpModel(
            in_features=STRUCT_DIM, jump_features=3 * len(CYC_KS),
            hidden=hp.hidden, n_layers=hp.n_layers,
        )

    def build_context(self, edges, signs, n_nodes, device):
        deg = _structural_features(edges, signs, n_nodes)
        cyc = _cycle_features(edges, signs, n_nodes)
        a_pos, a_neg = build_signed_adj(edges, signs, n_nodes, device)
        return (torch.from_numpy(deg).to(device),
                torch.from_numpy(cyc).to(device), a_pos, a_neg)


class SiGATRotorModel(SignedLinkModule):
    """SiGAT's motif attention (reused) with its transductive ``nn.Embedding``
    table replaced by the inductive Cayley-rotor on structural features. Tests
    H3: is the SiGAT advantage the *attention*, independent of the embedding? If
    this matches plain SiGAT, the rotor matches the best model at ~270x fewer
    params (the embedding was never the gap)."""

    def __init__(self, in_features: int, hidden: int = 32,
                 n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        from .sigat_model import MotifAttention
        n_blocks = max(1, hidden // 3)
        self.emb = CayleyRotorEmbedding(n_blocks=n_blocks, in_features=in_features)
        self.proj = nn.Linear(self.emb.embedding_dim, hidden)
        self.layers = nn.ModuleList(nn.ModuleDict({
            "pos": MotifAttention(hidden, n_heads),
            "neg": MotifAttention(hidden, n_heads),
            "mix": nn.Linear(3 * hidden, hidden),
        }) for _ in range(n_layers))
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def encode_nodes(self, struct, pos_buckets, neg_buckets) -> torch.Tensor:
        h = self.proj(self.emb(struct))           # inductive rotor, not a table
        for layer in self.layers:
            h_pos = layer["pos"](h, pos_buckets)
            h_neg = layer["neg"](h, neg_buckets)
            h = F.relu(layer["mix"](torch.cat([h_pos, h_neg, h], dim=-1)))
        return h


@register("sigat_rotor")
class SiGATRotorBaseline(SignedLinkBaseline):
    def build_model(self, meta: GraphMeta, hp: HParams):
        return SiGATRotorModel(in_features=STRUCT_DIM, hidden=hp.hidden,
                               n_heads=hp.n_heads, n_layers=hp.n_layers)

    def build_context(self, edges, signs, n_nodes, device):
        from .sigat_model import build_neighbour_lists
        struct = _structural_features(edges, signs, n_nodes)
        pos, neg = build_neighbour_lists(edges, signs, n_nodes)
        return (torch.from_numpy(struct).to(device), pos, neg)

    def default_hparams(self) -> HParams:
        return HParams(hidden=32, n_layers=1, n_heads=4)  # SiGAT's recipe


@register("cayley_rotor_jump_k4")
class CayleyRotorJumpK4Baseline(CayleyRotorJumpBaseline):
    """Jump-propagated cycles at arities k=3 AND k=4 (richer cycle signal than
    triangles alone), now that jump-routing recovers the cycle signal. 4-cycles
    explode faster (see n_tuples cap note) — enumerated under the same cap."""

    K4_KS = (3, 4)

    def build_model(self, meta: GraphMeta, hp: HParams):
        return CayleyRotorJumpModel(
            in_features=STRUCT_DIM, jump_features=3 * len(self.K4_KS),
            hidden=hp.hidden, n_layers=hp.n_layers,
        )

    def build_context(self, edges, signs, n_nodes, device):
        deg = _structural_features(edges, signs, n_nodes)
        cyc = _cycle_features(edges, signs, n_nodes, ks=self.K4_KS)
        a_pos, a_neg = build_signed_adj(edges, signs, n_nodes, device)
        return (torch.from_numpy(deg).to(device),
                torch.from_numpy(cyc).to(device), a_pos, a_neg)


@register("cayley_rotor_cyc")
class CayleyRotorCycBaseline(CayleyRotorBaseline):
    """Rotor embedding on degree **+ signed-k-cycle** features --- the convergence
    A/B: does the (beyond-WL) cycle signal lift the rotor toward the attention
    baseline while staying parameter-light?"""

    def build_model(self, meta: GraphMeta, hp: HParams):
        return CayleyRotorSignedModel(
            in_features=CYC_DIM, hidden=hp.hidden, n_layers=hp.n_layers,
        )

    def build_context(self, edges, signs, n_nodes, device):
        deg = _structural_features(edges, signs, n_nodes)
        cyc = _cycle_features(edges, signs, n_nodes)
        feats_t = torch.from_numpy(np.concatenate([deg, cyc], axis=1)).to(device)
        a_pos, a_neg = build_signed_adj(edges, signs, n_nodes, device)
        return (feats_t, a_pos, a_neg)


# --- registry-driven input-enrichment variants (2026-06-18) ------------------
# One spec-parameterised base (§6.5 #1: no per-variant build_context copy); each
# registered subclass differs only by its leakage-free FEATURE_SPEC. The rotor
# input dim is *derived* from the spec, so adding a feature set is one line.

class _RotorFeatureSpecBaseline(CayleyRotorBaseline):
    """``cayley_rotor`` with an arbitrary :mod:`structural_features` spec."""

    FEATURE_SPEC: tuple[str, ...] = ("degree",)

    def build_model(self, meta: GraphMeta, hp: HParams):
        return CayleyRotorSignedModel(
            in_features=feature_dim(self.FEATURE_SPEC),
            hidden=hp.hidden, n_layers=hp.n_layers,
        )

    def build_context(self, edges, signs, n_nodes, device):
        feats = build_node_features(self.FEATURE_SPEC, edges, signs, n_nodes)
        a_pos, a_neg = build_signed_adj(edges, signs, n_nodes, device)
        return (torch.from_numpy(feats).to(device), a_pos, a_neg)


@register("cayley_rotor_walk")
class CayleyRotorWalkBaseline(_RotorFeatureSpecBaseline):
    """degree + exact signed ``A^k`` walk profile (the new, cap-free feature)."""
    FEATURE_SPEC = ("degree", "walk_k3")


@register("cayley_rotor_cyc_walk")
class CayleyRotorCycWalkBaseline(_RotorFeatureSpecBaseline):
    """degree + cycle participation + walk profile (cap'd motif ⊕ exact reach)."""
    FEATURE_SPEC = ("degree", "cycle_k3", "walk_k3")


@register("cayley_rotor_full")
class CayleyRotorFullBaseline(_RotorFeatureSpecBaseline):
    """The full enriched input: degree + cycle + walk + clustering ratios."""
    FEATURE_SPEC = ("degree", "cycle_k3", "walk_k3", "ratios")


# --- is the Cayley-rotor geometry load-bearing? (the MLP-embed ablation) ------
# Control: identical architecture (proj + SGCN + classifier), but the S³ rotor
# embedding is replaced by a *higher-capacity* plain MLP of the same output dim.
# If the rotor does not beat this generic embedding, its geometry is decorative and
# the line is really "structural features + SGCN" (head ablation already showed the
# rotor *algebra* is not load-bearing at the readout — this tests the embedding).

class _MLPEmbed(nn.Module):
    """Plain nonlinear embedding matching ``CayleyRotorEmbedding``'s output dim.

    2-layer Tanh MLP (Tanh bounds the output like the rotor's S³ sphere). Given
    ``embedding_dim`` hidden width it carries *more* params than the rotor's single
    projection — a deliberately generous control (the rotor must beat extra capacity
    to count as load-bearing). ``embedding_dim`` attr mirrors the rotor's interface.
    """

    def __init__(self, in_features: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim), nn.Tanh(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class MLPEmbedSignedModel(CayleyRotorSignedModel):
    """``CayleyRotorSignedModel`` with the rotor embedding swapped for ``_MLPEmbed``.

    Same ``embedding_dim`` (``3 * n_blocks``), same proj/SGCN/classifier, so the
    *only* difference is the embedding map — the fair "is the rotor geometry
    load-bearing" control. ``encode_nodes`` is inherited unchanged.
    """

    def __init__(self, in_features: int, hidden: int = 32, n_layers: int = 2):
        nn.Module.__init__(self)  # bypass the rotor-building parent body, rebuild
        embedding_dim = 3 * max(1, hidden // 3)  # == CayleyRotorEmbedding(n_refs=1)
        self.emb = _MLPEmbed(in_features, embedding_dim)
        self.proj = nn.Linear(embedding_dim, hidden)
        self.layers = nn.ModuleList([
            SGCNLayer(in_dim=hidden, out_dim=hidden, first=(i == 0))
            for i in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )


class _MLPFeatureSpecBaseline(_RotorFeatureSpecBaseline):
    """MLP-embed control sharing the spec-driven feature pipeline."""

    def build_model(self, meta: GraphMeta, hp: HParams):
        return MLPEmbedSignedModel(
            in_features=feature_dim(self.FEATURE_SPEC),
            hidden=hp.hidden, n_layers=hp.n_layers,
        )


@register("mlp_embed")
class MLPEmbedBaseline(_MLPFeatureSpecBaseline):
    """Degree features, MLP embedding — control for ``cayley_rotor``."""
    FEATURE_SPEC = ("degree",)


@register("mlp_embed_walk")
class MLPEmbedWalkBaseline(_MLPFeatureSpecBaseline):
    """Degree + walk profile, MLP embedding — control for ``cayley_rotor_walk``."""
    FEATURE_SPEC = ("degree", "walk_k3")
