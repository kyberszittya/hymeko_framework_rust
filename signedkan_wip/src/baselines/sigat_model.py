"""SiGAT-style attention baseline (Huang et al. 2019).

This is an *in-protocol* re-implementation, not a faithful reproduction
of the 38-motif directed reference implementation. We use the same
two-motif decomposition as our SGCN re-impl (positive vs negative
neighbour sets) but replace mean-aggregation with multi-head attention,
which is the SiGAT-vs-SGCN architectural distinction.

For full reference numbers see Huang et al. 2019; this baseline is for
strict-Derr-protocol comparison against our HSiKAN/SGCN recipes
(plain BCE, no early stop, no class weighting, no weight decay).

Architecture:
    For each node v:
        h_pos(v) = MultiHeadAttn(h_v, {h_u : u ∈ N_pos(v)})
        h_neg(v) = MultiHeadAttn(h_v, {h_u : u ∈ N_neg(v)})
        z(v)     = [h_pos(v) ; h_neg(v) ; h_v]
    Edge classifier MLP on [z_u ; z_v].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_neighbour_lists(edges, signs, n_nodes: int):
    """Return per-node lists of (neighbour, sign-bucket) pairs.

    Returns (pos_idx, pos_offsets), (neg_idx, neg_offsets) — CSR-style
    flat arrays for batched attention.
    """
    pos_buckets: list[list[int]] = [[] for _ in range(n_nodes)]
    neg_buckets: list[list[int]] = [[] for _ in range(n_nodes)]
    for (u, v), s in zip(edges, signs):
        if s == 1:
            pos_buckets[int(u)].append(int(v))
            pos_buckets[int(v)].append(int(u))
        else:
            neg_buckets[int(u)].append(int(v))
            neg_buckets[int(v)].append(int(u))
    return pos_buckets, neg_buckets


class MotifAttention(nn.Module):
    """Multi-head attention from a node to its motif-typed neighbours.

    Q comes from the centre node; K, V come from the neighbour set.
    If a node has no neighbours of this motif type, output is zeros.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_O = nn.Linear(dim, dim, bias=False)

    def forward(self, h: torch.Tensor, neighbour_lists: list[list[int]]) -> torch.Tensor:
        """h: (n_nodes, dim). Returns (n_nodes, dim).

        Per-node attention over its motif-typed neighbour set.
        """
        n_nodes = h.shape[0]
        Q = self.W_Q(h).view(n_nodes, self.n_heads, self.head_dim)
        K_all = self.W_K(h).view(n_nodes, self.n_heads, self.head_dim)
        V_all = self.W_V(h).view(n_nodes, self.n_heads, self.head_dim)

        out = torch.zeros_like(h.view(n_nodes, self.n_heads, self.head_dim))

        # Group nodes by neighbour-list length to batch-process buckets
        # of similar sizes. For small graphs the per-node Python loop is
        # fine; for Bitcoin/Slashdot we group.
        from collections import defaultdict
        by_len: dict[int, list[int]] = defaultdict(list)
        for i, lst in enumerate(neighbour_lists):
            if lst:
                by_len[len(lst)].append(i)

        scale = 1.0 / (self.head_dim ** 0.5)
        device = h.device
        for L, nodes in by_len.items():
            nodes_t = torch.tensor(nodes, dtype=torch.long, device=device)
            # neighbour matrix (B, L)
            nbr_idx = torch.tensor(
                [neighbour_lists[i] for i in nodes],
                dtype=torch.long, device=device,
            )
            # Q: (B, H, D), K/V: (B, L, H, D)
            Qb = Q[nodes_t]                     # (B, H, D)
            Kb = K_all[nbr_idx]                 # (B, L, H, D)
            Vb = V_all[nbr_idx]                 # (B, L, H, D)
            # scores: (B, H, L)
            scores = torch.einsum('bhd,blhd->bhl', Qb, Kb) * scale
            attn = F.softmax(scores, dim=-1)
            # weighted: (B, H, D)
            agg = torch.einsum('bhl,blhd->bhd', attn, Vb)
            out[nodes_t] = agg

        return self.W_O(out.view(n_nodes, self.dim))


class SiGATAttn(nn.Module):
    """SiGAT-style attention baseline with pos/neg motif decomposition."""

    def __init__(self, n_nodes: int, hidden_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'pos': MotifAttention(hidden_dim, n_heads),
                'neg': MotifAttention(hidden_dim, n_heads),
                # combine [pos ; neg ; self] → hidden_dim
                'mix': nn.Linear(3 * hidden_dim, hidden_dim),
            }))
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, pos_buckets, neg_buckets) -> torch.Tensor:
        h = self.node_embed.weight
        for layer in self.layers:
            h_pos = layer['pos'](h, pos_buckets)
            h_neg = layer['neg'](h, neg_buckets)
            h = F.relu(layer['mix'](torch.cat([h_pos, h_neg, h], dim=-1)))
        return h

    def edge_logits(self, z: torch.Tensor, edges_t: torch.Tensor) -> torch.Tensor:
        z_u = z[edges_t[:, 0]]
        z_v = z[edges_t[:, 1]]
        return self.classifier(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_neighbour_lists(edges, signs, n_nodes: int):
    return _build_neighbour_lists(edges, signs, n_nodes)
