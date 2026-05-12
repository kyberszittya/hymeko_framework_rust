"""SGT — Signed Graph Transformer baseline.

A minimal, in-protocol drop-in alongside SGCN / SiGAT-attn for the
HSiKAN comparison sweep.  Architectural distinction from SiGAT-attn:

  * **True transformer encoder**: pre-LayerNorm + residual + position-
    wise FFN, instead of the bare per-bucket multi-head attention
    SiGAT-attn uses.
  * **Sign-aware attention bias**: per-edge bias term added to the
    raw QK^T scores BEFORE softmax, indexed by the signed adjacency
    (separate learned scalars for +1 / -1 / unconnected).  This
    replaces SiGAT's hard pos/neg bucketing.
  * **Sparse over 1-hop neighbours**: each node attends to its
    union(N_pos, N_neg) only, so memory scales with sum_v |N(v)|
    rather than |V|^2 — required for Epinions / Slashdot.

Same training recipe and edge classifier as SiGAT-attn so the
delta vs. SGT is purely the encoder architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_signed_neighbours(edges, signs, n_nodes: int):
    """Per-node combined neighbour list with a parallel sign array.

    For each node v we keep one list of (neighbour, sign) for every
    incident edge (both endpoints).  Returns ``(nbrs, signs)`` with
    ``nbrs[v]`` a list of neighbour-vertex IDs and ``signs[v]`` a
    list of {+1, -1} of the same length.  Used to construct a CSR-
    like batch for the transformer's sparse attention.
    """
    nbrs: list[list[int]] = [[] for _ in range(n_nodes)]
    sgns: list[list[int]] = [[] for _ in range(n_nodes)]
    for (u, v), s in zip(edges, signs):
        u, v, s = int(u), int(v), int(s)
        nbrs[u].append(v); sgns[u].append(s)
        nbrs[v].append(u); sgns[v].append(s)
    return nbrs, sgns


class SignedAttention(nn.Module):
    """Multi-head attention with a learned sign-conditional bias.

    Standard scaled-dot-product attention, plus a per-(query, key)
    bias term added to the raw scores.  Bias is one of two learnt
    scalars depending on the sign of the edge between the query and
    each key node.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.W_O = nn.Linear(dim, dim)
        # Per-head scalar bias for + / -.  +1 → bias_pos, -1 → bias_neg.
        self.bias_pos = nn.Parameter(torch.zeros(n_heads))
        self.bias_neg = nn.Parameter(torch.zeros(n_heads))

    def forward(self, h: torch.Tensor, nbrs: list[list[int]],
                  sgns: list[list[int]]) -> torch.Tensor:
        n = h.shape[0]
        device = h.device
        Q = self.W_Q(h).view(n, self.n_heads, self.head_dim)
        K = self.W_K(h).view(n, self.n_heads, self.head_dim)
        V = self.W_V(h).view(n, self.n_heads, self.head_dim)

        out = torch.zeros_like(Q)
        scale = 1.0 / (self.head_dim ** 0.5)

        # Group nodes by their neighbour-list length so each group can
        # be processed as a single dense (B, L, ...) batch.  Tested up
        # to ~150 unique lengths on Slashdot — well under the 1k-len
        # limit beyond which Python overhead bites.
        from collections import defaultdict
        by_len: dict[int, list[int]] = defaultdict(list)
        for i, lst in enumerate(nbrs):
            if lst:
                by_len[len(lst)].append(i)

        for L, nodes_list in by_len.items():
            nodes = torch.tensor(nodes_list, dtype=torch.long, device=device)
            nbr_idx = torch.tensor(
                [nbrs[i] for i in nodes_list],
                dtype=torch.long, device=device,
            )                                            # (B, L)
            nbr_sgn = torch.tensor(
                [sgns[i] for i in nodes_list],
                dtype=torch.float32, device=device,
            )                                            # (B, L)
            Qb = Q[nodes]                                # (B, H, D)
            Kb = K[nbr_idx]                              # (B, L, H, D)
            Vb = V[nbr_idx]                              # (B, L, H, D)
            scores = torch.einsum('bhd,blhd->bhl', Qb, Kb) * scale
            # Sign-conditional bias: nbr_sgn ∈ {+1, -1}.
            pos_mask = (nbr_sgn > 0).float().unsqueeze(1)      # (B,1,L)
            neg_mask = (nbr_sgn < 0).float().unsqueeze(1)
            bias = (pos_mask * self.bias_pos.view(1, -1, 1)
                    + neg_mask * self.bias_neg.view(1, -1, 1))
            scores = scores + bias
            attn = F.softmax(scores, dim=-1)             # (B, H, L)
            agg = torch.einsum('bhl,blhd->bhd', attn, Vb)
            out[nodes] = agg

        return self.W_O(out.view(n, self.dim))


class SGTBlock(nn.Module):
    """Transformer encoder block: pre-LN attention + pre-LN FFN."""

    def __init__(self, dim: int, n_heads: int = 4, ff_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SignedAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, h, nbrs, sgns):
        h = h + self.attn(self.ln1(h), nbrs, sgns)
        h = h + self.ff(self.ln2(h))
        return h


class SGT(nn.Module):
    """Signed Graph Transformer encoder + edge classifier."""

    def __init__(self, n_nodes: int, hidden_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.blocks = nn.ModuleList([
            SGTBlock(hidden_dim, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, nbrs, sgns) -> torch.Tensor:
        h = self.node_embed.weight
        for block in self.blocks:
            h = block(h, nbrs, sgns)
        return self.ln_f(h)

    def edge_logits(self, z: torch.Tensor,
                    edges_t: torch.Tensor) -> torch.Tensor:
        z_u = z[edges_t[:, 0]]
        z_v = z[edges_t[:, 1]]
        return self.classifier(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_signed_neighbours(edges, signs, n_nodes: int):
    """Public re-export so the run_final_cell-style runners can build
    the per-node neighbour buckets without importing the private
    underscore-prefixed helper."""
    return _build_signed_neighbours(edges, signs, n_nodes)
