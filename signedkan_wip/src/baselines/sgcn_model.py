"""SGCN baseline (Derr et al. 2018, "Signed Graph Convolutional Network").

This is *our re-implementation*, not the reference repo. The
reference is at https://github.com/benedekrozemberczki/SGCN; per
`PLAN_SGCN_GAP.md` the original plan was to clone+adapt, but that
path is dependency-heavy. Implementing in-pipeline gives us
honest, same-split numbers we can place next to HSiKAN.

Architecture (from Derr 2018 §3.2):

    Layer 1:
        h^B_v = σ( W^B [ Σ_{u∈N_pos(v)} x_u  ;  x_v ] )
        h^U_v = σ( W^U [ Σ_{u∈N_neg(v)} x_u  ;  x_v ] )

    Layer ℓ > 1:
        h^B_v = σ( W^B [ Σ_{u∈N_pos(v)} h^B_u ;
                          Σ_{u∈N_neg(v)} h^U_u ;
                          h^B_v ] )
        h^U_v = σ( W^U [ Σ_{u∈N_pos(v)} h^U_u ;
                          Σ_{u∈N_neg(v)} h^B_u ;
                          h^U_v ] )

    Final:  z_v = [ h^B_v ;  h^U_v ]   ∈ R^{2d}

For link-sign prediction we use the standard MLP-on-pair head
    P(sign(u,v) = +1) = σ( w · MLP([z_u ; z_v]) )

Trained with BCE (matched to our other recipes; we omit Derr's
extended structural-balance loss for protocol comparability —
all our other recipes use plain BCE).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_signed_adj(edges, signs, n_nodes: int, device: torch.device):
    """Two row-normalised sparse adjacency matrices: A_pos (positive
    edges) and A_neg (negative edges), each (n_nodes, n_nodes).
    Symmetrised — undirected aggregation, matching Derr 2018."""
    pos_mask = signs == 1
    neg_mask = signs == -1
    e_pos = edges[pos_mask]
    e_neg = edges[neg_mask]

    def _make(e_arr):
        if len(e_arr) == 0:
            idx = torch.zeros((2, 0), dtype=torch.long, device=device)
            val = torch.zeros((0,), device=device)
            return torch.sparse_coo_tensor(idx, val,
                                            (n_nodes, n_nodes)).coalesce()
        src = torch.tensor(e_arr[:, 0], dtype=torch.long, device=device)
        dst = torch.tensor(e_arr[:, 1], dtype=torch.long, device=device)
        # Symmetric: both (u→v) and (v→u).
        rows = torch.cat([src, dst])
        cols = torch.cat([dst, src])
        idx  = torch.stack([rows, cols], dim=0)
        val  = torch.ones(rows.shape[0], device=device)
        A = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()
        # Row-normalise so aggregation is a mean (Derr uses sum; we
        # mean for numerical stability with imbalanced degrees, then
        # let W learn the scale — equivalent up to W scaling).
        deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)
        # Build diag(1/deg) then mm — but easier: scale values by
        # 1/deg[row] in a sparse-aware way.
        A_idx = A.indices()
        A_val = A.values() / deg[A_idx[0]]
        return torch.sparse_coo_tensor(A_idx, A_val,
                                        (n_nodes, n_nodes)).coalesce()
    return _make(e_pos), _make(e_neg)


class SGCNLayer(nn.Module):
    """One Derr-style layer with B/U paths."""

    def __init__(self, in_dim: int, out_dim: int, first: bool):
        super().__init__()
        self.first = first
        if first:
            # B input: [pos-aggregated x  ;  self x]   = 2 * in_dim
            # U input: [neg-aggregated x  ;  self x]   = 2 * in_dim
            self.lin_B = nn.Linear(2 * in_dim, out_dim)
            self.lin_U = nn.Linear(2 * in_dim, out_dim)
        else:
            # B input: [pos-agg h_B ; neg-agg h_U ; self h_B] = 3 * in_dim
            # U input: [pos-agg h_U ; neg-agg h_B ; self h_U] = 3 * in_dim
            self.lin_B = nn.Linear(3 * in_dim, out_dim)
            self.lin_U = nn.Linear(3 * in_dim, out_dim)

    def forward(self, h_B, h_U, A_pos, A_neg):
        if self.first:
            # First layer: x is shared; pos goes to B, neg goes to U.
            x = h_B   # at first layer h_B == h_U == x by convention
            agg_pos = torch.sparse.mm(A_pos, x)
            agg_neg = torch.sparse.mm(A_neg, x)
            new_B = F.relu(self.lin_B(torch.cat([agg_pos, x], dim=-1)))
            new_U = F.relu(self.lin_U(torch.cat([agg_neg, x], dim=-1)))
        else:
            agg_pos_B = torch.sparse.mm(A_pos, h_B)
            agg_neg_U = torch.sparse.mm(A_neg, h_U)
            agg_pos_U = torch.sparse.mm(A_pos, h_U)
            agg_neg_B = torch.sparse.mm(A_neg, h_B)
            new_B = F.relu(self.lin_B(
                torch.cat([agg_pos_B, agg_neg_U, h_B], dim=-1)
            ))
            new_U = F.relu(self.lin_U(
                torch.cat([agg_pos_U, agg_neg_B, h_U], dim=-1)
            ))
        return new_B, new_U


class SGCN(nn.Module):
    """SGCN with two B/U layers + an MLP edge-sign classifier on
    [z_u ; z_v] where z = [h_B ; h_U]."""

    def __init__(self, n_nodes: int, hidden_dim: int = 32,
                 n_layers: int = 2):
        super().__init__()
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.node_embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.node_embed.weight, std=0.1)
        self.layers = nn.ModuleList()
        for li in range(n_layers):
            self.layers.append(SGCNLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                first=(li == 0),
            ))
        # z_u (2d) + z_v (2d) → 4d input
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, A_pos, A_neg) -> torch.Tensor:
        """Return per-node embedding z = [h_B ; h_U], shape (n_nodes, 2d)."""
        x = self.node_embed.weight
        h_B, h_U = x, x
        for layer in self.layers:
            h_B, h_U = layer(h_B, h_U, A_pos, A_neg)
        return torch.cat([h_B, h_U], dim=-1)

    def edge_logits(self, z, edges_t) -> torch.Tensor:
        """edges_t: (n_edges, 2) long. Returns (n_edges,) logits."""
        z_u = z[edges_t[:, 0]]
        z_v = z[edges_t[:, 1]]
        return self.classifier(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_signed_adj(edges, signs, n_nodes: int, device: torch.device):
    """Public wrapper for the data adapter."""
    return _build_signed_adj(edges, signs, n_nodes, device)


def extended_balance_loss(z_B: torch.Tensor, z_U: torch.Tensor,
                           edges_t: torch.Tensor, signs_t: torch.Tensor,
                           margin: float = 1.0) -> torch.Tensor:
    """Derr 2018 §3.3 extended structural balance loss.

    For each training edge (u, v) with sign s ∈ {+1, −1}:
      - if s = +1 (positive edge / "friend"):
          pull B(u), B(v) together   →   ‖B(u) − B(v)‖²
      - if s = −1 (negative edge / "enemy"):
          push B(u), B(v) apart       →   relu(margin − ‖B(u) − B(v)‖²)

    Returns the mean over training edges.

    z_B, z_U : (n_nodes, d) — first/second halves of SGCN's node embedding.
    edges_t  : (n_edges, 2) long
    signs_t  : (n_edges,)   ±1 float
    """
    u, v = edges_t[:, 0], edges_t[:, 1]
    diff_B = (z_B[u] - z_B[v]).pow(2).sum(dim=-1)              # (n_edges,)
    pos_mask = (signs_t == 1.0)
    neg_mask = ~pos_mask
    pos_term = diff_B[pos_mask].mean() if pos_mask.any() else \
                torch.zeros((), device=z_B.device)
    neg_term = (torch.relu(margin - diff_B[neg_mask]).mean()
                if neg_mask.any() else
                torch.zeros((), device=z_B.device))
    return pos_term + neg_term
