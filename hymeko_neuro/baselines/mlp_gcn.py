"""Simple sanity baselines: MLP edge classifier and sign-blind GCN.

Both share the same structure:
    node_embed (V, d) → [optional graph aggregation] → MLP head on
    [z_u; z_v] for each edge (u, v) → BCE on sign.

`MLPEdge` does no graph propagation — pure edge-endpoint MLP. Tests
"is the graph structure even helping?".

`SignBlindGCN` does k-layer GCN-style mean aggregation on the
unsigned adjacency (signs stripped). Tests "does sign-aware
aggregation matter vs sign-blind GCN?".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_unsigned_adj(edges, n_nodes: int, device: torch.device):
    """Symmetric, row-normalised unsigned adjacency. Ignores signs."""
    if len(edges) == 0:
        idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        val = torch.zeros((0,), device=device)
        return torch.sparse_coo_tensor(idx, val,
                                        (n_nodes, n_nodes)).coalesce()
    src = torch.tensor(edges[:, 0], dtype=torch.long, device=device)
    dst = torch.tensor(edges[:, 1], dtype=torch.long, device=device)
    rows = torch.cat([src, dst])
    cols = torch.cat([dst, src])
    idx  = torch.stack([rows, cols], dim=0)
    val  = torch.ones(rows.shape[0], device=device)
    A = torch.sparse_coo_tensor(idx, val, (n_nodes, n_nodes)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp_min(1.0)
    A_idx = A.indices()
    A_val = A.values() / deg[A_idx[0]]
    return torch.sparse_coo_tensor(A_idx, A_val,
                                    (n_nodes, n_nodes)).coalesce()


class MLPEdge(nn.Module):
    """No graph: just per-node embedding + MLP on (z_u, z_v)."""

    def __init__(self, n_nodes: int, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edges_t: torch.Tensor) -> torch.Tensor:
        z = self.embed.weight
        z_u = z[edges_t[:, 0]]
        z_v = z[edges_t[:, 1]]
        return self.head(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SignBlindGCN(nn.Module):
    """k-layer GCN that ignores signs. Mean-aggregates on the
    unsigned adjacency."""

    def __init__(self, n_nodes: int, hidden_dim: int = 32,
                 n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, hidden_dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, A) -> torch.Tensor:
        h = self.embed.weight
        for lin in self.layers:
            h = F.relu(lin(torch.sparse.mm(A, h)))
        return h

    def forward(self, A, edges_t: torch.Tensor) -> torch.Tensor:
        z = self.encode_nodes(A)
        z_u = z[edges_t[:, 0]]
        z_v = z[edges_t[:, 1]]
        return self.head(torch.cat([z_u, z_v], dim=-1)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
