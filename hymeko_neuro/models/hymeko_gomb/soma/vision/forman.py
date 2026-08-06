"""Forman-Ricci curvature on signed hypergraphs (Ricci-Stim phase 1).

Forman's combinatorial Ricci curvature (Forman 2003) on an undirected
graph with vertex set $V$ and edge set $E$:

    κ(e) = 2 - deg(u) - deg(v) + 2 |Δ(u, v)|

where $e = (u, v)$ and $|Δ(u, v)|$ is the number of triangles incident
on $e$. Positive κ marks dense / clique-like neighbourhoods;
negative κ marks bottleneck edges. The formula is unweighted — the
signed-weight extension belongs to a later phase if needed.

Vertex curvature κ_v is computed as the mean of κ over incident edges
(isolated vertices get κ_v = 0).

The module is stateless: no learnable parameters. It computes a
deterministic graph-theoretic invariant. The `nn.Module` wrapper
exists so the curvature head can be dropped into a larger PyTorch
graph and so future learnable variants can subclass it cleanly.

Reference: R. Forman, "Bochner's Method for Cell Complexes and
Combinatorial Ricci Curvature," Discrete & Computational Geometry,
2003.

Used by: AdaptiveQuadtree (phase 2) for subdivision decisions;
Bochner-coupled HypergraphConv (phase 4) as the curvature weighting;
SDRF rewiring (phase 6) for bottleneck identification.

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FormanCurvature:
    """Output of FormanCurvatureHead.

    Attributes
    ----------
    edge_kappa : Tensor[n_edges]
        Forman κ per edge. Higher κ ⇒ denser local neighbourhood.
    vertex_kappa : Tensor[n_nodes]
        Mean κ over incident edges; 0 for isolated vertices.
    degree : Tensor[n_nodes]
        Vertex degrees (used as a sanity by-product).
    triangle_count : Tensor[n_edges]
        Number of triangles incident on each edge.
    """

    edge_kappa: torch.Tensor
    vertex_kappa: torch.Tensor
    degree: torch.Tensor
    triangle_count: torch.Tensor


class FormanCurvatureHead(nn.Module):
    """Stateless computation of Forman-Ricci curvature on a graph.

    Parameters
    ----------
    treat_as_undirected : bool, default True
        If True, the edge list is treated as an undirected edge set
        — duplicate (u, v) and (v, u) entries are merged when
        computing degree and triangle counts. If False, the edge
        list is taken as-is.

    Preconditions
    -------------
    * Edges are 0-indexed; vertex indices ∈ [0, n_nodes).
    * Self-loops (u == v) are silently dropped.

    Postconditions
    --------------
    * Output ``edge_kappa`` has shape (n_edges,), one κ per input edge row.
    * Output ``vertex_kappa`` has shape (n_nodes,).
    * Reflexive / signed identity: on a triangle K₃, all κ = 0. On a
      cycle Cₙ (n ≥ 4), all κ = -2. On a complete graph Kₙ, all κ = 0.
    """

    def __init__(self, treat_as_undirected: bool = True) -> None:
        super().__init__()
        self.treat_as_undirected = treat_as_undirected

    def forward(
        self,
        edges: torch.Tensor,
        n_nodes: int,
    ) -> FormanCurvature:
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(
                f"edges must have shape (n_edges, 2), got "
                f"{tuple(edges.shape)}"
            )
        device = edges.device
        if edges.shape[0] == 0:
            return FormanCurvature(
                edge_kappa=torch.zeros(0, device=device),
                vertex_kappa=torch.zeros(n_nodes, device=device),
                degree=torch.zeros(n_nodes, dtype=torch.long, device=device),
                triangle_count=torch.zeros(0, device=device),
            )

        # Drop self-loops.
        keep = edges[:, 0] != edges[:, 1]
        e_clean = edges[keep]
        original_indices = torch.where(keep)[0]

        # Canonicalise edges (u < v) so the dense-adjacency build is
        # symmetric. The output κ array still corresponds to the
        # *original* input rows.
        if self.treat_as_undirected:
            u_min = torch.minimum(e_clean[:, 0], e_clean[:, 1])
            u_max = torch.maximum(e_clean[:, 0], e_clean[:, 1])
            canonical = torch.stack([u_min, u_max], dim=1)
        else:
            canonical = e_clean

        # Vectorised dense adjacency: A[u, v] = True iff edge (u, v) exists.
        # Memory: n_nodes² bools (~65 KB at n_nodes=256). Fine for our scale.
        u = canonical[:, 0]
        v = canonical[:, 1]
        adjacency = torch.zeros(
            n_nodes, n_nodes, dtype=torch.bool, device=device,
        )
        adjacency[u, v] = True
        if self.treat_as_undirected:
            adjacency[v, u] = True
        degree = adjacency.to(torch.int64).sum(dim=1)

        # Triangle count per edge: |adj(u) ∩ adj(v)| = elementwise AND, summed.
        # Vectorised: for all canonical edges at once.
        if canonical.shape[0] > 0:
            adj_u = adjacency[u]                          # (n_e, n_nodes)
            adj_v = adjacency[v]                          # (n_e, n_nodes)
            tri_count_clean = (adj_u & adj_v).sum(dim=-1) # (n_e,)
        else:
            tri_count_clean = torch.zeros(
                0, dtype=torch.int64, device=device,
            )
        edge_kappa_clean = (
            2.0
            - degree[u].float()
            - degree[v].float()
            + 2.0 * tri_count_clean.float()
        )

        # Scatter back to original input order.
        tri_per_row = torch.zeros(edges.shape[0], device=device)
        edge_kappa = torch.zeros(edges.shape[0], device=device)
        tri_per_row[original_indices] = tri_count_clean.float()
        edge_kappa[original_indices] = edge_kappa_clean

        # Vertex curvature: scatter-mean of incident-edge κ.
        vertex_kappa = torch.zeros(n_nodes, device=device)
        deg_count = torch.zeros(n_nodes, device=device)
        # Each canonical edge contributes its κ to both endpoints.
        vertex_kappa.index_add_(0, u, edge_kappa_clean)
        vertex_kappa.index_add_(0, v, edge_kappa_clean)
        deg_count.index_add_(0, u, torch.ones_like(edge_kappa_clean))
        deg_count.index_add_(0, v, torch.ones_like(edge_kappa_clean))
        nz = deg_count > 0
        vertex_kappa[nz] = vertex_kappa[nz] / deg_count[nz]

        return FormanCurvature(
            edge_kappa=edge_kappa,
            vertex_kappa=vertex_kappa,
            degree=degree,
            triangle_count=tri_per_row,
        )
