"""Star and clique sparse-tensor representations of weighted hypergraphs.

This module implements the canonical sparse-tensor primitives from
the user's dissertation framing, generalising signed-graph
edge incidence to per-arc-weighted hypergraph incidence. See
``docs/plans/2026-05-17-general-weighted-hyperedges/plan.tex``.

Two complementary forms:

* **Star expansion** $H^\\star$: bipartite graph with vertices
  $V \\sqcup \\mathcal{E}$ and per-arc weights $w(v, e)$. The
  canonical sparse representation; $\\sum_e |e|$ non-zero entries.
* **Clique expansion** $H^\\bigtriangleup$: vertex-by-vertex
  weighted adjacency obtained by materialising the product clique
  $w_{u,v}(e) = w(u, e) \\cdot w(v, e)$ within each hyperedge. We
  do NOT materialise this explicitly; it is computed lazily via
  the star-clique identity:

      A^△ = M^⋆ D^{-1} (M^⋆)^T   (minus the self-loop diagonal)

  where $D = \\diag(|e|)_{e \\in \\mathcal{E}}$.

The classical signed-graph case is the degenerate
$\\mathcal{W} = \\{-1, +1\\}$ specialisation with the per-edge
sign replicated across both incident vertices.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StarTensor:
    """A sparse-tensor representation of the star expansion of a
    weighted hypergraph.

    Attributes
    ----------
    indices : LongTensor, shape (2, nnz)
        ``indices[0]`` = vertex IDs, ``indices[1]`` = hyperedge IDs.
    values : FloatTensor, shape (nnz,) for real-valued $\\mathcal{W}$.
        Per-arc weights $w(v, e)$.
    n_vertices : int
        $|V|$.
    n_hyperedges : int
        $|\\mathcal{E}|$.
    edge_sizes : LongTensor, shape (n_hyperedges,)
        $|e|$ for each hyperedge — used by ``to_clique_adjacency``.
    """

    indices: torch.Tensor
    values: torch.Tensor
    n_vertices: int
    n_hyperedges: int
    edge_sizes: torch.Tensor

    @property
    def nnz(self) -> int:
        return int(self.values.shape[0])

    @property
    def device(self) -> torch.device:
        return self.values.device

    def to_sparse_coo(self) -> torch.Tensor:
        """Materialise as a :class:`torch.sparse_coo_tensor` of shape
        ``(n_vertices, n_hyperedges)``."""
        return torch.sparse_coo_tensor(
            self.indices,
            self.values,
            (self.n_vertices, self.n_hyperedges),
        ).coalesce()

    def to_clique_adjacency(
        self,
        *,
        include_self_loops: bool = False,
    ) -> torch.Tensor:
        """Compute the vertex-by-vertex \\emph{product clique}
        adjacency $A^\\bigtriangleup$ from the star tensor.

        Implementation: $A^\\bigtriangleup = M^\\star D^{-1} (M^\\star)^\\top$
        where $D$ is the diagonal of edge sizes. The result includes
        self-loops with value $\\sum_e w(v, e)^2 / |e|$ on the
        diagonal; pass ``include_self_loops=False`` (default) to zero
        them out for use as a vertex-vertex adjacency.

        Returns
        -------
        torch.Tensor : dense ``(n_vertices, n_vertices)`` adjacency.
            For very-sparse hypergraphs this is wasteful; a sparse
            return path is a v2 lever.
        """
        M = self.to_sparse_coo().to_dense()                 # (V, E)
        inv_sizes = 1.0 / self.edge_sizes.to(M.dtype).clamp(min=1.0)
        # A = M * diag(inv_sizes) * M^T  =>  (M * inv_sizes_per_col) @ M^T
        M_scaled = M * inv_sizes.view(1, -1)
        A = M_scaled @ M.t()                                  # (V, V)
        if not include_self_loops:
            A = A - torch.diag(torch.diag(A))
        return A

    def degrees(self) -> torch.Tensor:
        """Per-vertex degree under the star representation.

        Returns ``(n_vertices,)`` with the count of distinct
        hyperedges each vertex participates in.
        """
        rows = self.indices[0]
        return torch.bincount(rows, minlength=self.n_vertices)


def signed_graph_to_star(
    edges: torch.Tensor,
    signs: torch.Tensor,
    n_vertices: int,
) -> StarTensor:
    """Construct a star tensor from a signed graph.

    The signed-graph special case of the per-arc weight formalism:
    each edge becomes a binary hyperedge, and the sign is replicated
    across both endpoints as ``w(v, e) = sigma(e)``.

    Parameters
    ----------
    edges : LongTensor, shape (n_edges, 2)
    signs : FloatTensor (or IntTensor), shape (n_edges,) with values in {-1, +1}.
    n_vertices : int
    """
    if edges.dim() != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must be (n_edges, 2); got {tuple(edges.shape)}")
    if signs.shape[0] != edges.shape[0]:
        raise ValueError("edges and signs must have matching length")
    n_edges = edges.shape[0]
    # Each edge contributes two (vertex, hyperedge) arcs.
    src = edges[:, 0].long()
    dst = edges[:, 1].long()
    # Stack vertex axis: [src; dst] (2 * n_edges entries)
    rows = torch.cat([src, dst], dim=0)
    cols = torch.cat([torch.arange(n_edges), torch.arange(n_edges)], dim=0)
    cols = cols.to(rows.device)
    # Replicated signs.
    vals = torch.cat([signs.float(), signs.float()], dim=0)
    indices = torch.stack([rows, cols], dim=0)
    edge_sizes = torch.full((n_edges,), 2, dtype=torch.long, device=rows.device)
    return StarTensor(
        indices=indices,
        values=vals,
        n_vertices=int(n_vertices),
        n_hyperedges=int(n_edges),
        edge_sizes=edge_sizes,
    )


def cycle_pool_to_star(
    cycle_vertex_lists: list[torch.Tensor],
    cycle_signs: torch.Tensor,
    n_vertices: int,
) -> StarTensor:
    """Construct a star tensor from a cycle pool.

    Each cycle of arity $k$ becomes a hyperedge of cardinality $k$,
    with all $k$ incident arcs carrying the cycle's $\\sigma$-product.
    This is the cycle-as-hyperedge construction used by Gömb-strict's
    core shell.

    Parameters
    ----------
    cycle_vertex_lists : list of LongTensors
        Per-cycle vertex sequence; one tensor of shape ``(k_i,)`` per
        cycle. Arities $k_i$ may vary across the list.
    cycle_signs : FloatTensor, shape (n_cycles,) in {-1, +1}.
        Per-cycle $\\sigma$-product.
    n_vertices : int
    """
    n_cycles = len(cycle_vertex_lists)
    if cycle_signs.shape[0] != n_cycles:
        raise ValueError(
            f"cycle_signs length {cycle_signs.shape[0]} must match "
            f"len(cycle_vertex_lists) = {n_cycles}"
        )
    rows_chunks = []
    cols_chunks = []
    vals_chunks = []
    sizes = []
    for ci, verts in enumerate(cycle_vertex_lists):
        k = int(verts.shape[0])
        sizes.append(k)
        rows_chunks.append(verts.long())
        cols_chunks.append(torch.full((k,), ci, dtype=torch.long, device=verts.device))
        vals_chunks.append(
            torch.full((k,), float(cycle_signs[ci].item()),
                        dtype=torch.float32, device=verts.device),
        )
    if n_cycles == 0:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros((0,), dtype=torch.float32)
        edge_sizes = torch.zeros((0,), dtype=torch.long)
    else:
        rows = torch.cat(rows_chunks, dim=0)
        cols = torch.cat(cols_chunks, dim=0)
        vals = torch.cat(vals_chunks, dim=0)
        indices = torch.stack([rows, cols], dim=0)
        values = vals
        edge_sizes = torch.tensor(sizes, dtype=torch.long, device=values.device)
    return StarTensor(
        indices=indices,
        values=values,
        n_vertices=int(n_vertices),
        n_hyperedges=int(n_cycles),
        edge_sizes=edge_sizes,
    )


def verify_star_clique_identity(
    star: StarTensor, atol: float = 1e-5,
) -> bool:
    """Verify the star-clique identity numerically on a small fixture.

    Returns True iff
    ``M D^{-1} M^T == A^{△} + diag(M D^{-1} M^T)``
    holds within the absolute tolerance. Used by tests; not on the
    hot path.
    """
    M = star.to_sparse_coo().to_dense()
    inv_sizes = 1.0 / star.edge_sizes.to(M.dtype).clamp(min=1.0)
    full = (M * inv_sizes.view(1, -1)) @ M.t()
    off_diag = star.to_clique_adjacency(include_self_loops=False)
    diag = torch.diag(torch.diag(full))
    reconstructed = off_diag + diag
    return torch.allclose(full, reconstructed, atol=atol)
