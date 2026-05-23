"""Hodge Laplacians on a signed simplicial complex.

GömbSoma-Ricci-Stim phase 3. Builds the signed boundary operators
$\\partial_k$ and Hodge Laplacians $\\Delta_k$ at the three dimensions
relevant to vision: $k = 0$ (vertices), $k = 1$ (edges), $k = 2$
(triangles).

Boundary formulas (canonical / sorted-vertex orientation)
----------------------------------------------------------
For an edge $e = [u, v]$ with $u < v$:

    \\partial_1 e = +[v] - [u]

For a triangle $t = [v_0, v_1, v_2]$ with $v_0 < v_1 < v_2$:

    \\partial_2 t = +[v_1, v_2] - [v_0, v_2] + [v_0, v_1].

Hodge Laplacians
----------------
    \\Delta_0 = \\partial_1 \\partial_1^\\top                                 (n_v × n_v)
    \\Delta_1 = \\partial_1^\\top \\partial_1 + \\partial_2 \\partial_2^\\top  (n_e × n_e)
    \\Delta_2 = \\partial_2^\\top \\partial_2                                 (n_t × n_t)

$\\Delta_0$ reduces to the standard graph Laplacian $D - A$.
$\\Delta_1$ acts on edge-feature vectors (signed edge flow).
$\\Delta_2$ acts on triangle-feature vectors (face flow).

The fundamental identity $\\partial_1 \\partial_2 = 0$ is pinned by
unit test --- this is what makes the Hodge Laplacians well-defined
and what underpins the Hodge decomposition theorem.

All boundary and Laplacian tensors are returned as sparse COO
tensors. They are recomputable from `(edges, n_vertices, triangles)`
deterministically.

Used by: Bochner-coupled HypergraphConv (phase 4) for the
Hodge-smoothing term $\\alpha \\cdot (\\Delta_k h)$ in the message-
passing decomposition.

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HodgeOperators:
    """Bundle of boundaries and Hodge Laplacians for a signed complex.

    Tensors are sparse COO. Convert to dense via ``.to_dense()`` for
    spectral analysis on small examples; keep sparse for downstream
    propagation on large complexes.
    """

    boundary_1: torch.Tensor       # sparse (n_v, n_e)
    boundary_2: torch.Tensor       # sparse (n_e, n_t)  — empty if no triangles
    laplacian_0: torch.Tensor      # sparse (n_v, n_v)
    laplacian_1: torch.Tensor      # sparse (n_e, n_e)
    laplacian_2: torch.Tensor      # sparse (n_t, n_t)  — empty if no triangles
    n_vertices: int
    n_edges: int
    n_triangles: int


class HodgeLaplacian(nn.Module):
    """Compute $\\partial_k$ and $\\Delta_k$ on a signed simplicial complex.

    Preconditions
    -------------
    * Edges and triangles are non-negative integer index arrays.
    * Self-loops $(u, u)$ in edges are silently dropped.
    * Degenerate triangles (any two indices equal) are silently dropped.

    Postconditions
    --------------
    * Output is a HodgeOperators bundle of sparse COO tensors.
    * $\\partial_1 \\partial_2 = 0$ exactly (pinned by unit test).
    * $\\Delta_0$ equals the standard graph Laplacian $D - A$.
    """

    def __init__(self) -> None:
        super().__init__()

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        edges: torch.Tensor,
        n_vertices: int,
        triangles: torch.Tensor | None = None,
        edges_already_canonical: bool = False,
    ) -> HodgeOperators:
        """Compute Hodge operators.

        Preconditions
        -------------
        * ``edges`` has shape ``(n_e, 2)``.
        * ``triangles`` is ``None`` or has shape ``(n_t, 3)``.
        * If ``edges_already_canonical`` is True, the caller guarantees
          that each row is sorted ascending, no self-loops, no
          duplicates. Triangles do not need a parallel flag: their
          canonicalisation is cheap (``2 n_t`` scalar ops) and is
          retained unconditionally.

        Postconditions
        --------------
        * Output Hodge operators are byte-identical (modulo floating-
          point summation order in the sparse mm) regardless of
          ``edges_already_canonical`` — the flag is a pure performance
          hint. Verified by the regression test
          ``test_hodge_canonical_flag_invariance``.
        """
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(
                f"edges must have shape (n_edges, 2); got "
                f"{tuple(edges.shape)}"
            )
        if triangles is not None and triangles.numel() > 0:
            if triangles.ndim != 2 or triangles.shape[1] != 3:
                raise ValueError(
                    f"triangles must have shape (n_triangles, 3); got "
                    f"{tuple(triangles.shape)}"
                )
        device = edges.device

        # --- Canonicalise edges: sort each row, drop self-loops, dedupe ---
        if edges_already_canonical:
            edges_canonical = edges
            if __debug__ and edges.numel() > 0:
                # Cheap precondition check: sorted-row + no self-loop.
                # Skips the (relatively expensive) unique check; a
                # duplicate would manifest as a wrong-shape Laplacian
                # that the test suite picks up.
                assert bool(
                    (edges[:, 0] < edges[:, 1]).all().item()
                ), "edges_already_canonical=True but row(0) !< row(1) somewhere"
        else:
            edges_sorted, _ = torch.sort(edges, dim=1)
            keep = edges_sorted[:, 0] != edges_sorted[:, 1]
            edges_canonical = edges_sorted[keep]
            # Dedupe.
            if edges_canonical.shape[0] > 0:
                edges_canonical = torch.unique(edges_canonical, dim=0)
        n_e = edges_canonical.shape[0]

        # --- Canonicalise triangles ---
        if triangles is not None and triangles.numel() > 0:
            tri_sorted, _ = torch.sort(triangles, dim=1)
            keep_t = (
                (tri_sorted[:, 0] != tri_sorted[:, 1])
                & (tri_sorted[:, 1] != tri_sorted[:, 2])
                & (tri_sorted[:, 0] != tri_sorted[:, 2])
            )
            tri_canonical = tri_sorted[keep_t]
            if tri_canonical.shape[0] > 0:
                tri_canonical = torch.unique(tri_canonical, dim=0)
        else:
            tri_canonical = torch.zeros((0, 3), dtype=torch.long, device=device)
        n_t = tri_canonical.shape[0]

        # --- Build boundary_1: (n_v, n_e), entries ±1 ---
        if n_e > 0:
            rows = torch.cat([edges_canonical[:, 0], edges_canonical[:, 1]])
            cols = torch.arange(n_e, device=device).repeat(2)
            vals = torch.cat([
                -torch.ones(n_e, device=device),
                +torch.ones(n_e, device=device),
            ])
            boundary_1 = torch.sparse_coo_tensor(
                torch.stack([rows, cols], dim=0), vals, (n_vertices, n_e),
            ).coalesce()
        else:
            boundary_1 = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), device=device),
                (n_vertices, n_e),
            ).coalesce()

        # --- Build boundary_2: (n_e, n_t), entries +1, -1, +1 per triangle ---
        boundary_2 = self._build_boundary_2_vectorised(
            edges_canonical, tri_canonical, n_e, n_t, n_vertices, device,
        )

        # --- Laplacians ---
        # Δ_0 = ∂_1 ∂_1^T  (n_v × n_v)
        laplacian_0 = torch.sparse.mm(
            boundary_1, boundary_1.transpose(0, 1)
        ).coalesce()
        # Δ_1 = ∂_1^T ∂_1 + ∂_2 ∂_2^T  (n_e × n_e)
        if n_e > 0:
            term1 = torch.sparse.mm(
                boundary_1.transpose(0, 1), boundary_1
            ).coalesce()
            if n_t > 0:
                term2 = torch.sparse.mm(
                    boundary_2, boundary_2.transpose(0, 1)
                ).coalesce()
                laplacian_1 = (term1 + term2).coalesce()
            else:
                laplacian_1 = term1
        else:
            laplacian_1 = self._empty_sparse((0, 0), device)
        # Δ_2 = ∂_2^T ∂_2  (n_t × n_t)
        if n_t > 0:
            laplacian_2 = torch.sparse.mm(
                boundary_2.transpose(0, 1), boundary_2
            ).coalesce()
        else:
            laplacian_2 = self._empty_sparse((n_t, n_t), device)

        return HodgeOperators(
            boundary_1=boundary_1,
            boundary_2=boundary_2,
            laplacian_0=laplacian_0,
            laplacian_1=laplacian_1,
            laplacian_2=laplacian_2,
            n_vertices=n_vertices,
            n_edges=n_e,
            n_triangles=n_t,
        )

    # -----------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------

    @staticmethod
    def _empty_sparse(
        shape: tuple[int, int], device: torch.device,
    ) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), device=device),
            shape,
        ).coalesce()

    @staticmethod
    def _build_boundary_2_vectorised(
        edges_canonical: torch.Tensor,
        tri_canonical: torch.Tensor,
        n_e: int,
        n_t: int,
        n_vertices: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Assemble ``boundary_2`` of shape ``(n_e, n_t)`` without a
        Python loop over triangles.

        For a canonical triangle ``(v0, v1, v2)`` with
        ``v0 < v1 < v2``, the boundary is ``+[v1, v2] - [v0, v2] +
        [v0, v1]`` — three signed entries per triangle. We resolve
        edge identity by packing ``(u, v)`` into a single int64
        ``u * n_vertices + v`` and a ``torch.searchsorted`` against the
        (sorted) packed canonical edge keys.

        Missing edges (an undirected pair that does not appear in
        ``edges_canonical``) silently drop, matching the original
        Python-loop behaviour.

        Preconditions
        -------------
        * ``edges_canonical`` rows are sorted ascending; this is the
          same canonical form the Python-loop variant relied on.
        * ``tri_canonical`` rows are sorted ascending.
        * ``n_vertices >= max(edges_canonical) + 1``.
        """
        if n_t == 0 or n_e == 0:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), device=device),
                (n_e, n_t),
            ).coalesce()

        # Pack canonical (u, v) → u * n_vertices + v (u < v).
        packed_edges = (
            edges_canonical[:, 0].to(torch.long) * n_vertices
            + edges_canonical[:, 1].to(torch.long)
        )
        sort_perm = torch.argsort(packed_edges)
        sorted_packed = packed_edges[sort_perm]

        # Pack each triangle's three boundary pairs.
        v0 = tri_canonical[:, 0].to(torch.long)
        v1 = tri_canonical[:, 1].to(torch.long)
        v2 = tri_canonical[:, 2].to(torch.long)
        pack_v0v1 = v0 * n_vertices + v1
        pack_v0v2 = v0 * n_vertices + v2
        pack_v1v2 = v1 * n_vertices + v2

        # Resolve each pair against the canonical edge set.
        idx_v0v1_sorted = torch.searchsorted(sorted_packed, pack_v0v1)
        idx_v0v2_sorted = torch.searchsorted(sorted_packed, pack_v0v2)
        idx_v1v2_sorted = torch.searchsorted(sorted_packed, pack_v1v2)

        # Map sorted indices back to original edge indices; mark
        # out-of-range or non-matching as "missing" (-1) so we can
        # filter them.
        n_e_t = sorted_packed.shape[0]

        def _resolve(idx_sorted: torch.Tensor, pack: torch.Tensor) -> torch.Tensor:
            in_range = idx_sorted < n_e_t
            safe_idx = torch.where(
                in_range, idx_sorted, torch.zeros_like(idx_sorted),
            )
            matched = (
                in_range & (sorted_packed[safe_idx] == pack)
            )
            return torch.where(
                matched, sort_perm[safe_idx],
                torch.full_like(idx_sorted, -1),
            )

        eidx_v0v1 = _resolve(idx_v0v1_sorted, pack_v0v1)
        eidx_v0v2 = _resolve(idx_v0v2_sorted, pack_v0v2)
        eidx_v1v2 = _resolve(idx_v1v2_sorted, pack_v1v2)

        t_idx = torch.arange(n_t, device=device, dtype=torch.long)
        # Stack three signed contributions: +[v1,v2], -[v0,v2], +[v0,v1].
        rows = torch.cat([eidx_v1v2, eidx_v0v2, eidx_v0v1])
        cols = torch.cat([t_idx, t_idx, t_idx])
        vals = torch.cat([
            torch.ones(n_t, device=device),
            -torch.ones(n_t, device=device),
            torch.ones(n_t, device=device),
        ])
        keep = rows >= 0
        rows = rows[keep]
        cols = cols[keep]
        vals = vals[keep]
        if rows.numel() == 0:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), device=device),
                (n_e, n_t),
            ).coalesce()
        indices = torch.stack([rows, cols], dim=0)
        return torch.sparse_coo_tensor(
            indices, vals, (n_e, n_t),
        ).coalesce()
