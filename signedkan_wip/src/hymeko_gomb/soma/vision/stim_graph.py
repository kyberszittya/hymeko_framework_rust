"""StimulusGraphBuilder — turn an AnchorTree into a signed hypergraph.

GömbSoma-Ricci-Stim phase 5. Bridges Phase 2's `AdaptiveQuadtree`
output (a multi-scale geometry of anchors) and the
GömbSoma layer stack (Walk / Polygon / Triangle, optionally wrapped
in Bochner-coupling). For each `AnchorTree` + per-anchor feature
tensor, the builder produces:

  * a combined edge list (same-scale 4-connected siblings PLUS
    cross-scale parent–child links),
  * signed edges via feature-inner-product polarity,
  * walk / polygon / triangle primitives over the resulting graph,
  * σ-products per primitive (product of constituent edge signs),
  * per-primitive incidence matrices M_v,
  * per-primitive curvatures (mean Forman κ over the primitive's
    constituent edges),
  * the Hodge Laplacian Δ_0 on the anchor set.

The output ``StimulusGraph`` is consumed directly by
`BochnerHypergraphConv`-wrapped Walk / Polygon / Triangle layers:

    sg = builder(tree, anchor_features)
    bochner_walk.prepare(
        hodge_laplacian=sg.hodge_laplacian_0,
        primitive_curvatures=sg.walk_curvatures,
    )
    y = bochner_walk(anchor_features, sg.walks, sg.walk_signs, sg.M_v_walks)

Walks are length-2 (3 vertices, 2 edges); polygons are 4-cycles in
the same-scale grid; triangles are 3-cliques in the combined graph
(typically a parent + 2 of its adjacent same-scale children).

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from signedkan_wip.src.hymeko_gomb.soma.vision.forman import (
    FormanCurvatureHead,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.hodge import HodgeLaplacian
from signedkan_wip.src.hymeko_gomb.soma.vision.quadtree import AnchorTree


@dataclass
class StimulusGraph:
    """Multi-scale signed hypergraph derived from an AnchorTree."""

    # ----- edges (combined same-scale + cross-scale) -----
    edges: torch.Tensor              # LongTensor (n_edges, 2)
    edge_signs: torch.Tensor         # LongTensor (n_edges,) ∈ {-1, +1}
    edge_curvatures: torch.Tensor    # FloatTensor (n_edges,) Forman κ

    # ----- walks (length-2: 3 vertices, 2 edges, no backtracking) -----
    walks: torch.Tensor              # LongTensor (n_walks, 3)
    walk_signs: torch.Tensor         # LongTensor (n_walks,)
    M_v_walks: torch.Tensor          # sparse FloatTensor (n_anchors, n_walks)
    walk_curvatures: torch.Tensor    # FloatTensor (n_walks,)

    # ----- polygons (4-cycles in the same-scale grid) -----
    polygons: torch.Tensor           # LongTensor (n_polygons, 4)
    polygon_signs: torch.Tensor      # LongTensor (n_polygons,)
    M_v_polygons: torch.Tensor       # sparse FloatTensor (n_anchors, n_polygons)
    polygon_curvatures: torch.Tensor  # FloatTensor (n_polygons,)

    # ----- triangles (3-cliques in the combined graph) -----
    triangles: torch.Tensor          # LongTensor (n_triangles, 3)
    triangle_signs: torch.Tensor     # LongTensor (n_triangles,)
    M_v_triangles: torch.Tensor      # sparse FloatTensor (n_anchors, n_triangles)
    triangle_curvatures: torch.Tensor  # FloatTensor (n_triangles,)

    # ----- Hodge Laplacian on the anchor set -----
    hodge_laplacian_0: torch.Tensor  # sparse FloatTensor (n_anchors, n_anchors)

    n_anchors: int


class StimulusGraphBuilder(nn.Module):
    """Build a `StimulusGraph` from an `AnchorTree` + per-anchor features.

    Parameters
    ----------
    max_walks, max_polygons, max_triangles : int
        Budget caps per primitive family. Enumeration stops once the
        cap is reached (deterministic order).
    sign_threshold : float
        Threshold θ in σ(u, v) = sign(⟨f_u, f_v⟩ − θ). Default 0 → an
        edge is positive iff the feature inner product is non-negative.
    """

    def __init__(
        self,
        max_walks: int = 1024,
        max_polygons: int = 256,
        max_triangles: int = 256,
        sign_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_walks = max_walks
        self.max_polygons = max_polygons
        self.max_triangles = max_triangles
        self.sign_threshold = sign_threshold
        self.forman = FormanCurvatureHead()
        self.hodge = HodgeLaplacian()

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        anchor_tree: AnchorTree,
        anchor_features: torch.Tensor,
        edges_override: torch.Tensor | None = None,
        edge_signs_override: torch.Tensor | None = None,
    ) -> StimulusGraph:
        """Build a StimulusGraph from the anchor tree + features.

        Parameters
        ----------
        anchor_tree, anchor_features : as before.
        edges_override : optional Tensor[n_edges, 2]
            If provided, skip the internal same-scale + cross-scale
            edge construction and use these edges directly. Used by
            Phase-10 SDRF wiring: SDRF takes the initial edges,
            adds shortcut rewiring, and feeds the augmented set
            back here for primitive re-enumeration.
        edge_signs_override : optional Tensor[n_edges]
            Signs for the override edges. Required when
            edges_override is provided.
        """
        n = anchor_tree.n_anchors
        if anchor_features.ndim != 2 or anchor_features.shape[0] != n:
            raise ValueError(
                f"anchor_features must have shape ({n}, d), got "
                f"{tuple(anchor_features.shape)}"
            )
        device = anchor_features.device

        # ---- 1. Build edges (or use override) ----
        if edges_override is not None:
            if edge_signs_override is None:
                raise ValueError(
                    "edge_signs_override must be provided when "
                    "edges_override is set"
                )
            if edges_override.shape[0] != edge_signs_override.shape[0]:
                raise ValueError(
                    f"edges_override length {edges_override.shape[0]} "
                    f"!= edge_signs_override length "
                    f"{edge_signs_override.shape[0]}"
                )
            edges = edges_override.to(device)
            n_edges = edges.shape[0]
            edge_signs = edge_signs_override.to(device)
        else:
            same = self._same_scale_edges(anchor_tree)
            cross = self._cross_scale_edges(anchor_tree)
            edges = torch.cat([same, cross], dim=0).to(device)
            # Canonicalise each row to (min, max). Same-scale and
            # cross-scale builders never produce self-loops or
            # duplicates, so this single torch.sort suffices to
            # canonicalise the union; downstream Hodge can then skip
            # its own sort+unique pass.
            if edges.shape[0] > 0:
                edges, _ = torch.sort(edges, dim=1)
            n_edges = edges.shape[0]

            # ---- 2. Edge signs from feature inner products ----
            if n_edges == 0:
                edge_signs = torch.zeros(0, dtype=torch.long, device=device)
            else:
                u, v = edges[:, 0], edges[:, 1]
                dot = (anchor_features[u] * anchor_features[v]).sum(dim=-1)
                edge_signs = torch.where(
                    dot >= self.sign_threshold,
                    torch.ones(n_edges, dtype=torch.long, device=device),
                    -torch.ones(n_edges, dtype=torch.long, device=device),
                )

        # ---- 3. Edge curvatures (Forman κ) ----
        if n_edges == 0:
            edge_curv = torch.zeros(0, device=device)
        else:
            forman_out = self.forman(edges, n_nodes=n)
            edge_curv = forman_out.edge_kappa

        # ---- 4. Adjacency for primitive enumeration ----
        # Dense adjacency + edge-index matrices for vectorised
        # primitive enumeration. Memory: 2 × (n × n) long ≈ 0.5 MB at
        # n=256. Polygon enumeration uses the Python dict
        # `edge_lookup` because the plaquette walker keys by geometric
        # (anchor-id pair) lookup rather than by tensor index.
        edge_lookup: dict[tuple[int, int], int] = {}
        adj_dense = torch.zeros(n, n, dtype=torch.bool, device=device)
        edge_index_dense = torch.full(
            (n, n), -1, dtype=torch.long, device=device,
        )
        if n_edges > 0:
            u_e = edges[:, 0].long()
            v_e = edges[:, 1].long()
            adj_dense[u_e, v_e] = True
            adj_dense[v_e, u_e] = True
            arange_e = torch.arange(n_edges, device=device)
            edge_index_dense[u_e, v_e] = arange_e
            edge_index_dense[v_e, u_e] = arange_e
            # Keep the Python dict in sync for polygon enumeration which
            # uses geometric (position-based) keys.
            for i, (a, b) in enumerate(edges.tolist()):
                key = (min(a, b), max(a, b))
                if key not in edge_lookup:
                    edge_lookup[key] = i

        # ---- 5. Walks (length-2; 3 vertices, 2 edges) — vectorised ----
        walks, walk_eidx = self._enumerate_walks_vectorized(
            adj_dense, edge_index_dense, n, device,
        )
        # Honour the max_walks budget (deterministic prefix).
        if walks.shape[0] > self.max_walks:
            walks = walks[: self.max_walks]
            walk_eidx = walk_eidx[: self.max_walks]
        if walks.shape[0] > 0:
            walk_signs = edge_signs[walk_eidx].prod(dim=1).to(torch.long)
            walk_curv = edge_curv[walk_eidx].float().mean(dim=1)
        else:
            walk_signs = torch.zeros(0, dtype=torch.long, device=device)
            walk_curv = torch.zeros(0, device=device)
        M_v_walks = self._build_M_v(walks, n, k=3, device=device)

        # ---- 6. Polygons (4-cycles in same-scale grid) ----
        polys_list, poly_eidx_list = self._enumerate_polygons(
            anchor_tree, edge_lookup,
        )
        if polys_list:
            polys = torch.tensor(polys_list, dtype=torch.long, device=device)
            poly_eidx = torch.tensor(
                poly_eidx_list, dtype=torch.long, device=device,
            )
            poly_signs = edge_signs[poly_eidx].prod(dim=1).to(torch.long)
            poly_curv = edge_curv[poly_eidx].mean(dim=1)
        else:
            polys = torch.zeros((0, 4), dtype=torch.long, device=device)
            poly_signs = torch.zeros(0, dtype=torch.long, device=device)
            poly_curv = torch.zeros(0, device=device)
        M_v_polys = self._build_M_v(polys, n, k=4, device=device)

        # ---- 7. Triangles (3-cliques) — vectorised ----
        tris, tri_eidx = self._enumerate_triangles_vectorized(
            adj_dense, edge_index_dense, n, device,
        )
        if tris.shape[0] > self.max_triangles:
            tris = tris[: self.max_triangles]
            tri_eidx = tri_eidx[: self.max_triangles]
        if tris.shape[0] > 0:
            tri_signs = edge_signs[tri_eidx].prod(dim=1).to(torch.long)
            tri_curv = edge_curv[tri_eidx].float().mean(dim=1)
        else:
            tri_signs = torch.zeros(0, dtype=torch.long, device=device)
            tri_curv = torch.zeros(0, device=device)
        M_v_tris = self._build_M_v(tris, n, k=3, device=device)

        # ---- 8. Hodge Laplacian Δ_0 ----
        # In the non-override branch we canonicalised `edges` above
        # (sorted rows, no self-loops, no duplicates by construction).
        # In the override branch the caller is SDRF, which produces
        # rows in arbitrary order; we cannot assume canonicality.
        hodge_out = self.hodge(
            edges, n_vertices=n,
            triangles=tris if tris.shape[0] > 0 else None,
            edges_already_canonical=(edges_override is None),
        )

        return StimulusGraph(
            edges=edges,
            edge_signs=edge_signs,
            edge_curvatures=edge_curv,
            walks=walks,
            walk_signs=walk_signs,
            M_v_walks=M_v_walks,
            walk_curvatures=walk_curv,
            polygons=polys,
            polygon_signs=poly_signs,
            M_v_polygons=M_v_polys,
            polygon_curvatures=poly_curv,
            triangles=tris,
            triangle_signs=tri_signs,
            M_v_triangles=M_v_tris,
            triangle_curvatures=tri_curv,
            hodge_laplacian_0=hodge_out.laplacian_0,
            n_anchors=n,
        )

    # -----------------------------------------------------------------
    # Edge builders
    # -----------------------------------------------------------------

    def _same_scale_edges(self, t: AnchorTree) -> torch.Tensor:
        """4-connected adjacency between same-scale, same-size anchors."""
        pos = t.positions.tolist()
        siz = t.sizes.tolist()
        sca = t.scales.tolist()
        # Index anchors by (row, col, size, scale) for O(1) neighbour lookup.
        lookup: dict[tuple[int, int, int, int], int] = {}
        for i, ((r, c), s, sc) in enumerate(zip(pos, siz, sca)):
            lookup[(r, c, s, sc)] = i
        edges = []
        seen: set[tuple[int, int]] = set()
        for i, ((r, c), s, sc) in enumerate(zip(pos, siz, sca)):
            for dr, dc in ((s, 0), (-s, 0), (0, s), (0, -s)):
                key = (r + dr, c + dc, s, sc)
                if key in lookup:
                    j = lookup[key]
                    pair = (min(i, j), max(i, j))
                    if pair not in seen:
                        seen.add(pair)
                        edges.append([i, j])
        if not edges:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long)

    def _cross_scale_edges(self, t: AnchorTree) -> torch.Tensor:
        """Parent–child edges from AnchorTree.parent_indices."""
        edges = []
        pi = t.parent_indices.tolist()
        for child, parent in enumerate(pi):
            if parent >= 0:
                edges.append([child, parent])
        if not edges:
            return torch.zeros((0, 2), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long)

    # -----------------------------------------------------------------
    # Primitive enumeration
    # -----------------------------------------------------------------

    def _enumerate_polygons(
        self,
        t: AnchorTree,
        edge_lookup: dict[tuple[int, int], int],
    ) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
        """4-cycle plaquettes in the same-scale grid at each depth.

        A plaquette is identified by its top-left anchor v_tl with
        same-size right neighbour v_tr, bottom neighbour v_bl, and
        diagonal v_br = (right then down). All four must exist at
        the same scale and size.
        """
        pos = t.positions.tolist()
        siz = t.sizes.tolist()
        sca = t.scales.tolist()
        lookup: dict[tuple[int, int, int, int], int] = {}
        for i, ((r, c), s, sc) in enumerate(zip(pos, siz, sca)):
            lookup[(r, c, s, sc)] = i
        polys: list[tuple[int, int, int, int]] = []
        eidx: list[tuple[int, int, int, int]] = []
        for i, ((r, c), s, sc) in enumerate(zip(pos, siz, sca)):
            # Top-left of a 2×2 plaquette: need (r, c+s), (r+s, c), (r+s, c+s).
            tr_key = (r,     c + s, s, sc)
            bl_key = (r + s, c,     s, sc)
            br_key = (r + s, c + s, s, sc)
            if tr_key in lookup and bl_key in lookup and br_key in lookup:
                tr = lookup[tr_key]
                bl = lookup[bl_key]
                br = lookup[br_key]
                # Cycle order: tl -> tr -> br -> bl -> tl.
                # Edges of the cycle: (i,tr), (tr,br), (br,bl), (bl,i).
                # All four must be present in the edge_lookup — under
                # override paths, the supplied edge set may not include
                # the implied same-scale grid edges, so we guard the
                # lookup.
                def lookup_edge(a: int, b: int) -> int | None:
                    return edge_lookup.get((min(a, b), max(a, b)))
                ea = lookup_edge(i, tr)
                eb = lookup_edge(tr, br)
                ec = lookup_edge(br, bl)
                ed = lookup_edge(bl, i)
                if None in (ea, eb, ec, ed):
                    continue  # plaquette edges not all in the active edge set
                polys.append((i, tr, br, bl))
                eidx.append((ea, eb, ec, ed))
                if len(polys) >= self.max_polygons:
                    break
        return polys, eidx

    # -----------------------------------------------------------------
    # Sparse incidence
    # -----------------------------------------------------------------

    @staticmethod
    def _enumerate_walks_vectorized(
        adj_dense: torch.Tensor,
        edge_index_dense: torch.Tensor,
        n: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorised length-2 walk enumeration via dense adjacency.

        For each vertex b, all (a, c) pairs with a ∈ adj(b), c ∈ adj(b),
        a ≠ c form a walk (a, b, c).

        Builds a 3D boolean mask ``walks_mask[b, a, c]`` of shape
        ``(n, n, n)`` — at n = 256 that is 16 MB of bool memory.
        Acceptable for our anchor scale and overwhelmingly faster
        than the Python nested-loop equivalent.
        """
        if n == 0 or adj_dense.sum() == 0:
            return (
                torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0, 2), dtype=torch.long, device=device),
            )
        # walks_mask[b, a, c] = adj[b, a] AND adj[b, c] AND a != c
        not_self = ~torch.eye(n, dtype=torch.bool, device=device)
        walks_mask = (
            adj_dense.unsqueeze(2)              # (n, n, 1) for a
            & adj_dense.unsqueeze(1)            # (n, 1, n) for c
            & not_self.unsqueeze(0)             # a != c
        )
        # walks_mask is indexed [b, a, c]. Extract.
        b_idx, a_idx, c_idx = walks_mask.nonzero(as_tuple=True)
        if b_idx.numel() == 0:
            return (
                torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0, 2), dtype=torch.long, device=device),
            )
        walks = torch.stack([a_idx, b_idx, c_idx], dim=1)
        # Edge indices: (a, b) → e1, (b, c) → e2.
        e1 = edge_index_dense[a_idx, b_idx]
        e2 = edge_index_dense[b_idx, c_idx]
        eidx = torch.stack([e1, e2], dim=1)
        return walks, eidx

    @staticmethod
    def _enumerate_triangles_vectorized(
        adj_dense: torch.Tensor,
        edge_index_dense: torch.Tensor,
        n: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorised 3-clique enumeration.

        Triangles are canonical-ordered triples (a, b, c) with
        a < b < c and all three pairwise edges present. The mask:

            mask[a, b, c] = adj[a, b] AND adj[a, c] AND adj[b, c]
                            AND a < b AND b < c

        ``(n, n, n)`` bool tensor — same memory bound as walks.
        """
        if n == 0 or adj_dense.sum() == 0:
            return (
                torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device),
            )
        # Canonical-order mask: a < b < c.
        idx = torch.arange(n, device=device)
        lt_ab = idx.unsqueeze(0) < idx.unsqueeze(1)   # (n, n): row > col is False, etc.
        # We want a < b: idx[a] < idx[b]. With a as dim-0, b as dim-1: a < b → row idx < col idx.
        a_lt_b = idx.unsqueeze(1) < idx.unsqueeze(0)  # (n, n): a_lt_b[a, b] = a < b
        # Mask: a < b AND b < c
        a_lt_b_3d = a_lt_b.unsqueeze(2)               # (n, n, 1): on (a, b)
        b_lt_c_3d = a_lt_b.unsqueeze(0)               # (1, n, n): on (b, c)
        canonical_mask = a_lt_b_3d & b_lt_c_3d        # (n, n, n)
        # Edge mask: adj[a, b] AND adj[a, c] AND adj[b, c]
        edges_mask = (
            adj_dense.unsqueeze(2)         # (n, n, 1): adj[a, b]
            & adj_dense.unsqueeze(1)       # (n, 1, n): adj[a, c]
            & adj_dense.unsqueeze(0)       # (1, n, n): adj[b, c]
        )
        full_mask = canonical_mask & edges_mask
        a_idx, b_idx, c_idx = full_mask.nonzero(as_tuple=True)
        if a_idx.numel() == 0:
            return (
                torch.zeros((0, 3), dtype=torch.long, device=device),
                torch.zeros((0, 3), dtype=torch.long, device=device),
            )
        tris = torch.stack([a_idx, b_idx, c_idx], dim=1)
        e_ab = edge_index_dense[a_idx, b_idx]
        e_ac = edge_index_dense[a_idx, c_idx]
        e_bc = edge_index_dense[b_idx, c_idx]
        eidx = torch.stack([e_ab, e_ac, e_bc], dim=1)
        return tris, eidx

    @staticmethod
    def _build_M_v(
        primitives: torch.Tensor,
        n_anchors: int,
        k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Vertex-to-primitive incidence, uniform weight 1/k per (v, p)."""
        n_p = primitives.shape[0]
        if n_p == 0:
            return torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), device=device),
                (n_anchors, n_p),
            ).coalesce()
        rows = primitives.reshape(-1)
        cols = torch.arange(n_p, device=device).repeat_interleave(k)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.full((rows.shape[0],), 1.0 / k, device=device)
        return torch.sparse_coo_tensor(
            indices, values, (n_anchors, n_p),
        ).coalesce()
