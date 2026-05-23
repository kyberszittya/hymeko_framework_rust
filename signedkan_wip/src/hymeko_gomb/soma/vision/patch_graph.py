"""PatchGraphBuilder — turn an image into a signed 4-connected patch graph.

For an image of shape (C, H, W) and patch size P:
  * vertices: (H/P) × (W/P) flat patches
  * edges: between spatially 4-adjacent patches (manhattan-1)
  * signs: edge sign = +1 if src patch is brighter than dst patch
    (mean over channels), -1 otherwise
  * walks: all length-2 walks (3 vertices) over the grid
  * walk signs: σ-product of the walk's two constituent edges

The grid topology (edges, walks, M_v) is shared across the batch and
precomputed in __init__. Only signs and vertex features depend on
the image content.

Used by WalkConvImageClassifier in this subpackage.
"""
from __future__ import annotations

import torch


def _patch_index(r: int, c: int, w_patches: int) -> int:
    return r * w_patches + c


class PatchGraphBuilder:
    """Precompute and cache the patch-graph topology for a fixed image size.

    Parameters
    ----------
    image_h : int
        Image height in pixels. Must be divisible by ``patch_size``.
    image_w : int
        Image width in pixels.
    patch_size : int
        Side length of each square patch.

    Attributes
    ----------
    h_patches, w_patches : int
        Grid dimensions in patches.
    n_patches : int
        Total vertex count.
    edges : LongTensor[n_edges, 2]
        Directed edge list. Each undirected adjacency contributes
        BOTH directions so the sign function can be antisymmetric
        (brightness src vs dst).
    walks : LongTensor[n_walks, 3]
        All length-2 walks (no backtracking — middle vertex's two
        incident edges are required to be distinct).
    walk_edge_idx : LongTensor[n_walks, 2]
        Indices into ``edges`` of the two constituent edges of each
        walk. Used to derive walk signs from edge signs.
    M_v : sparse COO Tensor[n_patches, n_walks]
        Vertex-to-walk incidence with uniform weight 1/3.

    Preconditions
    -------------
    * image_h % patch_size == 0
    * image_w % patch_size == 0
    * patch_size >= 1
    """

    def __init__(
        self,
        image_h: int,
        image_w: int,
        patch_size: int,
    ) -> None:
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if image_h % patch_size != 0 or image_w % patch_size != 0:
            raise ValueError(
                f"image size ({image_h}, {image_w}) must be divisible by "
                f"patch_size ({patch_size})"
            )
        self.image_h = image_h
        self.image_w = image_w
        self.patch_size = patch_size
        self.h_patches = image_h // patch_size
        self.w_patches = image_w // patch_size
        self.n_patches = self.h_patches * self.w_patches

        self.edges = self._build_grid_edges()
        self.walks, self.walk_edge_idx = self._enumerate_walks()
        self.M_v = self._build_M_v()

    # -----------------------------------------------------------------
    # Topology construction
    # -----------------------------------------------------------------

    def _build_grid_edges(self) -> torch.Tensor:
        """4-connected grid, directed (src→dst pair per undirected adj)."""
        edges = []
        for r in range(self.h_patches):
            for c in range(self.w_patches):
                v = _patch_index(r, c, self.w_patches)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h_patches and 0 <= nc < self.w_patches:
                        u = _patch_index(nr, nc, self.w_patches)
                        edges.append((v, u))
        return torch.tensor(edges, dtype=torch.long)

    def _enumerate_walks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """All length-2 walks (a, b, c) with a ≠ c (no backtracking).

        Returns
        -------
        walks : LongTensor[n_walks, 3]
        walk_edge_idx : LongTensor[n_walks, 2]
            For each walk, the indices in ``self.edges`` of edges
            (a→b) and (b→c).
        """
        # Build an adjacency list from edges.
        adj: list[list[tuple[int, int]]] = [[] for _ in range(self.n_patches)]
        for e_idx, (src, dst) in enumerate(self.edges.tolist()):
            adj[src].append((dst, e_idx))

        walks = []
        walk_edge_idx = []
        for b in range(self.n_patches):
            # All pairs of distinct neighbours of b.
            # adj[b] gives (neighbour, edge_idx) where edge_idx is the
            # b→neighbour direction. For the walk a→b→c we want the
            # a→b edge (from adj[a]) and the b→c edge (from adj[b]).
            neigh = adj[b]
            for (a_in, _b_to_a_idx) in neigh:
                # Find the index of edge a→b in adj[a].
                e_ab = next(
                    idx for (dst, idx) in adj[a_in] if dst == b
                )
                for (c_out, e_bc_idx) in neigh:
                    if c_out == a_in:
                        continue  # no backtracking
                    walks.append((a_in, b, c_out))
                    walk_edge_idx.append((e_ab, e_bc_idx))
        return (
            torch.tensor(walks, dtype=torch.long),
            torch.tensor(walk_edge_idx, dtype=torch.long),
        )

    def _build_M_v(self) -> torch.Tensor:
        """Sparse vertex-to-walk incidence; uniform 1/3 per (v, walk)."""
        n_walks = self.walks.shape[0]
        if n_walks == 0:
            indices = torch.zeros((2, 0), dtype=torch.long)
            values = torch.zeros((0,))
            return torch.sparse_coo_tensor(
                indices, values, (self.n_patches, n_walks),
            ).coalesce()
        rows = self.walks.reshape(-1)
        cols = torch.arange(n_walks).repeat_interleave(3)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.full((rows.shape[0],), 1.0 / 3.0)
        return torch.sparse_coo_tensor(
            indices, values, (self.n_patches, n_walks),
        ).coalesce()

    # -----------------------------------------------------------------
    # Per-image encoding
    # -----------------------------------------------------------------

    def patchify(self, image: torch.Tensor) -> torch.Tensor:
        """Turn (C, H, W) image into (n_patches, C * patch_size^2)."""
        if image.ndim != 3:
            raise ValueError(
                f"expected (C, H, W) image, got shape {tuple(image.shape)}"
            )
        c, h, w = image.shape
        if h != self.image_h or w != self.image_w:
            raise ValueError(
                f"image shape ({h}, {w}) doesn't match builder "
                f"({self.image_h}, {self.image_w})"
            )
        # (C, H, W) → (n_patches, C * P * P)
        unfolded = image.unfold(
            1, self.patch_size, self.patch_size,
        ).unfold(2, self.patch_size, self.patch_size)
        # shape: (C, h_patches, w_patches, P, P)
        unfolded = unfolded.permute(1, 2, 0, 3, 4).contiguous()
        # shape: (h_patches, w_patches, C, P, P)
        return unfolded.view(self.n_patches, -1)

    def edge_signs(self, patches: torch.Tensor) -> torch.Tensor:
        """For each edge (src→dst), σ = +1 if mean(src) ≥ mean(dst) else -1.

        Parameters
        ----------
        patches : Tensor[n_patches, patch_dim]

        Returns
        -------
        LongTensor[n_edges]
        """
        if patches.shape[0] != self.n_patches:
            raise ValueError(
                f"patches has {patches.shape[0]} rows, expected "
                f"{self.n_patches}"
            )
        brightness = patches.mean(dim=-1)
        src = self.edges[:, 0].to(patches.device)
        dst = self.edges[:, 1].to(patches.device)
        return torch.where(
            brightness[src] >= brightness[dst],
            torch.ones(src.shape[0], dtype=torch.long, device=patches.device),
            -torch.ones(src.shape[0], dtype=torch.long, device=patches.device),
        )

    def walk_signs(self, edge_signs: torch.Tensor) -> torch.Tensor:
        """σ-product of each walk's two constituent edges."""
        e1 = self.walk_edge_idx[:, 0].to(edge_signs.device)
        e2 = self.walk_edge_idx[:, 1].to(edge_signs.device)
        return edge_signs[e1] * edge_signs[e2]

    def encode(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-shot encode: image → (patches, walks, walk_signs, M_v).

        Returns
        -------
        patches : Tensor[n_patches, patch_dim]
        walks : LongTensor[n_walks, 3]
        walk_signs : LongTensor[n_walks]
        M_v : sparse Tensor[n_patches, n_walks]
        """
        patches = self.patchify(image)
        e_signs = self.edge_signs(patches)
        w_signs = self.walk_signs(e_signs)
        device = patches.device
        return (
            patches,
            self.walks.to(device),
            w_signs,
            self.M_v.to(device),
        )

    def __repr__(self) -> str:
        return (
            f"PatchGraphBuilder(image=({self.image_h}, {self.image_w}), "
            f"patch_size={self.patch_size}, "
            f"n_patches={self.n_patches}, n_edges={self.edges.shape[0]}, "
            f"n_walks={self.walks.shape[0]})"
        )
