"""AdaptiveQuadtree — content-driven recursive patch subdivision.

GömbSoma-Ricci-Stim phase 2. Replaces the rolling-shutter uniform-grid
anchor tiling with a quadtree whose subdivision decisions are driven
by per-region content complexity:

  * Pixel-variance criterion (default, content-driven): subdivide a
    patch if the standard deviation of its pixel values exceeds a
    threshold. Cheap, robust, and the natural baseline signal.
  * Forman-Ricci criterion (optional, structural): subdivide a patch
    if its vertex curvature on the same-scale 4-connected graph
    exceeds a threshold in magnitude. Combinable with the variance
    criterion via the ``variance_weight`` / ``curvature_weight``
    coefficients.

The score is
    s(v) = variance_weight * std(pixels in v's region)
         + curvature_weight * |kappa_v|
and v subdivides if ``s(v) > score_threshold``.

The output is a multi-scale anchor tree: the original coarse anchors
PLUS their refinements at deeper scales. Cross-scale parent-child
edges are recorded in ``parent_indices``. The Walk / Polygon /
Triangle layers (Phase 2 / 3-G / 4 of the main GömbSoma plan) are
topology-agnostic and consume the resulting hypergraph natively.

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.

Note on the κ-only criterion. On a uniform 4-connected grid the
combinatorial Forman κ is degenerate (every interior vertex has the
same degree and zero triangles), so |κ_v| alone cannot discriminate
between anchor positions. We therefore default to variance-only
subdivision; κ becomes meaningful after the SDRF rewiring phase
(Phase 6) and is then available as an optional weight here. This is
documented honestly in the phase-2 report rather than papered over.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from signedkan_wip.src.hymeko_gomb.soma.vision.forman import (
    FormanCurvatureHead,
)


@dataclass
class AnchorTree:
    """Multi-scale anchor geometry output.

    Attributes
    ----------
    positions : LongTensor[n, 2]
        (row, col) top-left pixel coordinate of each anchor.
    sizes : LongTensor[n]
        Side length in pixels of each anchor's region.
    scales : LongTensor[n]
        Depth in the quadtree (0 = coarsest).
    parent_indices : LongTensor[n]
        Index into the same tree of each anchor's parent; -1 for the
        scale-0 anchors.
    """

    positions: torch.Tensor
    sizes: torch.Tensor
    scales: torch.Tensor
    parent_indices: torch.Tensor

    @property
    def n_anchors(self) -> int:
        return int(self.positions.shape[0])


class AdaptiveQuadtree(nn.Module):
    """κ- and variance-driven recursive quadtree subdivision.

    Preconditions
    -------------
    * ``image_h % patch_size_initial == 0`` and ditto for width.
    * ``patch_size_initial`` and ``patch_size_min`` are powers of 2,
      with ``patch_size_min`` ≤ ``patch_size_initial``.
    * Score weights ≥ 0; threshold any float; at least one of
      ``variance_weight``, ``curvature_weight`` is positive.

    Postconditions
    --------------
    * ``AnchorTree.scales.max()`` ≤ ``max_depth``.
    * ``AnchorTree.n_anchors`` ≤ ``max_anchors``.
    * Children of a parent tile the parent's region with no gaps or
      overlaps; each child has half the parent's side length.
    * Deterministic: same image + same hyperparameters ⇒ same output.
    """

    def __init__(
        self,
        image_h: int,
        image_w: int,
        patch_size_initial: int,
        patch_size_min: int = 1,
        max_depth: int | None = None,
        max_anchors: int = 1024,
        variance_weight: float = 1.0,
        curvature_weight: float = 0.0,
        score_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        if patch_size_initial < 1:
            raise ValueError(
                f"patch_size_initial must be >= 1, got {patch_size_initial}"
            )
        if patch_size_min < 1 or patch_size_min > patch_size_initial:
            raise ValueError(
                f"patch_size_min must satisfy 1 <= "
                f"patch_size_min ({patch_size_min}) <= "
                f"patch_size_initial ({patch_size_initial})"
            )
        if image_h % patch_size_initial != 0 or image_w % patch_size_initial != 0:
            raise ValueError(
                f"image size ({image_h}, {image_w}) must be divisible by "
                f"patch_size_initial ({patch_size_initial})"
            )
        if variance_weight < 0 or curvature_weight < 0:
            raise ValueError("score weights must be non-negative")
        if variance_weight == 0 and curvature_weight == 0:
            raise ValueError(
                "at least one of variance_weight or curvature_weight "
                "must be positive"
            )
        self.image_h = image_h
        self.image_w = image_w
        self.patch_size_initial = patch_size_initial
        self.patch_size_min = patch_size_min
        if max_depth is None:
            # Max meaningful depth is when child_size hits patch_size_min.
            max_depth = int(math.log2(patch_size_initial // patch_size_min))
        self.max_depth = max_depth
        self.max_anchors = max_anchors
        self.variance_weight = variance_weight
        self.curvature_weight = curvature_weight
        self.score_threshold = score_threshold
        self.forman = FormanCurvatureHead()

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, image: torch.Tensor) -> AnchorTree:
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

        # Build scale-0 anchors as uniform tiling.
        positions: list[tuple[int, int]] = []
        sizes: list[int] = []
        scales: list[int] = []
        parent_indices: list[int] = []
        for r in range(0, self.image_h, self.patch_size_initial):
            for c_ in range(0, self.image_w, self.patch_size_initial):
                positions.append((r, c_))
                sizes.append(self.patch_size_initial)
                scales.append(0)
                parent_indices.append(-1)

        # Subdivide depth by depth. At each depth, examine the current
        # frontier (anchors at this depth) and decide which to split.
        current_depth = 0
        frontier = list(range(len(positions)))

        while frontier and current_depth < self.max_depth:
            scores = self._score_frontier(image, positions, sizes, frontier)
            new_frontier: list[int] = []
            for local_i, anchor_i in enumerate(frontier):
                if scores[local_i].item() <= self.score_threshold:
                    continue
                parent_size = sizes[anchor_i]
                child_size = parent_size // 2
                if child_size < self.patch_size_min:
                    continue
                if len(positions) + 4 > self.max_anchors:
                    break  # budget exhausted; stop adding
                pr, pc = positions[anchor_i]
                for dr in (0, child_size):
                    for dc in (0, child_size):
                        new_idx = len(positions)
                        positions.append((pr + dr, pc + dc))
                        sizes.append(child_size)
                        scales.append(current_depth + 1)
                        parent_indices.append(anchor_i)
                        new_frontier.append(new_idx)
            frontier = new_frontier
            current_depth += 1

        device = image.device
        return AnchorTree(
            positions=torch.tensor(positions, dtype=torch.long, device=device),
            sizes=torch.tensor(sizes, dtype=torch.long, device=device),
            scales=torch.tensor(scales, dtype=torch.long, device=device),
            parent_indices=torch.tensor(
                parent_indices, dtype=torch.long, device=device,
            ),
        )

    # -----------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------

    def _score_frontier(
        self,
        image: torch.Tensor,
        positions: list[tuple[int, int]],
        sizes: list[int],
        frontier: list[int],
    ) -> torch.Tensor:
        """Per-anchor subdivision score for the current frontier.

        Variance is computed in a single batched ``roi_align`` call
        (one CUDA op for all frontier anchors), giving the per-anchor
        mean intensity, then computing std as
        ``sqrt(E[x²] − (E[x])²)`` from two roi_align passes. This
        avoids the per-anchor Python ``region.std()`` loop.
        """
        device = image.device
        scores = torch.zeros(len(frontier), device=device)
        if self.variance_weight > 0 and frontier:
            from torchvision.ops import roi_align
            # Build boxes (n, 5) = [batch_idx=0, x1, y1, x2, y2].
            rs = torch.tensor(
                [positions[ai][0] for ai in frontier],
                dtype=torch.float32, device=device,
            )
            cs = torch.tensor(
                [positions[ai][1] for ai in frontier],
                dtype=torch.float32, device=device,
            )
            ss = torch.tensor(
                [sizes[ai] for ai in frontier],
                dtype=torch.float32, device=device,
            )
            boxes = torch.stack([
                torch.zeros_like(rs),
                cs, rs, cs + ss, rs + ss,
            ], dim=1)
            # Pool to 1×1 to get per-region mean.
            mean_pooled = roi_align(
                image.unsqueeze(0), boxes, output_size=1,
            ).reshape(len(frontier), -1).mean(dim=-1)
            # Pool squared image to get per-region mean of squares.
            mean_sq_pooled = roi_align(
                (image * image).unsqueeze(0), boxes, output_size=1,
            ).reshape(len(frontier), -1).mean(dim=-1)
            variance = (mean_sq_pooled - mean_pooled ** 2).clamp(min=0)
            std = variance.sqrt()
            scores += self.variance_weight * std
        if self.curvature_weight > 0:
            # Build a 4-conn adjacency on the frontier (anchors at the
            # current depth, sharing an edge in their spatial bounding
            # rectangles) and run FormanCurvatureHead.
            kappa = self._frontier_curvature(positions, sizes, frontier, device)
            scores += self.curvature_weight * kappa.abs()
        return scores

    def _frontier_curvature(
        self,
        positions: list[tuple[int, int]],
        sizes: list[int],
        frontier: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Vertex κ on the 4-connected frontier graph (current depth)."""
        n = len(frontier)
        # Build a quick reverse-index for spatial neighbour lookup.
        # Two anchors share a 4-conn edge iff they have the same size
        # and one is directly adjacent (top/bottom/left/right) by
        # exactly one side length.
        edges: list[tuple[int, int]] = []
        local_to_pos = [(positions[ai], sizes[ai]) for ai in frontier]
        pos_to_local: dict[tuple[int, int, int], int] = {
            (p[0], p[1], s): li for li, (p, s) in enumerate(local_to_pos)
        }
        for li, ((r, c), s) in enumerate(local_to_pos):
            for dr, dc in ((s, 0), (-s, 0), (0, s), (0, -s)):
                key = (r + dr, c + dc, s)
                if key in pos_to_local:
                    lj = pos_to_local[key]
                    if lj > li:  # add each undirected edge once
                        edges.append((li, lj))
        if not edges:
            return torch.zeros(n, device=device)
        edges_t = torch.tensor(edges, dtype=torch.long, device=device)
        out = self.forman(edges_t, n_nodes=n)
        return out.vertex_kappa
