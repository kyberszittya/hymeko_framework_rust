"""RicciStimBackbone — shared feature-extraction backbone.

GömbSoma-Ricci-Stim phase 9 (consolidation). The classifier (Phase 7)
and detector (Phase 8) had identical feature-extraction logic:

  image
    → AdaptiveQuadtree
    → per-anchor adaptive-avg-pool + Linear patch encoder
    → StimulusGraphBuilder
    → 3 parallel Bochner-wrapped Walk / Polygon / Triangle branches
    → sum

The only structural difference between classifier and detector was
the head (pooled cls vs per-anchor cls + bbox). Phase 9 consolidates
the backbone into a single shared module; the classifier and
detector become thin wrappers that add their head on top.

This clears the §6.5 #3 anti-pattern flag from Phase 8 (per-experiment
scaffold duplication) and keeps the backbone available as a building
block for future heads (segmentation, keypoint detection, dense
prediction, etc.).

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from signedkan_wip.src.hymeko_gomb.soma import (
    BochnerHypergraphConv,
    HypergraphConvConfig,
    PolygonConvLayer,
    WalkConvLayer,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.quadtree import (
    AdaptiveQuadtree,
    AnchorTree,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.sdrf import (
    SDRFRewiring,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.stim_graph import (
    StimulusGraphBuilder,
)


class RicciStimBackbone(nn.Module):
    """Shared feature-extraction backbone for Ricci-Stim heads.

    Forward signature
    -----------------
    image : Tensor[C, H, W]

    Returns
    -------
    features : Tensor[n_anchors, d_hidden]
        Per-anchor features after the 3 Bochner-wrapped branches.
    tree : AnchorTree
        The per-image multi-scale anchor geometry; needed by heads
        that emit per-anchor outputs (detector) or by any decoder
        downstream.

    Postconditions
    --------------
    * ``features.shape == (tree.n_anchors, self.d_hidden)``
    * The 3 branches (walk / polygon / triangle) are summed; empty
      primitive families contribute zero (graceful degradation).
    """

    def __init__(
        self,
        image_h: int = 28,
        image_w: int = 28,
        patch_size_initial: int = 4,
        patch_size_min: int = 1,
        in_channels: int = 1,
        d_hidden: int = 16,
        max_depth: int | None = 2,
        max_anchors: int = 256,
        score_threshold: float = 0.05,
        bochner_alpha: float = 0.0,
        bochner_beta: float = 0.0,
        use_sdrf: bool = False,
        sdrf_max_iters: int = 5,
        sdrf_kappa_target: float = -2.0,
    ) -> None:
        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.in_channels = in_channels
        self.d_hidden = d_hidden
        self.patch_fixed_size = 4
        self.use_sdrf = use_sdrf

        self.quadtree = AdaptiveQuadtree(
            image_h=image_h, image_w=image_w,
            patch_size_initial=patch_size_initial,
            patch_size_min=patch_size_min,
            max_depth=max_depth, max_anchors=max_anchors,
            score_threshold=score_threshold,
        )

        patch_dim = in_channels * (self.patch_fixed_size ** 2)
        self.patch_encoder = nn.Linear(patch_dim, d_hidden)

        self.graph_builder = StimulusGraphBuilder()
        # SDRF is a graph-functional (no learnable params) used to
        # relieve κ-bottlenecks before the conv branches see the
        # graph. Toggled by ``use_sdrf``.
        self.sdrf = SDRFRewiring(
            max_iters=sdrf_max_iters,
            min_kappa_target=sdrf_kappa_target,
        )

        walk_cfg = HypergraphConvConfig(
            in_features=d_hidden, out_features=d_hidden, k_arity=3,
        )
        self.walk_layer = BochnerHypergraphConv(
            WalkConvLayer(walk_cfg),
            alpha=bochner_alpha, beta=bochner_beta,
        )
        poly_cfg = HypergraphConvConfig(
            in_features=d_hidden, out_features=d_hidden, k_arity=4,
        )
        self.poly_layer = BochnerHypergraphConv(
            PolygonConvLayer(poly_cfg),
            alpha=bochner_alpha, beta=bochner_beta,
        )
        tri_cfg = HypergraphConvConfig(
            in_features=d_hidden, out_features=d_hidden, k_arity=3,
        )
        self.tri_layer = BochnerHypergraphConv(
            PolygonConvLayer(tri_cfg),
            alpha=bochner_alpha, beta=bochner_beta,
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self, image: torch.Tensor,
    ) -> tuple[torch.Tensor, AnchorTree]:
        if image.ndim != 3:
            raise ValueError(
                f"expected (C, H, W); got shape {tuple(image.shape)}"
            )
        tree = self.quadtree(image)
        features = self._encode_anchors(image, tree)
        sg = self.graph_builder(tree, features)
        if self.use_sdrf:
            # SDRF takes the initial edge set, adds shortcut rewiring.
            # If no shortcuts were added (e.g., on paths/stars where
            # the Forman κ heuristic finds no monotone improvement),
            # the rewired edge set is identical to the initial set —
            # the second graph_builder pass would produce a bit-identical
            # StimulusGraph. Skip it.
            sdrf_out = self.sdrf(
                sg.edges, n_vertices=tree.n_anchors,
                anchor_features=features, edge_signs=sg.edge_signs,
            )
            if sdrf_out.n_added > 0:
                sg = self.graph_builder(
                    tree, features,
                    edges_override=sdrf_out.edges,
                    edge_signs_override=sdrf_out.edge_signs,
                )
        h = (
            self._walk_branch(features, sg)
            + self._poly_branch(features, sg)
            + self._tri_branch(features, sg)
        )
        return h, tree

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _encode_anchors(
        self, image: torch.Tensor, tree: AnchorTree,
    ) -> torch.Tensor:
        """Batched per-anchor patch encoding via roi_align.

        Replaces the per-anchor Python-loop adaptive_avg_pool2d with a
        single CUDA call. roi_align does bilinear sampling to a fixed
        output size, which is the correct generalisation of avg-pooling
        to non-integer scale factors and lets us batch all anchors
        in one operation.
        """
        from torchvision.ops import roi_align
        n = tree.n_anchors
        device = image.device
        # roi_align expects boxes in (n, 5) format: [batch_idx, x1, y1, x2, y2].
        r = tree.positions[:, 0].float().to(device)
        c = tree.positions[:, 1].float().to(device)
        s = tree.sizes.float().to(device)
        boxes = torch.stack([
            torch.zeros(n, device=device),  # batch index
            c, r, c + s, r + s,
        ], dim=1)
        # image: (C, H, W) → (1, C, H, W) for roi_align.
        pooled = roi_align(
            image.unsqueeze(0), boxes,
            output_size=self.patch_fixed_size,
            aligned=True,
        )                                    # (n, C, P, P)
        flat = pooled.reshape(n, -1)         # (n, C * P²)
        return self.patch_encoder(flat)      # (n, d_hidden)

    def _walk_branch(self, features, sg):
        if sg.walks.shape[0] == 0:
            return torch.zeros_like(features)
        self.walk_layer.prepare(
            hodge_laplacian=sg.hodge_laplacian_0,
            primitive_curvatures=sg.walk_curvatures,
        )
        return self.walk_layer(
            features, sg.walks, sg.walk_signs, sg.M_v_walks,
        )

    def _poly_branch(self, features, sg):
        if sg.polygons.shape[0] == 0:
            return torch.zeros_like(features)
        self.poly_layer.prepare(
            hodge_laplacian=sg.hodge_laplacian_0,
            primitive_curvatures=sg.polygon_curvatures,
        )
        return self.poly_layer(
            features, sg.polygons, sg.polygon_signs, sg.M_v_polygons,
        )

    def _tri_branch(self, features, sg):
        if sg.triangles.shape[0] == 0:
            return torch.zeros_like(features)
        self.tri_layer.prepare(
            hodge_laplacian=sg.hodge_laplacian_0,
            primitive_curvatures=sg.triangle_curvatures,
        )
        return self.tri_layer(
            features, sg.triangles, sg.triangle_signs, sg.M_v_triangles,
        )

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
