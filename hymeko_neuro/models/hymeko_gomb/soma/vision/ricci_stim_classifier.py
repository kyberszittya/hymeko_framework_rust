"""End-to-end RicciStimClassifier — GömbSoma-Ricci-Stim phase 7
(refactored under Phase 9 backbone consolidation).

Thin wrapper around `RicciStimBackbone`: backbone produces
per-anchor features, the classifier head pools them and predicts a
class.

Backbone responsibility:
    image → AdaptiveQuadtree → encoder → StimulusGraph → 3 Bochner
    branches → sum → per-anchor features.

Head responsibility:
    features → global mean pool → Linear → logits.

The split lets `RicciStimClassifier` and `RicciStimDetector` share
~250 LOC of feature-extraction logic; the classifier and detector
files now hold only their respective head plus the per-image
forward dispatch.

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hymeko_neuro.models.hymeko_gomb.soma.vision.ricci_stim_backbone import (
    RicciStimBackbone,
)
from hymeko_neuro.models.hymeko_gomb.soma.vision.walk_conv_classifier import (
    Readout,
    _AttentionReadout,
)
from hymeko_neuro.experiments.vision.spatial_pyramid import SpatialPyramidPool


class _AnchorSpatialTreeReadout(nn.Module):
    """Dynamic spatial-tree pool over a *variable* set of anchors by position.

    The walk-conv spatial tree pools a fixed patch grid; RicciStim's adaptive
    quadtree gives a *variable* anchor set with 2-D positions, so we bin each
    anchor into the pyramid cells (levels 1×1, 2×2, 4×4) by its normalised
    centre, mean-pool features per cell (empty cells → 0), and (dynamic) gate
    each cell by a learned sigmoid of its activity. ``out_dim = 21·d``,
    independent of the anchor count — the readout flatten cannot provide here.

    # Preconditions ``centers`` in [0,1]² (row, col). # Postconditions
    ``forward(features (N,d), centers (N,2)) -> (out_dim,)``.
    """

    def __init__(self, d_hidden: int, levels: tuple[int, ...] = (1, 2, 4),
                 dynamic: bool = True) -> None:
        super().__init__()
        self.pool = SpatialPyramidPool(d_hidden, levels, dynamic)
        self.out_dim = self.pool.out_dim

    def forward(self, features: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        return self.pool(features, centers)


class RicciStimClassifier(nn.Module):
    """Image classifier on top of `RicciStimBackbone`.

    Parameters are passed straight through to the backbone. The
    classifier-specific args are ``n_classes`` and ``readout``.

    ``readout`` selects how the *variable* per-anchor features collapse to a
    vector: ``MEAN_POOL`` (default, the historical behaviour) or ``ATTENTION``
    (a content-weighted softmax pool — the only scale-free readout that handles
    the adaptive quadtree's variable anchor count; flatten cannot, since the
    anchor count differs per image). Reuses ``_AttentionReadout`` from the
    WalkConv classifier (no second implementation — §6.1).
    """

    def __init__(
        self,
        image_h: int = 28,
        image_w: int = 28,
        patch_size_initial: int = 4,
        patch_size_min: int = 1,
        in_channels: int = 1,
        d_hidden: int = 16,
        n_classes: int = 10,
        max_depth: int | None = 2,
        max_anchors: int = 256,
        score_threshold: float = 0.05,
        bochner_alpha: float = 0.0,
        bochner_beta: float = 0.0,
        use_sdrf: bool = False,
        sdrf_max_iters: int = 5,
        sdrf_kappa_target: float = -2.0,
        use_arity_mixer: bool = False,
        use_highway: bool = False,
        use_pyramid: bool = False,
        ablate_structural_branches: bool = False,
        readout: Readout = Readout.MEAN_POOL,
        cache_geometry: bool = False,
    ) -> None:
        super().__init__()
        if readout is Readout.FLATTEN:
            raise ValueError(
                "FLATTEN readout is unsupported on RicciStim: the adaptive "
                "quadtree yields a variable anchor count per image, so a fixed "
                "n_anchors·d head is ill-defined. Use MEAN_POOL or ATTENTION."
            )
        self.backbone = RicciStimBackbone(
            image_h=image_h, image_w=image_w,
            patch_size_initial=patch_size_initial,
            patch_size_min=patch_size_min,
            in_channels=in_channels,
            d_hidden=d_hidden,
            max_depth=max_depth, max_anchors=max_anchors,
            score_threshold=score_threshold,
            bochner_alpha=bochner_alpha, bochner_beta=bochner_beta,
            use_sdrf=use_sdrf,
            sdrf_max_iters=sdrf_max_iters,
            sdrf_kappa_target=sdrf_kappa_target,
            use_arity_mixer=use_arity_mixer,
            use_highway=use_highway,
            use_pyramid=use_pyramid,
            ablate_structural_branches=ablate_structural_branches,
            cache_geometry=cache_geometry,
        )
        self.n_classes = n_classes
        self.readout_kind = readout
        self.image_h, self.image_w = image_h, image_w
        # Readout over the variable anchor set. MEAN_POOL = h.mean(0) (no module);
        # ATTENTION = content pool (out d); SPATIAL_TREE(_STATIC) = quadtree pool
        # over anchor positions (out 21·d). _AttentionReadout ignores its
        # n_patches arg, so it handles the variable count directly.
        tree_modes = (Readout.SPATIAL_TREE, Readout.SPATIAL_TREE_STATIC)
        if readout is Readout.ATTENTION:
            self.readout: nn.Module | None = _AttentionReadout(0, d_hidden)
            out_dim = d_hidden
        elif readout in tree_modes:
            self.readout = _AnchorSpatialTreeReadout(
                d_hidden, dynamic=readout is Readout.SPATIAL_TREE)
            out_dim = self.readout.out_dim
        else:
            self.readout = None
            out_dim = d_hidden
        self.head = nn.Linear(out_dim, n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Forward over a batch of images.

        Parameters
        ----------
        images : Tensor[B, C, H, W] or Tensor[C, H, W]

        Returns
        -------
        logits : Tensor[B, n_classes] (or [n_classes] for unbatched)
        """
        if images.ndim == 3:
            return self._forward_single(images)
        if images.ndim != 4:
            raise ValueError(
                f"expected (B, C, H, W) or (C, H, W); got "
                f"shape {tuple(images.shape)}"
            )
        return torch.stack(
            [self._forward_single(images[b]) for b in range(images.shape[0])],
            dim=0,
        )

    def _forward_single(self, image: torch.Tensor) -> torch.Tensor:
        h, tree = self.backbone(image)             # (n_anchors, d_hidden)
        if isinstance(self.readout, _AnchorSpatialTreeReadout):
            pooled = self.readout(h, self._anchor_centers(tree, h.device))
        elif self.readout is not None:
            pooled = self.readout(h)
        else:
            pooled = h.mean(dim=0)
        return self.head(pooled)

    def _anchor_centers(self, tree, device: torch.device) -> torch.Tensor:
        """Anchor centres normalised to [0,1]² (row, col) for spatial binning."""
        pos = tree.positions.float().to(device)            # (N, 2) top-left (row, col)
        size = tree.sizes.float().to(device).unsqueeze(-1)  # (N, 1)
        centers = pos + size / 2.0
        scale = torch.tensor([self.image_h, self.image_w], device=device)
        return (centers / scale).clamp(0.0, 1.0)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
