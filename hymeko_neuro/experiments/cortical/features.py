"""Per-image, per-depth, retinotopically-binned feature extraction
for the GömbSoma cortical benchmark.

Wraps a backbone (default: :class:`RicciStimBackbone`) and emits the
:class:`PerDepthFeatures` dataclass expected by
:class:`BrainScorer`. The retinotopic binning mimics V1→V2→V4
hierarchy: depth-0 (coarsest) gets few large bins (V4-like), depth-2
(finest) gets many small bins (V1-like).

Object-oriented commitment: ``CorticalFeatureExtractor`` is an
``nn.Module`` so callers compose it like any model; the
``BinningConfig`` dataclass holds the retinotopic-grid choice as
an explicit, inspectable object rather than a string-typed config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.nn as nn


# ─── Retinotopic binning configuration ──────────────────────────────


@dataclass(frozen=True)
class BinningConfig:
    """Per-depth retinotopic-grid resolution.

    ``bins_per_depth[d] = (n_h_bins, n_w_bins)`` gives the grid
    GömbSoma anchors at depth ``d`` are pooled into. The default
    cortical mapping mirrors V1→V2→V4 receptive-field density:

    - depth-0 → V4-like (2 × 2 bins, coarsest)
    - depth-1 → V2-like (4 × 4 bins)
    - depth-2 → V1-like (8 × 8 bins, finest)
    """

    bins_per_depth: dict[int, tuple[int, int]] = field(
        default_factory=lambda: {0: (2, 2), 1: (4, 4), 2: (8, 8)}
    )

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(sorted(self.bins_per_depth.keys()))

    def n_bins(self, depth: int) -> int:
        h, w = self.bins_per_depth[depth]
        return h * w


@dataclass(frozen=True)
class PerDepthFeatures:
    """Per-image features grouped by depth.

    Attributes
    ----------
    per_depth : dict[int, Tensor]
        ``depth → Tensor[n_bins_h * n_bins_w, d_hidden]`` for one image.
    flat : Tensor
        Concatenation of all depths' features into a single
        ``[total_d]`` vector. Convenience for downstream scoring.
    """

    per_depth: dict[int, torch.Tensor]
    flat: torch.Tensor

    @property
    def depths(self) -> tuple[int, ...]:
        return tuple(sorted(self.per_depth.keys()))

    @property
    def total_d(self) -> int:
        return int(self.flat.shape[0])


# ─── Feature extractor ──────────────────────────────────────────────


class CorticalFeatureExtractor(nn.Module):
    """Wrap a backbone to emit per-depth, retinotopically-binned features.

    The backbone is expected to follow the GömbSoma
    :class:`RicciStimBackbone` contract: ``forward(image[C,H,W]) ->
    (features[n_anchors, d_hidden], tree)`` where ``tree`` exposes
    ``positions[n_anchors, 2]`` and ``scales[n_anchors]``.

    For per-image extraction:

    1. ``features, tree = backbone(image)``
    2. For each depth ``d`` in :attr:`binning_config.depths`:
       a. Select anchors with ``tree.scales == d``.
       b. Map each anchor's pixel position to a retinotopic bin.
       c. Mean-pool features within each bin.
       d. Flatten to ``Tensor[n_bins_h * n_bins_w, d_hidden]``.
    3. Concatenate to a single flat vector for downstream scoring.

    Empty bins (no anchor at that location) get zero — they're not
    NaN'd, so the scoring pipeline doesn't need NaN-aware regression.
    """

    def __init__(
        self,
        backbone: nn.Module,
        image_h: int,
        image_w: int,
        d_hidden: int,
        binning_config: BinningConfig | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.d_hidden = int(d_hidden)
        self.binning_config = binning_config or BinningConfig()

    def extract_one(self, image: torch.Tensor) -> PerDepthFeatures:
        """Per-image feature extraction.

        Parameters
        ----------
        image
            ``Tensor[C, H, W]`` — single image.

        Returns
        -------
        PerDepthFeatures
            Per-depth pooled features + flattened concatenation.
        """
        if image.ndim != 3:
            raise ValueError(
                f"expected (C, H, W); got shape {tuple(image.shape)}"
            )
        features, tree = self.backbone(image)
        # tree must expose .positions and .scales (Tensor[n_anchors, 2]
        # and Tensor[n_anchors] respectively).
        positions = tree.positions
        scales = tree.scales

        per_depth: dict[int, torch.Tensor] = {}
        for depth in self.binning_config.depths:
            n_h, n_w = self.binning_config.bins_per_depth[depth]
            mask = scales == depth
            depth_features = features[mask] if mask.any() else features.new_zeros(
                (0, self.d_hidden)
            )
            depth_positions = positions[mask] if mask.any() else positions.new_zeros(
                (0, 2)
            )
            binned = self._bin_features(
                depth_features, depth_positions, n_h, n_w
            )
            per_depth[depth] = binned

        # Flatten all depths into a single vector.
        flat = torch.cat([per_depth[d].flatten() for d in sorted(per_depth)])
        return PerDepthFeatures(per_depth=per_depth, flat=flat)

    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Per-batch convenience: stacks :meth:`extract_one`'s flat
        vectors into a single ``[N, total_d]`` tensor for scoring.

        Parameters
        ----------
        images
            ``Tensor[N, C, H, W]``.

        Returns
        -------
        Tensor[N, total_d]
            One row per image, suitable for direct PLS / Ridge input.
        """
        if images.ndim != 4:
            raise ValueError(
                f"expected (N, C, H, W); got shape {tuple(images.shape)}"
            )
        rows = []
        for i in range(images.shape[0]):
            pdf = self.extract_one(images[i])
            rows.append(pdf.flat)
        return torch.stack(rows, dim=0)

    @property
    def total_d(self) -> int:
        return self.d_hidden * sum(
            self.binning_config.n_bins(d) for d in self.binning_config.depths
        )

    # ─── Internals ───────────────────────────────────────────

    def _bin_features(
        self,
        features: torch.Tensor,    # [m, d]
        positions: torch.Tensor,   # [m, 2] = (row, col) pixel coords
        n_h: int,
        n_w: int,
    ) -> torch.Tensor:
        """Mean-pool features into an ``n_h × n_w`` retinotopic grid.

        Returns ``Tensor[n_h * n_w, d_hidden]``. Empty bins are zero.

        Vectorised via ``scatter_add_`` — no Python-level loop over
        anchors (CLAUDE §5 data-oriented design).
        """
        d = self.d_hidden
        n_bins = n_h * n_w
        out = features.new_zeros((n_bins, d))
        if features.shape[0] == 0:
            return out
        bin_h = (positions[:, 0].float() / self.image_h * n_h).long().clamp(
            0, n_h - 1
        )
        bin_w = (positions[:, 1].float() / self.image_w * n_w).long().clamp(
            0, n_w - 1
        )
        flat_idx = bin_h * n_w + bin_w               # [m]
        idx_expand = flat_idx.unsqueeze(-1).expand_as(features)  # [m, d]
        out.scatter_add_(0, idx_expand, features)
        counts = features.new_zeros((n_bins,))
        ones = features.new_ones((features.shape[0],))
        counts.scatter_add_(0, flat_idx, ones)
        counts = counts.clamp_min(1.0)
        return out / counts.unsqueeze(-1)


__all__ = [
    "BinningConfig",
    "PerDepthFeatures",
    "CorticalFeatureExtractor",
]
