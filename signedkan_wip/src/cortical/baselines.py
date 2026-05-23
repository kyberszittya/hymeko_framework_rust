"""Parameter-matched baseline models for the GömbSoma cortical
benchmark.

The 2026-05-16 plan named "parameter-matched ResNet" as the
canonical baseline. We ship :class:`ResNetTinyCortical` — a small
3-stage residual net whose per-stage outputs are mean-pooled into
the same retinotopic-bin shape :class:`CorticalFeatureExtractor`
emits for the GömbSoma backbone, so :class:`BrainScorer` compares
apples to apples.

Object-oriented commitment: the baseline is an ``nn.Module`` with
an ``extract_one`` / ``extract_batch`` interface mirroring
:class:`CorticalFeatureExtractor`, so the scorer doesn't care which
model produced the features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import BinningConfig, PerDepthFeatures


class _ResBlock(nn.Module):
    """3×3 → 3×3 residual block, no batchnorm."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.c1(x))
        out = self.c2(out)
        return F.relu(out + x)


class ResNetTinyCortical(nn.Module):
    """3-stage ResNet-tiny exposing per-depth retinotopic features.

    Three stages, each a stride-2 downsample + one residual block:

      input C×H×W →
      stage_0 (d_hidden, H/2, W/2)  ←  pooled to depth-0 binning
      stage_1 (d_hidden, H/4, W/4)  ←  pooled to depth-1 binning
      stage_2 (d_hidden, H/8, W/8)  ←  pooled to depth-2 binning

    Each stage's output is mean-pooled into the ``BinningConfig``-
    specified ``(n_h, n_w)`` retinotopic grid and concatenated into
    the same flat vector :class:`CorticalFeatureExtractor` emits, so
    the scoring pipeline is model-agnostic.

    Parameter count is calibrated to match GömbSoma's
    ``RicciStimBackbone`` at default config (~30 k params for
    ``d_hidden=16``). See :func:`assert_param_match`.
    """

    def __init__(
        self,
        image_h: int,
        image_w: int,
        in_channels: int = 1,
        d_hidden: int = 16,
        binning_config: BinningConfig | None = None,
    ) -> None:
        super().__init__()
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        self.in_channels = int(in_channels)
        self.d_hidden = int(d_hidden)
        self.binning_config = binning_config or BinningConfig()

        # Stage 0: in_channels → d_hidden (stride 2 downsample).
        self.stem = nn.Conv2d(
            in_channels, d_hidden, 3, stride=2, padding=1
        )
        self.block0 = _ResBlock(d_hidden)
        # Stage 1: d_hidden → d_hidden (stride 2).
        self.down1 = nn.Conv2d(d_hidden, d_hidden, 3, stride=2, padding=1)
        self.block1 = _ResBlock(d_hidden)
        # Stage 2: d_hidden → d_hidden (stride 2).
        self.down2 = nn.Conv2d(d_hidden, d_hidden, 3, stride=2, padding=1)
        self.block2 = _ResBlock(d_hidden)

    def _forward_features(self, image: torch.Tensor) -> dict[int, torch.Tensor]:
        """Per-stage feature maps keyed by depth (0, 1, 2)."""
        x = image.unsqueeze(0)  # add batch
        x = F.relu(self.stem(x))
        x0 = self.block0(x)   # [1, d, H/2, W/2]
        x1 = self.block1(F.relu(self.down1(x0)))  # [1, d, H/4, W/4]
        x2 = self.block2(F.relu(self.down2(x1)))  # [1, d, H/8, W/8]
        return {0: x0.squeeze(0), 1: x1.squeeze(0), 2: x2.squeeze(0)}

    def extract_one(self, image: torch.Tensor) -> PerDepthFeatures:
        """Per-image feature extraction matching the GömbSoma interface."""
        if image.ndim != 3:
            raise ValueError(
                f"expected (C, H, W); got shape {tuple(image.shape)}"
            )
        feat_maps = self._forward_features(image)
        per_depth: dict[int, torch.Tensor] = {}
        for depth in self.binning_config.depths:
            if depth not in feat_maps:
                # Depth requested by the binning config but not
                # emitted by the network — fill with zeros.
                n_h, n_w = self.binning_config.bins_per_depth[depth]
                per_depth[depth] = image.new_zeros((n_h * n_w, self.d_hidden))
                continue
            fmap = feat_maps[depth]  # [d, h, w]
            n_h, n_w = self.binning_config.bins_per_depth[depth]
            # Adaptive-average-pool to (n_h, n_w), then permute to
            # (n_h*n_w, d) to match the GömbSoma feature layout.
            pooled = F.adaptive_avg_pool2d(
                fmap.unsqueeze(0), (n_h, n_w)
            ).squeeze(0)  # [d, n_h, n_w]
            pooled_flat = pooled.permute(1, 2, 0).reshape(
                n_h * n_w, self.d_hidden
            )
            per_depth[depth] = pooled_flat
        flat = torch.cat([per_depth[d].flatten() for d in sorted(per_depth)])
        return PerDepthFeatures(per_depth=per_depth, flat=flat)

    def extract_batch(self, images: torch.Tensor) -> torch.Tensor:
        """``[N, C, H, W]`` → ``[N, total_d]``."""
        if images.ndim != 4:
            raise ValueError(
                f"expected (N, C, H, W); got shape {tuple(images.shape)}"
            )
        rows = []
        for i in range(images.shape[0]):
            rows.append(self.extract_one(images[i]).flat)
        return torch.stack(rows, dim=0)

    @property
    def total_d(self) -> int:
        return self.d_hidden * sum(
            self.binning_config.n_bins(d) for d in self.binning_config.depths
        )


def count_parameters(model: nn.Module) -> int:
    """Convenience: count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_param_match(
    model_a: nn.Module, model_b: nn.Module, factor: float = 1.5,
) -> None:
    """Assert ``model_a`` and ``model_b`` parameter counts are within ``factor``.

    Raises ``AssertionError`` if not. Used by the benchmark pipeline
    to enforce the "parameter-matched baseline" contract from the
    cortical-benchmark plan.
    """
    pa = count_parameters(model_a)
    pb = count_parameters(model_b)
    if pa == 0 or pb == 0:
        raise AssertionError(
            f"one model has zero parameters: a={pa}, b={pb}"
        )
    ratio = max(pa, pb) / min(pa, pb)
    if ratio > factor:
        raise AssertionError(
            f"param counts not within factor {factor}: "
            f"a={pa:,} b={pb:,} (ratio {ratio:.2f})"
        )


__all__ = [
    "ResNetTinyCortical",
    "count_parameters",
    "assert_param_match",
]
