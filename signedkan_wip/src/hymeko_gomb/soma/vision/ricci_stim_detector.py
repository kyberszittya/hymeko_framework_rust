"""End-to-end RicciStimDetector — GömbSoma-Ricci-Stim phase 8
(refactored under Phase 9 backbone consolidation).

Thin wrapper around `RicciStimBackbone` + two per-anchor heads
(class logits + bbox offsets). The backbone is shared with
`RicciStimClassifier`; the detector differs only in the heads and
the per-image return format (DetectionOutput vs pooled logits).

Plan: docs/plans/2026-05-14-gomb-soma-ricci-stim/.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_backbone import (
    RicciStimBackbone,
)


@dataclass
class DetectionOutput:
    """Per-image detector output."""

    cls_logits: torch.Tensor
    bbox_offsets: torch.Tensor
    anchor_positions: torch.Tensor
    anchor_sizes: torch.Tensor

    @property
    def n_anchors(self) -> int:
        return int(self.cls_logits.shape[0])


class RicciStimDetector(nn.Module):
    """Per-anchor object detector on top of `RicciStimBackbone`."""

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
    ) -> None:
        super().__init__()
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
        )
        self.n_classes = n_classes
        # +1 for background class.
        self.cls_head = nn.Linear(d_hidden, n_classes + 1)
        self.bbox_head = nn.Linear(d_hidden, 4)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self, images: torch.Tensor,
    ) -> DetectionOutput | list[DetectionOutput]:
        if images.ndim == 3:
            return self._forward_single(images)
        if images.ndim != 4:
            raise ValueError(
                f"expected (B, C, H, W) or (C, H, W); got "
                f"shape {tuple(images.shape)}"
            )
        return [self._forward_single(images[b]) for b in range(images.shape[0])]

    def _forward_single(self, image: torch.Tensor) -> DetectionOutput:
        h, tree = self.backbone(image)
        cls_logits = self.cls_head(h)
        bbox_offsets = self.bbox_head(h)
        return DetectionOutput(
            cls_logits=cls_logits,
            bbox_offsets=bbox_offsets,
            anchor_positions=tree.positions,
            anchor_sizes=tree.sizes,
        )

    # -----------------------------------------------------------------
    # Decode utility
    # -----------------------------------------------------------------

    @staticmethod
    def decode_boxes(
        out: DetectionOutput,
    ) -> torch.Tensor:
        """Decode (dx, dy, dw, dh) offsets against anchor (r, c, s) into
        absolute (cx, cy, w, h)."""
        positions = out.anchor_positions.float()
        sizes = out.anchor_sizes.float().unsqueeze(-1)
        cx = positions[:, 0:1] + 0.5 * sizes
        cy = positions[:, 1:2] + 0.5 * sizes
        dx = out.bbox_offsets[:, 0:1]
        dy = out.bbox_offsets[:, 1:2]
        dw = out.bbox_offsets[:, 2:3]
        dh = out.bbox_offsets[:, 3:4]
        pred_cx = cx + dx * sizes
        pred_cy = cy + dy * sizes
        pred_w = sizes * torch.exp(dw)
        pred_h = sizes * torch.exp(dh)
        return torch.cat([pred_cx, pred_cy, pred_w, pred_h], dim=-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
