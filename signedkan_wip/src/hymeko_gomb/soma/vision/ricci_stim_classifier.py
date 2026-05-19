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

from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_backbone import (
    RicciStimBackbone,
)


class RicciStimClassifier(nn.Module):
    """Image classifier on top of `RicciStimBackbone`.

    Parameters are passed straight through to the backbone; the only
    classifier-specific arg is ``n_classes``.
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
        self.head = nn.Linear(d_hidden, n_classes)

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
        h, _tree = self.backbone(image)
        return self.head(h.mean(dim=0))

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
