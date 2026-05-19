"""WalkConvImageClassifier — minimal GömbSoma vision pipeline.

Architecture
------------
    image (C, H, W)
       │
       │   PatchGraphBuilder.encode  →  (patches, walks, walk_signs, M_v)
       ▼
    patch_embed: Linear(patch_dim → d_hidden)
       │
       │   WalkConvLayer(d_hidden, d_hidden, k_arity=3)
       ▼
    walk-convolved vertex features
       │
       │   global mean pool over patches
       ▼
    Linear(d_hidden → n_classes)
       │
       ▼
    logits

This is the simplest end-to-end GömbSoma vision model: one walk-conv
layer, no polygons, no triangles, no Clifford-FIR (those come in
phases 3-5). The point is to validate that the walk-only sensorimotor
hypothesis even runs end-to-end and produces something non-trivial
on a vision task.

If walk-only already gives a usable signal on MNIST, that's evidence
the compositional hierarchy is on the right track. If it fails, we
have a clear baseline against which polygons / triangles must lift.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from signedkan_wip.src.hymeko_gomb.soma.hg_conv import HypergraphConvConfig
from signedkan_wip.src.hymeko_gomb.soma.walk_layer import WalkConvLayer
from signedkan_wip.src.hymeko_gomb.soma.vision.patch_graph import (
    PatchGraphBuilder,
)


class WalkConvImageClassifier(nn.Module):
    """Image classifier with one WalkConv layer over a patch graph.

    Parameters
    ----------
    image_h, image_w : int
        Image dimensions.
    patch_size : int
        Patch side. ``image_h`` and ``image_w`` must be divisible by it.
    in_channels : int
        Image channels (1 for MNIST, 3 for CIFAR / natural).
    d_hidden : int
        Hidden width for the walk-conv layer.
    n_classes : int
        Output class count.
    use_sign_branching : bool
        Whether WalkConv routes positive vs negative walks through
        independent banks. Default True.
    """

    def __init__(
        self,
        image_h: int,
        image_w: int,
        patch_size: int,
        in_channels: int,
        d_hidden: int,
        n_classes: int,
        use_sign_branching: bool = True,
    ) -> None:
        super().__init__()
        self.builder = PatchGraphBuilder(image_h, image_w, patch_size)
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(self.patch_dim, d_hidden)
        self.walk_conv = WalkConvLayer(
            HypergraphConvConfig(
                in_features=d_hidden,
                out_features=d_hidden,
                k_arity=3,
                use_sign_branching=use_sign_branching,
            )
        )
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
            return self._forward_single(images).unsqueeze(0).squeeze(0)
        if images.ndim != 4:
            raise ValueError(
                f"expected (B, C, H, W) or (C, H, W), got "
                f"shape {tuple(images.shape)}"
            )
        # Per-image encoding; the patch-graph topology is the same
        # for the whole batch but signs and features differ.
        logits = []
        for b in range(images.shape[0]):
            logits.append(self._forward_single(images[b]))
        return torch.stack(logits, dim=0)

    def _forward_single(self, image: torch.Tensor) -> torch.Tensor:
        patches, walks, walk_signs, M_v = self.builder.encode(image)
        x = self.patch_embed(patches)            # (n_patches, d_hidden)
        x = self.walk_conv(x, walks, walk_signs, M_v)
        # Global mean pool over patches.
        pooled = x.mean(dim=0)
        return self.head(pooled)                  # (n_classes,)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"WalkConvImageClassifier("
            f"image=({self.builder.image_h}, {self.builder.image_w}), "
            f"patch_size={self.builder.patch_size}, "
            f"n_patches={self.builder.n_patches}, "
            f"n_walks={self.builder.walks.shape[0]}, "
            f"d_hidden={self.patch_embed.out_features}, "
            f"n_classes={self.head.out_features}, "
            f"n_params={self.n_parameters()})"
        )
