"""Rust-backed drop-in replacement for ``AdaptiveQuadtree``.

The state machine (depth-by-depth subdivision, threshold check, 4-conn
adjacency, Forman κ, budget cap) runs in
`hymeko_py.build_quadtree_rs`; variance scoring stays here on the
GPU via a single ``torchvision.ops.roi_align`` per depth (the same
call the Python reference uses).

Drop-in: the constructor signature, the ``forward`` signature, and
the returned ``AnchorTree`` are identical to the Python reference at
``quadtree.py``.

Plan: ``docs/plans/2026-05-16-gomb-soma-quadtree-triton/``.
Report: ``reports/2026-05-16-gomb-soma-quadtree-rust.md``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import hymeko as _hymeko_native
    _HYMEKO_AVAILABLE = hasattr(_hymeko_native, "build_quadtree_rs")
except ImportError:
    _hymeko_native = None
    _HYMEKO_AVAILABLE = False

from signedkan_wip.src.hymeko_gomb.soma.vision.quadtree import AnchorTree


class AdaptiveQuadtreeRust(nn.Module):
    """Rust-state-machine + GPU-variance variant of
    :class:`AdaptiveQuadtree`.

    See ``AdaptiveQuadtree`` in
    ``signedkan_wip/src/hymeko_gomb/soma/vision/quadtree.py`` for the
    algorithm. This class wraps the Rust port
    (``hymeko.build_quadtree_rs``) so the per-anchor ``.item()`` CUDA
    syncs and Python-side edge-building loop are eliminated.

    Preconditions / postconditions
    ------------------------------
    Identical to ``AdaptiveQuadtree``. The output ``AnchorTree`` is
    set-equal (up to row ordering of the four children per parent) to
    the Python reference for the same inputs.

    Build dependency
    ----------------
    Requires the ``hymeko`` Python module (built via ``maturin
    develop`` from ``hymeko_py/``). If absent, construction raises
    ``RuntimeError`` early — the failure must not be silent at
    forward time.
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
        if not _HYMEKO_AVAILABLE:
            raise RuntimeError(
                "AdaptiveQuadtreeRust requires the `hymeko` Python "
                "extension built via `maturin develop --release "
                "--manifest-path hymeko_py/Cargo.toml`. Re-build and "
                "import again. (Detected hymeko module is missing "
                "build_quadtree_rs symbol — likely stale binary.)"
            )
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
                f"image size ({image_h}, {image_w}) must be divisible "
                f"by patch_size_initial ({patch_size_initial})"
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
            max_depth = int(math.log2(patch_size_initial // patch_size_min))
        self.max_depth = max_depth
        self.max_anchors = max_anchors
        self.variance_weight = float(variance_weight)
        self.curvature_weight = float(curvature_weight)
        self.score_threshold = float(score_threshold)

    def forward(self, image: torch.Tensor) -> AnchorTree:
        if image.ndim != 3:
            raise ValueError(
                f"expected (C, H, W) image, got shape {tuple(image.shape)}"
            )
        _c, h, w = image.shape
        if h != self.image_h or w != self.image_w:
            raise ValueError(
                f"image shape ({h}, {w}) doesn't match builder "
                f"({self.image_h}, {self.image_w})"
            )

        device = image.device
        # Pre-compute image and its square once; reused across depths
        # for the roi_align variance score.
        image_sq = image * image
        # roi_align needs (B, C, H, W); we have one image.
        image_batched = image.unsqueeze(0)
        image_sq_batched = image_sq.unsqueeze(0)

        # Local imports to avoid hauling torchvision unless this class
        # is actually used.
        from torchvision.ops import roi_align

        variance_weight = self.variance_weight
        curvature_weight = self.curvature_weight

        def score_callback(positions_list, sizes_list):
            """Called by Rust once per depth. Returns the per-anchor
            variance (std-dev style) for the supplied frontier."""
            n = len(positions_list)
            if n == 0:
                return []
            if variance_weight == 0.0:
                # Skip the GPU pass entirely if variance is unweighted.
                return [0.0] * n
            rs = torch.tensor(
                [p[0] for p in positions_list],
                dtype=torch.float32, device=device,
            )
            cs = torch.tensor(
                [p[1] for p in positions_list],
                dtype=torch.float32, device=device,
            )
            ss = torch.tensor(
                sizes_list, dtype=torch.float32, device=device,
            )
            boxes = torch.stack([
                torch.zeros_like(rs),
                cs, rs, cs + ss, rs + ss,
            ], dim=1)
            mean_pooled = roi_align(
                image_batched, boxes, output_size=1,
            ).reshape(n, -1).mean(dim=-1)
            mean_sq_pooled = roi_align(
                image_sq_batched, boxes, output_size=1,
            ).reshape(n, -1).mean(dim=-1)
            variance = (mean_sq_pooled - mean_pooled ** 2).clamp(min=0)
            std = variance.sqrt()
            # One CUDA → CPU sync per depth; cheap (< 100 floats).
            return std.detach().cpu().tolist()

        positions_np, sizes_np, scales_np, parents_np = (
            _hymeko_native.build_quadtree_rs(
                self.image_h,
                self.image_w,
                self.patch_size_initial,
                self.patch_size_min,
                self.max_depth,
                self.max_anchors,
                variance_weight,
                curvature_weight,
                self.score_threshold,
                score_callback,
            )
        )

        # Move into torch tensors on the requested device. AnchorTree
        # callers expect long tensors.
        return AnchorTree(
            positions=torch.from_numpy(positions_np).to(
                dtype=torch.long, device=device,
            ),
            sizes=torch.from_numpy(sizes_np).to(
                dtype=torch.long, device=device,
            ),
            scales=torch.from_numpy(scales_np).to(
                dtype=torch.long, device=device,
            ),
            parent_indices=torch.from_numpy(parents_np).to(
                dtype=torch.long, device=device,
            ),
        )
