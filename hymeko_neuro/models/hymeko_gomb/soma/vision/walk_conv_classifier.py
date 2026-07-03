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

import enum

import torch
import torch.nn as nn

from hymeko_neuro.core import ChebyshevCRActivation

from hymeko_neuro.models.hymeko_gomb.soma.hg_conv import (
    Aggregation,
    HypergraphConvConfig,
    MessageActivation,
)
from hymeko_neuro.models.hymeko_gomb.soma.walk_layer import WalkConvLayer
from hymeko_neuro.models.hymeko_gomb.soma.vision.patch_graph import (
    PatchGraphBuilder,
)
from hymeko_neuro.experiments.vision.spatial_pyramid import (
    SpatialPyramidPool,
    grid_positions,
)


class PatchEncoder(enum.Enum):
    """How raw patches are embedded. ``LINEAR`` (default) is the historical bare
    projection; ``CHEBY_CR`` appends a Chebyshev-CR HSiKAN cell (the user's
    "Chebyshev-CR patches")."""

    LINEAR = "linear"
    CHEBY_CR = "cheby_cr"


class Readout(enum.Enum):
    """How walk-convolved patch features collapse to a vector for the head.

    ``MEAN_POOL`` (default) averages over patches — permutation-invariant and
    content-blind (the suspected H2 bottleneck). ``FLATTEN`` concatenates every
    patch's features — keeps *absolute* position, but the head is
    ``n_patches·d`` (does not scale) and absolute position is the class signal
    only on *centred* inputs. ``ATTENTION`` is the scalable position-aware pool:
    a learned per-patch score → softmax → weighted sum, ``out_dim == d``
    (scale-free). It attends to *where the content is* (informative on
    random-position inputs), rather than encoding absolute position."""

    MEAN_POOL = "mean_pool"
    FLATTEN = "flatten"
    ATTENTION = "attention"
    POS_ATTENTION = "pos_attention"
    SPATIAL_TREE = "spatial_tree"
    SPATIAL_TREE_STATIC = "spatial_tree_static"  # same pyramid, no learned gate
    MULTI_QUERY = "multi_query"                  # K learned queries, K·d pool


class _MeanPoolReadout(nn.Module):
    """Mean over the patch axis. out_dim == d_hidden. Position-blind."""

    def __init__(self, n_patches: int, d_hidden: int) -> None:
        super().__init__()
        self.out_dim = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-2)                       # over patches; batch-safe


class _FlattenReadout(nn.Module):
    """Concatenate all patch features. out_dim == n_patches * d_hidden.
    Preserves per-patch position (permuting patches changes the output)."""

    def __init__(self, n_patches: int, d_hidden: int) -> None:
        super().__init__()
        self.out_dim = n_patches * d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-2], -1)        # flatten patches; batch-safe


class _AttentionReadout(nn.Module):
    """Learned attention pool: per-patch score → softmax over patches → weighted
    sum. ``out_dim == d_hidden`` (scale-free, unlike flatten). Content-weighted
    and permutation-equivariant: it attends to *where the informative content is*
    regardless of absolute position — the scalable position-aware readout.

    # Postconditions output is a convex combination of patch features
    (attention weights are a softmax: non-negative, sum to 1)."""

    def __init__(self, n_patches: int, d_hidden: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_hidden, 1)
        self.out_dim = d_hidden

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax attention weights over patches, (..., n_patches)."""
        return torch.softmax(self.score(x).squeeze(-1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attention_weights(x)              # (..., n_patches)
        return (w.unsqueeze(-1) * x).sum(dim=-2)   # (..., d_hidden); batch-safe


class _PosAttentionReadout(nn.Module):
    """Attention pool with a learned per-patch positional embedding.

    Features are augmented with position (``x + pos``) before *both* scoring and
    the weighted sum, so the pooled vector encodes *what + where* — the spatial
    layout that pure content-attention discards (Phase 1 found content-attention
    < flatten by exactly that gap). ``out_dim == d_hidden`` (O(d) head); the
    positional table is ``n_patches·d`` params, far smaller than flatten's
    ``n_patches·d·n_classes`` head. Position-aware → permutation-*sensitive*."""

    def __init__(self, n_patches: int, d_hidden: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(n_patches, d_hidden) * 0.02)
        self.score = nn.Linear(d_hidden, 1)
        self.out_dim = d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = x + self.pos                              # inject position (broadcasts over batch)
        w = torch.softmax(self.score(xp).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1) * xp).sum(dim=-2)      # (..., d_hidden); batch-safe


class _SpatialTreeReadout(nn.Module):
    """Dynamic spatial-tree (quadtree-pyramid) pool.

    Pools the patch-feature map over a quadtree of regions — levels ``(1, 2,
    4)`` give a 1×1, 2×2, 4×4 grid of cells (21 cells total). Each cell
    mean-pools its region (multi-scale *position* is preserved), and — the
    *dynamic* part — is gated by a learned sigmoid of its own activity, so the
    tree emphasises cells where the content is. ``out_dim = (Σ levelᵢ²)·d``,
    *independent of the patch-grid size* — so unlike flatten it scales, and
    unlike mean-pool it keeps coarse-to-fine spatial layout.

    # Preconditions ``h_patches, w_patches >= max(levels)``.
    # Postconditions ``forward((n_patches, d)) -> (out_dim,)``.
    """

    def __init__(self, h_patches: int, w_patches: int, d_hidden: int,
                 levels: tuple[int, ...] = (1, 2, 4), dynamic: bool = True) -> None:
        super().__init__()
        self.pool = SpatialPyramidPool(d_hidden, levels, dynamic)
        # Fixed patch grid → precompute the pooling matrix once; forward is then a
        # single (batchable) fused matmul, no per-call binning.
        self.pool.set_fixed_positions(grid_positions(h_patches, w_patches))
        self.out_dim = self.pool.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)                                      # (..., out_dim)


class _MultiQueryReadout(nn.Module):
    """Multi-query attention pool: ``n_queries`` learned query vectors each
    attend (softmax) over the patches → ``n_queries`` pooled vectors, concatenated.
    ``out_dim = n_queries·d`` — independent of the patch count (scale-free, like
    the spatial tree) but content-based rather than grid-binned. K=1 recovers the
    single-query attention pool; K>1 gives multiple content "slots".

    # Postconditions ``forward((..., N, d)) -> (..., n_queries·d)``; each slot is
    a softmax-convex combination of patch features. Batch-safe."""

    def __init__(self, n_patches: int, d_hidden: int, n_queries: int = 8) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d_hidden) * 0.1)
        self.out_dim = n_queries * d_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("kd,...nd->...kn", self.queries, x)   # (..., K, N)
        w = torch.softmax(scores, dim=-1)                           # over patches
        pooled = torch.einsum("...kn,...nd->...kd", w, x)           # (..., K, d)
        return pooled.reshape(*pooled.shape[:-2], -1)               # (..., K·d)


def _build_patch_encoder(kind: PatchEncoder, patch_dim: int, d_hidden: int) -> nn.Module:
    """Linear patch embed, optionally followed by a Chebyshev-CR HSiKAN cell."""
    embed = nn.Linear(patch_dim, d_hidden)
    if kind is PatchEncoder.LINEAR:
        return embed
    return nn.Sequential(embed, ChebyshevCRActivation(d_hidden))


def _build_readout(kind: Readout, h_patches: int, w_patches: int,
                   d_hidden: int) -> nn.Module:
    n_patches = h_patches * w_patches
    if kind is Readout.MEAN_POOL:
        return _MeanPoolReadout(n_patches, d_hidden)
    if kind is Readout.FLATTEN:
        return _FlattenReadout(n_patches, d_hidden)
    if kind is Readout.ATTENTION:
        return _AttentionReadout(n_patches, d_hidden)
    if kind is Readout.POS_ATTENTION:
        return _PosAttentionReadout(n_patches, d_hidden)
    if kind is Readout.MULTI_QUERY:
        return _MultiQueryReadout(n_patches, d_hidden)
    return _SpatialTreeReadout(h_patches, w_patches, d_hidden,
                               dynamic=kind is Readout.SPATIAL_TREE)


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
    aggregation : Aggregation
        How walk messages pool back to patches. ``SUM`` (default) is the
        historical sign-blind pool — pair with ``use_sign_branching=True`` for
        the sign-as-routing base-Soma. ``HOLONOMY`` weights each walk by its
        σ-product (sign-as-connection) — pair with ``use_sign_branching=False``
        so sign enters only as transport. See ``Aggregation``.
    message_activation : MessageActivation
        Walk-message nonlinearity (``GELU`` default; ``CR`` / ``CHEBY_CR`` for
        the HSiKAN cell). See ``MessageActivation``.
    patch_encoder : PatchEncoder
        Patch embedding (``LINEAR`` default; ``CHEBY_CR`` for Chebyshev-CR patches).
    readout : Readout
        Patch→vector collapse (``MEAN_POOL`` default; ``FLATTEN`` preserves position).
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
        aggregation: Aggregation = Aggregation.SUM,
        message_activation: MessageActivation = MessageActivation.GELU,
        patch_encoder: PatchEncoder = PatchEncoder.LINEAR,
        readout: Readout = Readout.MEAN_POOL,
    ) -> None:
        super().__init__()
        self.builder = PatchGraphBuilder(image_h, image_w, patch_size)
        self.patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = _build_patch_encoder(
            patch_encoder, self.patch_dim, d_hidden,
        )
        self.walk_conv = WalkConvLayer(
            HypergraphConvConfig(
                in_features=d_hidden,
                out_features=d_hidden,
                k_arity=3,
                use_sign_branching=use_sign_branching,
                aggregation=aggregation,
                message_activation=message_activation,
            )
        )
        self.readout = _build_readout(
            readout, self.builder.h_patches, self.builder.w_patches, d_hidden)
        self.head = nn.Linear(self.readout.out_dim, n_classes)

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
        return self._forward_batched(images)

    def _forward_single(self, image: torch.Tensor) -> torch.Tensor:
        patches, walks, walk_signs, M_v = self.builder.encode(image)
        x = self.patch_embed(patches)            # (n_patches, d_hidden)
        x = self.walk_conv(x, walks, walk_signs, M_v)
        pooled = self.readout(x)                 # (d_hidden,) or (n_patches*d_hidden,)
        return self.head(pooled)                 # (n_classes,)

    def _forward_batched(self, images: torch.Tensor) -> torch.Tensor:
        """Whole batch in one pass: the grid topology (walks, M_v) is shared, so
        every stage runs on ``(B, n_patches, d)`` — no per-image Python loop."""
        device = images.device
        patches = self.builder.patchify_batch(images)            # (B, N, patch_dim)
        e_signs = self.builder.edge_signs(patches)               # (B, n_edges)
        w_signs = self.builder.walk_signs(e_signs)               # (B, n_walks)
        x = self.patch_embed(patches)                            # (B, N, d)
        x = self.walk_conv(x, self.builder.walks.to(device),
                           w_signs, self.builder.M_v.to(device))  # (B, N, d)
        pooled = self.readout(x)                                 # (B, out_dim)
        return self.head(pooled)                                 # (B, n_classes)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"WalkConvImageClassifier("
            f"image=({self.builder.image_h}, {self.builder.image_w}), "
            f"patch_size={self.builder.patch_size}, "
            f"n_patches={self.builder.n_patches}, "
            f"n_walks={self.builder.walks.shape[0]}, "
            f"readout_dim={self.readout.out_dim}, "
            f"n_classes={self.head.out_features}, "
            f"n_params={self.n_parameters()})"
        )
