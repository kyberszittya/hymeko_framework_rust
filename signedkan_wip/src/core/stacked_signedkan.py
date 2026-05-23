"""ResNet-style stackable HSIKAN — Phase 16 (2026-05-20 overnight).

The existing :class:`MultiLayerSignedKAN` (in
``signedkan_wip.src.core.signedkan``) already supports multi-layer
stacking with several skip-connection variants. Phase 16 exposes
the residual-stack pattern as a named architecture:

  * :class:`SignedKANResidualBlock` — single residual block (pre-
    norm + SignedKAN layer + identity skip on the vertex side).
    Thin diagrammable wrapper over the building blocks inside
    :class:`MultiLayerSignedKAN.encode_triads`'s loop.
  * :class:`StackedSignedKAN` — composes ``n_blocks`` of the above
    with ResNet-style defaults
    (``inner_skip="residual"``, ``layer_norm_between=True``,
    ``use_residual=True``, ``jk_mode="last"``).

The ResNet-style defaults make the model behave like a
``ResNet-tiny`` on the signed-graph side: identity skip on every
block, pre-LayerNorm, no jumping-knowledge mixing. Depth becomes
the headline knob the P-graph framework can search over.

Empirical validation: 5-seed × $L \\in \\{1, 2, 4, 8\\}$ A/B on
Bitcoin Alpha lives in
``signedkan_wip/experiments/runs/run_hsikan_depth_scaling.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .signedkan import (
    MultiLayerSignedKAN,
    MultiLayerSignedKANConfig,
    SignedKANLayer,
    SignedKANConfig,
)


@dataclass
class StackedSignedKANConfig:
    """Configuration for the ResNet-style stackable HSIKAN.

    All knobs forward to :class:`MultiLayerSignedKANConfig`; the
    defaults are tuned for the ResNet correspondence:

    * ``inner_skip="residual"`` — identity skip inside each
      SignedKAN layer (not highway gates).
    * ``use_residual=True`` — identity skip on the vertex side
      between blocks.
    * ``layer_norm_between=True`` — pre-norm before each block.
    * ``jk_mode="last"`` — the final block's triad embedding is
      the model output (the ResNet head pattern).
    """

    n_nodes: int
    n_blocks: int = 4
    hidden_dim: int = 32
    grid: int = 5
    k: int = 3
    use_minus_branch: bool = True
    init_scale: float = 0.1
    spline_kind: str = "bspline"

    def to_multilayer_config(self) -> MultiLayerSignedKANConfig:
        """Project onto :class:`MultiLayerSignedKANConfig` with
        ResNet-style defaults fixed."""
        return MultiLayerSignedKANConfig(
            n_nodes=self.n_nodes,
            n_layers=self.n_blocks,
            hidden_dim=self.hidden_dim,
            grid=self.grid,
            k=self.k,
            use_minus_branch=self.use_minus_branch,
            init_scale=self.init_scale,
            spline_kinds=[self.spline_kind] * self.n_blocks,
            # ResNet-style defaults:
            inner_skip="residual",
            outer_skip="none",
            use_residual=True,
            layer_norm_between=True,
            jk_mode="last",
            pool_mode="mean",
            share_weights=False,
        )


class SignedKANResidualBlock(nn.Module):
    """One ResNet-style residual block over signed triads.

    Architecture (one block, given input ``h_v``):

    .. math::

       \\tilde h_v &= \\mathrm{LayerNorm}(h_v) \\\\
       h_t        &= \\mathrm{SignedKANLayer}(\\tilde h_v,
                          \\mathrm{triad\\_v},
                          \\mathrm{triad\\_sigma}) \\\\
       h_v^{new}  &= h_v + M_{vt} h_t

    The vertex-side residual `h_v + M_{vt} h_t` is the analogue of
    ResNet's `x + F(x)` identity skip. The inner
    :class:`SignedKANLayer` is also configured with
    ``inner_skip="residual"`` so the spline activations themselves
    are residual; layered together this gives the "double-skip"
    pattern (within-layer + between-layer) that ResNet uses.

    Use :class:`StackedSignedKAN` to compose multiple blocks; this
    class is the diagrammable building block.
    """

    def __init__(
        self,
        n_nodes: int,
        hidden_dim: int,
        *,
        grid: int = 5,
        k: int = 3,
        spline_kind: str = "bspline",
        use_minus_branch: bool = True,
        init_scale: float = 0.1,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        self.layer = SignedKANLayer(SignedKANConfig(
            n_nodes=n_nodes,
            hidden_dim=hidden_dim,
            grid=grid, k=k,
            use_minus_branch=use_minus_branch,
            init_scale=init_scale,
            spline_kind=spline_kind,
            inner_skip="residual",
            outer_skip="none",
        ))

    def forward(
        self,
        h_v: torch.Tensor,
        triad_v: torch.Tensor,
        triad_sigma: torch.Tensor,
        M_vt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(h_v_new, h_t)``.

        ``h_v_new`` carries the residual update for the next block;
        ``h_t`` is the per-triad embedding (used by the head at the
        final block).
        """
        h_v_norm = self.norm(h_v)
        h_t = self.layer(h_v_norm, triad_v, triad_sigma)
        h_v_new = h_v + torch.sparse.mm(M_vt, h_t)
        return h_v_new, h_t


class StackedSignedKAN(nn.Module):
    """ResNet-style stacked HSIKAN.

    Thin wrapper over :class:`MultiLayerSignedKAN` with the ResNet
    defaults baked in (see
    :meth:`StackedSignedKANConfig.to_multilayer_config`). Exposes
    the same ``encode_triads`` API so existing training harnesses
    (``run_compare.run_one``) can swap it in without changes.

    .. note::
        The forward path delegates directly to
        :class:`MultiLayerSignedKAN.encode_triads`. The reason for a
        thin wrapper rather than a from-scratch reimplementation:
        the `MultiLayerSignedKAN` machinery is already battle-
        tested by the 21+ HSIKAN tests in the repo. Phase 16's
        contribution is the *named architecture* (ResNet
        correspondence) + the empirical depth-scaling result.
    """

    def __init__(self, cfg: StackedSignedKANConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner = MultiLayerSignedKAN(cfg.to_multilayer_config())

    @property
    def node_embed(self):  # back-compat for callers that read .node_embed
        return self.inner.node_embed

    def encode_triads(
        self,
        triad_v: torch.Tensor,
        triad_sigma: torch.Tensor,
        M_vt: torch.Tensor,
        return_h_v: bool = False,
    ):
        return self.inner.encode_triads(
            triad_v, triad_sigma, M_vt, return_h_v=return_h_v,
        )

    def num_parameters(self) -> int:
        return self.inner.num_parameters()


__all__ = [
    "SignedKANResidualBlock",
    "StackedSignedKAN",
    "StackedSignedKANConfig",
]
