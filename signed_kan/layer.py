"""The signed-KAN layer core: backend-aggregated signed messages → linear mix → spline → skip.

``h' = skip( σ_spline(W_self·x + W_+·A⁺x + W_-·A⁻x), x )``

One layer, four orthogonal axes (all config, no per-variant subclass): the **aggregation backend** (:mod:`backends`),
the **edge spline** (:mod:`splines`), the **skip/highway** (here), and the **incidence** (held by
:class:`~signed_kan.backbone.SignedKANBackbone`). This is the shared core that ``hymeko_rl`` (dense/inductive) and,
after phase 3, ``signedkan_wip`` (sparse/transductive) both build on.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backends import AggregationBackend, DenseBatchedBackend
from .splines import make_activation

SKIP_MODES = ("none", "residual", "highway")


class HighwaySkip(nn.Module):
    """Schmidhuber-style highway gate (Srivastava, Greff & Schmidhuber, 2015): ``T = σ(W_T·x + b_T)``,
    output ``T·H + (1 - T)·carry``. The gate bias is initialised to ``-2`` so ``T ≈ 0.12`` at the start —
    carry-dominant, i.e. the input flows through the "highway" and the network learns to transform. A
    projection aligns ``x`` to ``H``'s width when ``in_dim != out_dim``.

    # Preconditions ``in_dim, out_dim >= 1``.
    # Postconditions ``forward(H, x)`` returns an ``out_dim``-wide tensor shaped like ``H``.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(in_dim, out_dim)
        nn.init.constant_(self.gate.bias, -2.0)   # T ≈ 0.12 at init → carry-dominant (the highway init)
        self.carry: nn.Module = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, transformed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.gate(x))
        carried: torch.Tensor = self.carry(x)
        out: torch.Tensor = t * transformed + (1.0 - t) * carried
        return out


class _ResidualSkip(nn.Module):
    """``H + proj(x)`` (KAN-residual / ResNet special case). Projects ``x`` when ``in_dim != out_dim``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.carry: nn.Module = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, transformed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = transformed + self.carry(x)
        return out


def _make_skip(mode: str, in_dim: int, out_dim: int) -> nn.Module | None:
    """Skip Strategy. ``"none"`` returns ``None`` (no skip — parity with a plain signed-conv); ``"residual"``
    and ``"highway"`` return the wrapping module. # Errors ``ValueError`` on an unknown mode."""
    if mode == "none":
        return None
    if mode == "residual":
        return _ResidualSkip(in_dim, out_dim)
    if mode == "highway":
        return HighwaySkip(in_dim, out_dim)
    raise ValueError(f"skip must be one of {SKIP_MODES}; got {mode!r}")


class SignedKANLayer(nn.Module):
    """One signed-KAN convolution: ``skip( activation(W_self·x + W_+·agg_pos + W_-·agg_neg), x )``.

    The signed aggregation is delegated to an :class:`~signed_kan.backends.AggregationBackend` so the same layer
    serves the dense/batched (RL) and sparse (vision) paths. ``activation="cr"`` is the Catmull-Rom KAN spline
    that defines HSiKAN; ``skip="none"`` (default) reproduces a plain signed-conv (parity); ``"highway"`` adds
    the Schmidhuber gate (the H in HSiKAN).

    # Preconditions ``in_dim, out_dim >= 1``; ``activation in {cr, relu, tanh}``; ``skip in SKIP_MODES``.
    # Postconditions ``forward(x, a_pos, a_neg)`` returns ``(..., out_dim)`` shaped like the aggregation output.
    """

    def __init__(self, in_dim: int, out_dim: int, *, activation: str = "cr", skip: str = "none",
                 grid: int = 5, backend: AggregationBackend | None = None) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError(f"in_dim/out_dim must be >= 1; got {in_dim}/{out_dim}")
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_pos = nn.Linear(in_dim, out_dim)
        self.w_neg = nn.Linear(in_dim, out_dim)
        self.act = make_activation(activation, out_dim, grid=grid)
        self.skip = _make_skip(skip, in_dim, out_dim)
        self.backend: AggregationBackend = backend if backend is not None else DenseBatchedBackend()

    def forward(self, x: torch.Tensor, a_pos: torch.Tensor, a_neg: torch.Tensor) -> torch.Tensor:
        agg_pos, agg_neg = self.backend.aggregate(a_pos, a_neg, x)
        h: torch.Tensor = self.act(self.w_self(x) + self.w_pos(agg_pos) + self.w_neg(agg_neg))
        if self.skip is None:
            return h
        gated: torch.Tensor = self.skip(h, x)
        return gated
