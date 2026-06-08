"""HSiKAN-style hypergraph convolution on vision (the "generalize the
hypergraph conv" version, not HGNN).

The user's correction on 2026-05-04 was that the overnight vision run
used HGNN (Feng 2019) directly instead of generalizing the
HSiKAN-style hypergraph conv we built for signed graphs. This file
ports HSiKAN to vision:

| HSiKAN (signed graphs)           | this file (vision)                    |
|----------------------------------|---------------------------------------|
| edge sign s ∈ {-1, +1}           | contrast polarity p = sign(I - mean)  |
| signed-branch sum over (+,-)     | polarity-branch sum over (above,below)|
| Catmull-Rom activation per branch| Catmull-Rom activation per branch     |
| arities k ∈ {3,4,5} (cycle size) | arities k ∈ {5,8,12} (RF size)        |
| αₖ learnable arity mixer         | αₖ learnable RF-scale mixer           |
| signed-incidence H_k             | signed-incidence H_k (vert × patch)   |

Architecture:

    image (B, 1, H, W)
        ↓
    pixel embed (1 → d)
        ↓
    HSiKANVisionLayer × n_layers
        for each arity k (RF size):
            split each patch into above-mean and below-mean branches
            signed-branch hypergraph conv with Catmull-Rom activation
        αₖ-mix the n_arity branch outputs
        ↓
    global mean pool → linear classifier

Run:
    python -m signedkan_wip.src.vision.hsikan_vision \
        --dataset mnist --hidden 32 --n-epochs 5 --arities 5,8,12
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# ─── Receptive-field incidence ──────────────────────────────────────


def build_rf_incidence(
    H: int, W: int, kernel: int, stride: int,
) -> tuple[torch.Tensor, int]:
    """Sparse (V × E) receptive-field incidence: each kernel×kernel
    receptive field becomes one hyperedge containing all kernel² pixels."""
    rows = list(range(0, H - kernel + 1, stride))
    cols = list(range(0, W - kernel + 1, stride))
    n_edges = len(rows) * len(cols)
    n_verts = H * W
    inc = torch.zeros(n_verts, n_edges, dtype=torch.float32)
    e_idx = 0
    for r in rows:
        for c in cols:
            for dr in range(kernel):
                for dc in range(kernel):
                    inc[(r + dr) * W + (c + dc), e_idx] = 1.0
            e_idx += 1
    return inc, n_edges


def build_rf_edge_members(
    H: int, W: int, kernel: int, stride: int,
) -> tuple[torch.Tensor, int]:
    """For each receptive-field hyperedge, return its vertex membership
    as ``edge_members[E, K]`` (long) — the K = kernel² vertex indices
    of the pixels that compose that RF, in row-major position-within-RF
    order. Used by the t-norm pooling path (2026-05-30, Kóczy fuzzy
    signature work).

    Equivalent to ``argwhere(inc, axis=0)`` for each column, but in a
    contiguous index tensor amenable to ``x[:, edge_members, :]``
    gather → ``(B, E, K, d)`` for direct t-norm pooling over the K
    dimension.

    Returns:
        edge_members: (E, K) long.
        n_edges: int.
    """
    rows = list(range(0, H - kernel + 1, stride))
    cols = list(range(0, W - kernel + 1, stride))
    n_edges = len(rows) * len(cols)
    K = kernel * kernel
    out = torch.empty(n_edges, K, dtype=torch.long)
    e_idx = 0
    for r in rows:
        for c in cols:
            k_idx = 0
            for dr in range(kernel):
                for dc in range(kernel):
                    out[e_idx, k_idx] = (r + dr) * W + (c + dc)
                    k_idx += 1
            e_idx += 1
    return out, n_edges


def build_rf_position_incidence(
    H: int, W: int, kernel: int, stride: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Same RF tiling as ``build_rf_incidence`` but also returns the
    per-(vertex, edge) position-within-RF index, used by the
    spatial-filter HSiKAN variant.

    Returns:
        inc:     (V, E) float32, 1 iff vertex v ∈ RF e (same as
                 ``build_rf_incidence``).
        pos_inc: (V, E) long, in [0, kernel² - 1] for (v, e) with v ∈ e;
                 ``-1`` elsewhere (the spatial-filter forward clamps
                 these to 0 and zeroes them via ``inc``).
        n_edges: int.

    Position-within-RF is flat row-major: for an RF at (i, j) and pixel
    at (i+dr, j+dc), pos = dr*kernel + dc.
    """
    rows = list(range(0, H - kernel + 1, stride))
    cols = list(range(0, W - kernel + 1, stride))
    n_edges = len(rows) * len(cols)
    n_verts = H * W
    inc = torch.zeros(n_verts, n_edges, dtype=torch.float32)
    pos_inc = torch.full((n_verts, n_edges), -1, dtype=torch.long)
    e_idx = 0
    for r in rows:
        for c in cols:
            for dr in range(kernel):
                for dc in range(kernel):
                    v = (r + dr) * W + (c + dc)
                    inc[v, e_idx] = 1.0
                    pos_inc[v, e_idx] = dr * kernel + dc
            e_idx += 1
    return inc, pos_inc, n_edges


# ─── Catmull-Rom activation (lifted from signedkan_wip.src.core.splines) ─


class CRActivation(nn.Module):
    """Per-channel Catmull-Rom learnable activation (m control points
    on [-3, 3]). One activation per (channel, branch) = one set of
    control points."""
    def __init__(self, channels: int, n_branches: int, m: int = 8,
                 init_scale: float = 0.05, clamp_to_01: bool = False):
        """``clamp_to_01``: when True, the output is passed through a
        sigmoid so it stays in $[0,1]$. Used by the FuzzySignatureLayer
        (CR-as-learnable-membership-function). Default False preserves
        the prior unbounded activation behaviour."""
        super().__init__()
        self.m = m
        self.clamp_to_01 = clamp_to_01
        # Control point y-values; x is uniform on [-3, 3].
        # Shape (n_branches, channels, m)
        self.cpts = nn.Parameter(
            init_scale * torch.randn(n_branches, channels, m)
        )
        x = torch.linspace(-3.0, 3.0, m)
        self.register_buffer("x_grid", x)
        # CR offsets for i-1, i, i+1, i+2 — registered as a buffer so it
        # rides device transfers. Used by the no-cat fast path.
        self.register_buffer("_cr_offset", torch.tensor([-1, 0, 1, 2],
                                                        dtype=torch.long))

    def forward(self, x: torch.Tensor, branch_idx: int) -> torch.Tensor:
        """x: (..., channels), output: (..., channels)
        Catmull-Rom interpolation between control points.

        Speed note (2026-05-29 torch.profiler probe + fix): the prior
        impl did 4 separate advanced-indexing ops to gather
        ``cp[ch_idx, i{0..3}]`` — each backward materialised an
        ``index_put_`` accumulator, and the 4-op chain reached **76 % of
        CUDA time** at training-loop scale. Stacking the four indices
        into a single ``(..., 4)`` indexing op collapses those to ONE
        ``index_put_`` backward (4× fewer hot-kernel calls). Numerical
        parity with the legacy path is asserted in
        ``signedkan_wip/tests/test_cractivation_parity.py``.
        """
        # Find segment via clamp + bucketize.
        x_norm = (x.clamp(-3.0, 3.0) + 3.0) / 6.0 * (self.m - 1)  # in [0, m-1]
        i = x_norm.floor().long().clamp(0, self.m - 2)            # in [0, m-2]
        t = (x_norm - i.float()).clamp(0.0, 1.0)
        cp = self.cpts[branch_idx]  # (channels, m)

        # Stacked indices for Catmull-Rom's i-1, i, i+1, i+2 via broadcast
        # addition — replaces a torch.stack of 4 clamped tensors that was
        # showing up as ~12 % CUDA via aten::cat (2026-05-29 profile after
        # the initial CR fix). Net result is the same (..., channels, 4)
        # tensor, but no concat materialisation. One advanced-indexing op
        # → one index_put_ backward, still 4× fewer than the legacy.
        i_stack = (i.unsqueeze(-1) + self._cr_offset).clamp(0, self.m - 1)
        ch_idx = torch.arange(cp.shape[0], device=x.device).expand_as(i)
        ch_idx_4 = ch_idx.unsqueeze(-1).expand_as(i_stack)
        p_all = cp[ch_idx_4, i_stack]   # (..., channels, 4)
        p0, p1, p2, p3 = p_all.unbind(dim=-1)

        t2 = t * t
        t3 = t2 * t
        out = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        # Optional [0,1] clamp for the FuzzySignatureLayer's
        # CR-as-learnable-membership-function interpretation
        # (μ(x) = sigmoid(CR(x))). Default False preserves the
        # unbounded activation used by the HSiKAN-vision path.
        if self.clamp_to_01:
            out = torch.sigmoid(out)
        return out

    # Legacy path retained for parity testing. Functionally equivalent;
    # used only by `tests/test_cractivation_parity.py`.
    def _forward_legacy(self, x: torch.Tensor, branch_idx: int) -> torch.Tensor:
        x_norm = (x.clamp(-3.0, 3.0) + 3.0) / 6.0 * (self.m - 1)
        i = x_norm.floor().long().clamp(0, self.m - 2)
        t = (x_norm - i.float()).clamp(0.0, 1.0)
        cp = self.cpts[branch_idx]
        i0 = (i - 1).clamp(0, self.m - 1)
        i1 = i
        i2 = (i + 1).clamp(0, self.m - 1)
        i3 = (i + 2).clamp(0, self.m - 1)
        ch_idx = torch.arange(cp.shape[0], device=x.device).expand_as(i)
        p0 = cp[ch_idx, i0]
        p1 = cp[ch_idx, i1]
        p2 = cp[ch_idx, i2]
        p3 = cp[ch_idx, i3]
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )


# ─── Signed-branch hypergraph conv ──────────────────────────────────


class SignedBranchConv(nn.Module):
    """For one arity k:
       - patch-mean μ_e from H_k incidence
       - per-pixel polarity branch: above (s=+1) or below (s=-1) μ_e
       - Catmull-Rom activation per branch
       - aggregated back to vertices via H_k

    This is the HSiKAN structure ported to vision: the "sign" of a
    (pixel, patch) pair is determined by whether the pixel is above
    or below the patch mean."""
    def __init__(self, d_in: int, d_out: int, n_edges: int, m: int = 8,
                 tie_we: bool = False, spatial_filter: str = "none",
                 kernel: int = 0, pooling: str = "sum"):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_out, bias=False)
        # Per-hyperedge scalar weight (analogous to W_e in HGNN).
        # ``tie_we=True`` shares a single scalar across all RF positions
        # — the minimum change to make this op translation-equivariant
        # (the per-edge index breaks equivariance; boundary D_v effects
        # remain). Forward unchanged: broadcasts cleanly over (n_e,)
        # or (1,). 2026-05-29 follow-up to the vision-rebench-strong
        # report: tests whether the residual 4.5-7 pp CNN gap is the
        # W_e equivariance break.
        self.tie_we = tie_we
        self.W_e = nn.Parameter(torch.ones(1 if tie_we else n_edges))
        # Within-RF spatial filter — translation-equivariant
        # position-within-RF weighting, shared across all RF positions
        # so the operator stays spatial-aware *without* breaking
        # equivariance. Initial all-ones reproduces the uniform-mean
        # aggregation bit-for-bit; gradient descent moves it away.
        # Modes:
        #   "none"        — disabled (current uniform mean).
        #   "scalar"      — W_pos[K], one scalar per position-within-RF,
        #                   channel-invariant. Cheap. MNIST partial win
        #                   (+0.71 pp), Fashion null (2026-05-29 report).
        #   "per_channel" — W_pos[K, d_out], one scalar per position ×
        #                   per output channel. K·d_out params per arity;
        #                   meant to address Fashion's null at "scalar".
        if spatial_filter not in ("none", "scalar", "per_channel"):
            raise ValueError(
                f"spatial_filter must be one of none/scalar/per_channel; "
                f"got {spatial_filter!r}"
            )
        self.spatial_filter = spatial_filter
        if spatial_filter != "none":
            if kernel <= 0:
                raise ValueError("spatial_filter requires kernel > 0")
            if spatial_filter == "scalar":
                self.W_pos = nn.Parameter(torch.ones(kernel * kernel))
            else:  # per_channel
                self.W_pos = nn.Parameter(torch.ones(kernel * kernel, d_out))
        else:
            self.register_parameter("W_pos", None)
        # T-norm pooling (2026-05-30, Kóczy fuzzy-signature work).
        # ``pooling`` ∈ {"sum", "min", "product", "lukasiewicz"}:
        #   - "sum" (default): current weighted-sum aggregation, identical to
        #     standard HSiKAN / depthwise-separable CNN view. NOT a t-norm.
        #   - "min": Gödel/Zadeh t-norm — strictest fuzzy conjunction.
        #     T(a,b) = min(a,b); n-ary: min over RF members.
        #   - "product": algebraic / probabilistic t-norm.
        #     T(a,b) = a·b; n-ary: Π over RF members.
        #   - "lukasiewicz": Łukasiewicz t-norm.
        #     T(a,b) = max(0, a+b-1); n-ary: max(0, Σ - (K-1)).
        # All t-norms require inputs in [0,1]; sigmoid is applied to
        # x_proj before pooling. The four-step polarity/signed-branches
        # path is preserved; only the v_to_e aggregation changes.
        if pooling not in ("sum", "min", "product", "lukasiewicz"):
            raise ValueError(
                f"pooling must be one of sum/min/product/lukasiewicz; "
                f"got {pooling!r}"
            )
        self.pooling = pooling
        # Two branches: 0 = above-mean (s=+1), 1 = below-mean (s=-1).
        self.activation = CRActivation(channels=d_out, n_branches=2, m=m)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x: torch.Tensor, inc: torch.Tensor,
                D_v_inv_sqrt: torch.Tensor, D_e_inv: torch.Tensor,
                pos_inc: torch.Tensor | None = None,
                edge_members: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, V, d_in)
        inc: (V, E) sparse-friendly incidence (dense here for clarity)
        pos_inc: (V, E) long, position-within-RF index in [-1, K-1];
                 required when spatial_filter≠"none", ignored otherwise.
        edge_members: (E, K) long, vertex indices per RF in row-major
                 position-within-RF order. Required when pooling≠"sum"
                 (t-norm pooling needs explicit per-edge gather); ignored
                 when pooling="sum" (sum-pooling uses dense einsum).
        Returns: (B, V, d_out)
        """
        x_proj = self.proj_in(x)  # (B, V, d_out)
        # If spatial-filter is enabled, weight the binary incidence by
        # ``W_pos`` at each (v, e) position-within-RF. Non-membership
        # entries (pos_inc=-1) are clamped to 0 and zeroed via the
        # binary inc. With W_pos=ones at init, both modes exactly
        # reproduce the uniform-mean aggregation.
        sf = self.spatial_filter
        # `w_inc` is what each einsum consumes. In `none`/`scalar` modes
        # it's the 2D incidence (possibly already scaled per-position);
        # in `per_channel` mode it's a 3D (V, E, d_out) tensor that adds
        # the per-channel weighting.
        if sf == "none":
            w_inc = inc
        else:
            if pos_inc is None:
                raise ValueError(
                    f"spatial_filter={sf!r} forward needs pos_inc; got None"
                )
            pos_idx = pos_inc.clamp_min(0)
            if sf == "scalar":
                # W_pos: (K,) → weighted_inc still 2D (V, E) — fast.
                w_inc = inc * self.W_pos[pos_idx]
            else:  # per_channel
                # W_pos: (K, d_out) → 3D (V, E, d_out).
                w_inc = inc.unsqueeze(-1) * self.W_pos[pos_idx]

        pool = self.pooling
        if pool != "sum" and edge_members is None:
            raise ValueError(
                f"pooling={pool!r} forward needs edge_members; got None"
            )

        # Direction-dispatched einsums (sum mode) or gather-then-tnorm
        # (t-norm modes). Only v_to_e changes between modes — e_to_v is
        # Sugeno-style defuzzification (sum-back), unchanged.
        def v_to_e(x_in: torch.Tensor) -> torch.Tensor:
            # x_in: (B, V, d) → (B, E, d)
            if pool == "sum":
                if w_inc.dim() == 2:
                    return torch.einsum("ve,bvd->bed", w_inc, x_in)
                return torch.einsum("ved,bvd->bed", w_inc, x_in)
            # T-norm path: gather pixel features per RF, apply sigmoid
            # (inputs must lie in [0, 1]), then pool over the K dim.
            # x_per_e: (B, E, K, d) by gathering x_in at edge_members.
            x_per_e = x_in[:, edge_members, :]  # (B, E, K, d)
            x_per_e = torch.sigmoid(x_per_e)
            if pool == "min":
                return x_per_e.amin(dim=2)
            if pool == "product":
                # Product t-norm: Π over K. Numerically: exp(Σ log).
                # Add eps to avoid log(0) when sigmoid saturates.
                return torch.exp(torch.log(x_per_e.clamp(min=1e-7)).sum(dim=2))
            # Łukasiewicz: max(0, Σ - (K-1)).
            K = x_per_e.shape[2]
            return (x_per_e.sum(dim=2) - (K - 1)).clamp(min=0.0)

        def e_to_v(x_in: torch.Tensor) -> torch.Tensor:
            # x_in: (B, E, d) → (B, V, d). Defuzzification: weighted
            # sum (Sugeno-style), unchanged by pooling choice.
            if w_inc.dim() == 2:
                return torch.einsum("ve,bed->bvd", w_inc, x_in)
            return torch.einsum("ved,bed->bvd", w_inc, x_in)

        # 1. compute per-patch mean (per channel): μ_e = (H^T x) / D_e
        h_t_x = v_to_e(x_proj)
        mu_e = h_t_x * D_e_inv.unsqueeze(0).unsqueeze(-1)   # (B, E, d_out)

        # 2. per (pixel, patch) determine polarity. We need to know,
        # for each pixel-patch pair, whether x_proj[pixel] > μ[patch].
        # To keep this dense-friendly: pull μ back to vertices weighted
        # by incidence (avg μ over patches the pixel belongs to).
        mu_v = e_to_v(mu_e) * D_v_inv_sqrt.unsqueeze(0).unsqueeze(-1) ** 2
        # polarity: above (+1) or below (0). Use float for differentiable α below.
        polarity = (x_proj - mu_v)   # (B, V, d_out)

        # 3. Compute the two branch contributions, gated by polarity sign.
        above = self.activation(polarity, branch_idx=0)
        below = self.activation(polarity, branch_idx=1)
        gate = torch.sigmoid(polarity * 4.0)   # smooth sign
        x_branch = gate * above + (1.0 - gate) * below   # (B, V, d_out)

        # 4. Hypergraph propagation H W_e D_e^{-1} H^T over the
        # branch-activated features.
        h_t_x_b = v_to_e(x_branch)
        h_t_x_b = h_t_x_b * (D_e_inv * self.W_e).unsqueeze(0).unsqueeze(-1)
        out = e_to_v(h_t_x_b)
        out = D_v_inv_sqrt.unsqueeze(0).unsqueeze(-1) * out
        return out + self.bias


# ─── HSiKAN-style multi-arity vision layer ──────────────────────────


class HSiKANVisionLayer(nn.Module):
    """For each receptive-field size k (an "arity"), run a
    SignedBranchConv. Combine arities with a learnable αₖ mixer
    (softmax-normalized so α sums to 1, as in MixedAritySignedKAN)."""
    def __init__(self, d_in: int, d_out: int, H: int, W: int,
                 arities: list[tuple[int, int]], m: int = 8,
                 tie_we: bool = False, spatial_filter: str = "none",
                 pooling: str = "sum"):
        """arities: list of (kernel, stride) pairs. The default for
        28×28 input is [(5,2), (8,4), (12,4)] — a 3-scale RF mix.

        ``tie_we``: see ``SignedBranchConv.__init__``.
        ``spatial_filter`` ∈ {"none","scalar","per_channel"}: registers
        per-arity ``pos_*`` buffers when ≠"none" and passes them to
        each conv forward.
        ``pooling`` ∈ {"sum","min","product","lukasiewicz"}: registers
        per-arity ``em_*`` (edge-members) buffer when ≠"sum" and passes
        it to each conv forward (2026-05-30 t-norm pooling)."""
        super().__init__()
        self.arities = arities
        self.spatial_filter = spatial_filter
        self.pooling = pooling
        self.convs = nn.ModuleList()
        for (k, s) in arities:
            if spatial_filter != "none":
                inc, pos_inc, n_e = build_rf_position_incidence(
                    H, W, kernel=k, stride=s)
                self.register_buffer(f"pos_{k}_{s}", pos_inc)
            else:
                inc, n_e = build_rf_incidence(H, W, kernel=k, stride=s)
            if pooling != "sum":
                edge_members, _ = build_rf_edge_members(
                    H, W, kernel=k, stride=s)
                self.register_buffer(f"em_{k}_{s}", edge_members)
            D_v = inc.sum(dim=1).clamp(min=1)
            D_e = inc.sum(dim=0).clamp(min=1)
            conv = SignedBranchConv(
                d_in, d_out, n_e, m=m, tie_we=tie_we,
                spatial_filter=spatial_filter,
                kernel=k if spatial_filter != "none" else 0,
                pooling=pooling,
            )
            self.convs.append(conv)
            self.register_buffer(f"inc_{k}_{s}", inc)
            self.register_buffer(f"Dv_{k}_{s}", D_v.pow(-0.5))
            self.register_buffer(f"De_{k}_{s}", D_e.pow(-1.0))
        # αₖ mixer (one weight per arity, softmax-normalized).
        self.alpha_logits = nn.Parameter(torch.zeros(len(arities)))

    def alpha_weights(self) -> torch.Tensor:
        return F.softmax(self.alpha_logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, d_in)
        alpha = self.alpha_weights()
        out = 0.0
        for i, (k, s) in enumerate(self.arities):
            inc = getattr(self, f"inc_{k}_{s}")
            D_v = getattr(self, f"Dv_{k}_{s}")
            D_e = getattr(self, f"De_{k}_{s}")
            pos_inc = (getattr(self, f"pos_{k}_{s}", None)
                       if self.spatial_filter != "none" else None)
            em = (getattr(self, f"em_{k}_{s}", None)
                  if self.pooling != "sum" else None)
            out = out + alpha[i] * self.convs[i](
                x, inc, D_v, D_e, pos_inc, em)
        return out


class HSiKANVisionClassifier(nn.Module):
    """Embed → n HSiKANVisionLayer → mean-pool → linear head."""
    def __init__(self, H: int, W: int, n_classes: int,
                 hidden: int = 32, n_layers: int = 2,
                 arities: list[tuple[int, int]] | None = None,
                 m: int = 8, tie_we: bool = False,
                 spatial_filter: str = "none", pooling: str = "sum"):
        """``tie_we`` / ``spatial_filter`` / ``pooling``: propagate to
        every per-arity conv in every layer.
        See ``SignedBranchConv.__init__``."""
        super().__init__()
        if arities is None:
            arities = [(5, 2), (8, 4), (12, 4)]
        self.tie_we = tie_we
        self.spatial_filter = spatial_filter
        self.pooling = pooling
        self.embed = nn.Linear(1, hidden)
        self.layers = nn.ModuleList([
            HSiKANVisionLayer(hidden, hidden, H, W, arities, m=m,
                              tie_we=tie_we, spatial_filter=spatial_filter,
                              pooling=pooling)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)
        for L in self.layers:
            v = L(v) + v   # residual; HSiKAN-style
        z = v.mean(dim=1)
        return self.head(z)

    def alpha_summary(self) -> dict:
        return {
            f"layer_{i}": dict(zip(
                [f"k={k}_s={s}" for (k, s) in L.arities],
                [round(a.item(), 4) for a in L.alpha_weights()],
            ))
            for i, L in enumerate(self.layers)
        }


# ─── Train + eval (mirrors neocog_hgnn.main) ────────────────────────


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "fashion", "cifar10"],
                    default="mnist")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--m", type=int, default=8, help="Catmull-Rom control points")
    ap.add_argument("--arities", type=str, default="5_2,8_4,12_4",
                    help="comma-separated kernel_stride pairs")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arities = []
    for tok in args.arities.split(","):
        k, s = tok.split("_")
        arities.append((int(k), int(s)))

    # Load dataset.
    transform = T.Compose([T.ToTensor()])
    data_root = Path("/tmp/torchvision_cache")
    data_root.mkdir(exist_ok=True)
    if args.dataset == "mnist":
        train_ds = torchvision.datasets.MNIST(data_root, train=True,
                                                download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST(data_root, train=False,
                                               download=True, transform=transform)
        H, W, n_classes = 28, 28, 10
    elif args.dataset == "fashion":
        train_ds = torchvision.datasets.FashionMNIST(
            data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.FashionMNIST(
            data_root, train=False, download=True, transform=transform)
        H, W, n_classes = 28, 28, 10
    elif args.dataset == "cifar10":
        transform = T.Compose([T.Grayscale(), T.ToTensor()])
        train_ds = torchvision.datasets.CIFAR10(
            data_root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(
            data_root, train=False, download=True, transform=transform)
        H, W, n_classes = 32, 32, 10

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HSiKANVisionClassifier(
        H, W, n_classes, hidden=args.hidden, n_layers=args.n_layers,
        arities=arities, m=args.m,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0

    model.eval()
    correct = total = 0
    t1 = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()
    eval_time = time.time() - t1
    acc = correct / total

    print(json.dumps({
        "dataset": args.dataset,
        "model": "hsikan_vision",
        "arities": arities,
        "hidden": args.hidden,
        "n_layers": args.n_layers,
        "m": args.m,
        "n_epochs": args.n_epochs,
        "seed": args.seed,
        "test_accuracy": acc,
        "train_time_s": train_time,
        "eval_time_s": eval_time,
        "n_params": n_params(model),
        "alpha": model.alpha_summary(),
    }))


if __name__ == "__main__":
    main()
