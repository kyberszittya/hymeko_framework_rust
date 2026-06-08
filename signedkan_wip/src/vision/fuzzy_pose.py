"""FuzzySignaturePose: keypoint pose detection on the rev-6 fuzzy
signature backbone.

Plan: ``docs/plans/2026-05-31-fuzzy-pose-detection/plan.tex``.

The rev-6 ``FuzzySignatureClassifier`` validated 2026-05-30 (test
accuracy 0.5354 on MNIST 10k subset, 9.6k params via multi-arity
Atanassov-CR + HSiKAN-relaxed internals) is reused as a backbone.
The classifier head (mean-pool + Linear → logits) is replaced by:

  1. per-pixel ``Linear(d → n_kp)``: each vertex emits n_kp scores.
  2. reshape to per-keypoint spatial heatmaps (B, n_kp, H, W).
  3. differentiable soft-argmax over (H, W) → predicted (x, y) coords.

The pose head is fully-convolutional and small (264 params at d=32,
n_kp=8). Total model ≈ 9,576 params — pose detection at the same
budget as the rev-6 classifier.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

import torch.nn.functional as F

from .fuzzy_signature import (
    FuzzySignatureLayer,
    MultiArityFuzzySignatureLayer,
)
from .hsikan_vision import CRActivation, HSiKANVisionLayer


# ─── Soft-argmax (differentiable) ───────────────────────────────────


def soft_argmax_2d(heatmaps: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Differentiable soft-argmax over the last two dimensions of a
    heatmap tensor.

    Args:
        heatmaps: (..., H, W) score tensor. Any finite values; the
            softmax handles normalisation.
        beta: temperature. Lower → sharper (closer to argmax).
            Higher → more diffuse (gradient washed across all pixels).
            Default 1.0; the smoke tunes from there.

    Returns:
        coords: (..., 2) tensor of (x, y) coordinates in pixel space.
            x is the column (W) index, y is the row (H) index, both
            in floating-point pixel coordinates.
    """
    H, W = heatmaps.shape[-2:]
    # Flatten H, W for softmax normalisation.
    flat = heatmaps.flatten(start_dim=-2) / beta              # (..., H*W)
    weights = torch.softmax(flat, dim=-1)                      # (..., H*W)
    weights = weights.view(*heatmaps.shape)                    # (..., H, W)
    # Coordinate grids (broadcasted automatically).
    device = heatmaps.device
    dtype = heatmaps.dtype
    ys = torch.arange(H, device=device, dtype=dtype).view(H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, W)
    coord_y = (weights * ys).sum(dim=(-2, -1))                 # (...)
    coord_x = (weights * xs).sum(dim=(-2, -1))                 # (...)
    return torch.stack([coord_x, coord_y], dim=-1)             # (..., 2)


# ─── FuzzySignaturePoseModel ───────────────────────────────────────


class FuzzySignaturePoseModel(nn.Module):
    """Fuzzy signature backbone + per-pixel pose head + soft-argmax.

    The backbone mirrors the rev-6 validated ``FuzzySignatureClassifier``
    construction exactly, minus the final ``mean + Linear(d, n_cls)``
    head.

    Args:
        H, W: input image dimensions (e.g. 32 for the synthetic-pose
            smoke).
        n_keypoints: number of keypoints to predict.
        d, n_layers, arities: backbone width/depth/scales. Defaults
            match the validated rev-6 config.
        soft_argmax_beta: temperature for the soft-argmax. Default 1.0.

    Forward:
        x_img: (B, 1, H, W) input image.
        Returns: (B, n_keypoints, 2) predicted (x, y) coords in pixel
        space.
    """

    def __init__(self, H: int, W: int, n_keypoints: int,
                 *, d: int = 32, n_layers: int = 8,
                 arities: list[tuple[int, int]] | None = None,
                 m: int = 8,
                 # rev-6 backbone defaults
                 t_norm_kind: str = "min",
                 t_conorm_kind: str = "max",
                 cr_input_scale: str = "raw",
                 residual_kind: str = "additive_centered",
                 init_kind: str = "ramp",
                 ramp_strength: float = 1.5,
                 gate_center_learnable: bool = True,
                 fuzzification_kind: str = "linear",
                 # pose head
                 soft_argmax_beta: float = 1.0):
        super().__init__()
        if fuzzification_kind not in ("sigmoid", "linear"):
            raise ValueError(
                "fuzzification_kind must be 'sigmoid' or 'linear'; "
                f"got {fuzzification_kind!r}"
            )
        if n_keypoints < 1:
            raise ValueError(f"n_keypoints must be >= 1; got {n_keypoints}")
        if soft_argmax_beta <= 0:
            raise ValueError(
                f"soft_argmax_beta must be > 0; got {soft_argmax_beta}"
            )
        if arities is None:
            arities = [(3, 1), (5, 2)]
        self.H, self.W = H, W
        self.n_keypoints = n_keypoints
        self.d = d
        self.n_layers = n_layers
        self.arities = arities
        self.fuzzification_kind = fuzzification_kind
        self.soft_argmax_beta = soft_argmax_beta

        self.embed = nn.Linear(1, d)
        layer_kwargs = dict(
            t_norm_kind=t_norm_kind, t_conorm_kind=t_conorm_kind,
            m=m, cr_input_scale=cr_input_scale,
            residual_kind=residual_kind,
            init_kind=init_kind, ramp_strength=ramp_strength,
            gate_center_learnable=gate_center_learnable,
        )
        if len(arities) == 1:
            kernel, stride = arities[0]
            self.layers = nn.ModuleList([
                FuzzySignatureLayer(
                    d=d, H=H, W=W, kernel=kernel, stride=stride,
                    **layer_kwargs,
                )
                for _ in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                MultiArityFuzzySignatureLayer(
                    d=d, H=H, W=W, arities=arities, **layer_kwargs,
                )
                for _ in range(n_layers)
            ])
        # Per-pixel pose head: Linear(d → n_kp).
        self.pose_head = nn.Linear(d, n_keypoints)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """x_img: (B, 1, H, W) → (B, n_keypoints, 2) predicted coords."""
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)                       # (B, V, 1)
        v = self.embed(v)                               # (B, V, d)
        if self.fuzzification_kind == "sigmoid":
            v = torch.sigmoid(v)
        for L in self.layers:
            v = L(v)
        # v: (B, V, d) where V = H*W.
        scores = self.pose_head(v)                      # (B, V, n_kp)
        # Reshape to per-keypoint heatmaps (B, n_kp, H, W).
        heatmaps = scores.view(B, self.H, self.W, self.n_keypoints)
        heatmaps = heatmaps.permute(0, 3, 1, 2).contiguous()  # (B, n_kp, H, W)
        coords = soft_argmax_2d(heatmaps, beta=self.soft_argmax_beta)
        return coords                                   # (B, n_kp, 2)

    def heatmaps(self, x_img: torch.Tensor) -> torch.Tensor:
        """Same as forward but returns the raw heatmaps before
        soft-argmax. Useful for the auxiliary heatmap-supervision loss
        and for visualisation."""
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)
        if self.fuzzification_kind == "sigmoid":
            v = torch.sigmoid(v)
        for L in self.layers:
            v = L(v)
        scores = self.pose_head(v)
        return scores.view(B, self.H, self.W, self.n_keypoints) \
            .permute(0, 3, 1, 2).contiguous()


# ─── HSiKANPoseModel (vanilla HSiKAN backbone, for paired comparison) ─


class HSiKANPoseModel(nn.Module):
    """Vanilla HSiKAN backbone + same per-pixel pose head as
    FuzzySignaturePoseModel — paired comparison target.

    Reuses ``HSiKANVisionLayer`` (signed-branch / patch-mean polarity /
    CR per branch / αₖ multi-arity mixer) instead of the
    fuzzy-signature stack. Architecturally the same outer interface
    (image → per-pixel features → pose head → soft-argmax) so paired
    smokes use identical data, hyperparams, optimiser.

    The contrast is the *layer machinery*:
      - Fuzzy: Atanassov μ⁺/μ⁻ pair + t-norm/t-conorm + additive
        residual on [0,1] memberships.
      - Vanilla HSiKAN: per-pixel polarity gate against patch mean +
        signed-branch CR + sum-pooling + additive residual on
        unbounded signal.

    Same backbone shape (image → per-pixel features), same pose head
    (Linear → heatmaps → soft-argmax), so the comparison isolates the
    contribution of the fuzzy primitives.
    """

    def __init__(self, H: int, W: int, n_keypoints: int,
                 *, d: int = 32, n_layers: int = 8,
                 arities: list[tuple[int, int]] | None = None,
                 m: int = 8,
                 # vanilla-HSiKAN defaults (no fuzzy axes; same as
                 # HSiKANVisionClassifier).
                 tie_we: bool = False,
                 spatial_filter: str = "per_channel",
                 pooling: str = "sum",
                 # pose head
                 soft_argmax_beta: float = 1.0):
        super().__init__()
        if n_keypoints < 1:
            raise ValueError(f"n_keypoints must be >= 1; got {n_keypoints}")
        if soft_argmax_beta <= 0:
            raise ValueError(
                f"soft_argmax_beta must be > 0; got {soft_argmax_beta}"
            )
        if arities is None:
            arities = [(3, 1), (5, 2)]
        self.H, self.W = H, W
        self.n_keypoints = n_keypoints
        self.d = d
        self.n_layers = n_layers
        self.arities = arities
        self.soft_argmax_beta = soft_argmax_beta

        self.embed = nn.Linear(1, d)
        self.layers = nn.ModuleList([
            HSiKANVisionLayer(
                d_in=d, d_out=d, H=H, W=W, arities=arities, m=m,
                tie_we=tie_we, spatial_filter=spatial_filter,
                pooling=pooling,
            )
            for _ in range(n_layers)
        ])
        # Same per-pixel pose head as the fuzzy variant.
        self.pose_head = nn.Linear(d, n_keypoints)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """x_img: (B, 1, H, W) → (B, n_keypoints, 2) predicted coords."""
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)                              # (B, V, d)
        for L in self.layers:
            v = L(v) + v                                # HSiKAN-style additive residual
        scores = self.pose_head(v)                      # (B, V, n_kp)
        heatmaps = scores.view(B, self.H, self.W, self.n_keypoints)
        heatmaps = heatmaps.permute(0, 3, 1, 2).contiguous()
        return soft_argmax_2d(heatmaps, beta=self.soft_argmax_beta)

    def heatmaps(self, x_img: torch.Tensor) -> torch.Tensor:
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)
        for L in self.layers:
            v = L(v) + v
        scores = self.pose_head(v)
        return scores.view(B, self.H, self.W, self.n_keypoints) \
            .permute(0, 3, 1, 2).contiguous()


# ─── HybridHSiKANFuzzyLayer (highway α-mix at the layer level) ─────


class HybridHSiKANFuzzyLayer(nn.Module):
    """Parallel HSiKAN + Fuzzy-signature layer with a learnable
    per-channel highway α-mix.

    Forward:
      ``out = x + (1−α)·HSiKAN_layer(x) + α·(Fuzzy_layer(x) − x)``

    where ``α = σ(α_raw) ∈ [0,1]`` is a learnable per-channel scalar.
    At init ``α ≈ hybrid_alpha_init`` (default 0.05) — HSiKAN-dominant
    so the validated blob-prior is preserved. Gradient descent lifts
    α only where the fuzzy branch contributes useful signal.

    Design motivation 2026-05-31: vanilla HSiKAN converges to 99.97%
    variance-explained on synthetic Gaussian-blob pose in 9 epochs
    via its implicit blob-detection prior, while the bounded-fuzzy
    architecture cost 60 epochs to reach 55%. The hybrid keeps
    HSiKAN's strong prior AND admits the fuzzy branch where it
    earns its keep — generalising the rev-5 highway-gate idea
    from the residual to the architecture level.

    Both branches must produce ``(B, V, d)`` outputs. The HSiKAN
    branch is a ``HSiKANVisionLayer`` (no built-in residual; we add
    it explicitly here). The fuzzy branch is a
    ``FuzzySignatureLayer`` / ``MultiArityFuzzySignatureLayer``
    with ``residual_kind="additive_centered"``,
    ``cr_input_scale="raw"`` so it accepts unbounded carrier x.
    """

    def __init__(self, d: int, H: int, W: int,
                 arities: list[tuple[int, int]],
                 m: int = 8,
                 # HSiKAN branch knobs
                 tie_we: bool = False,
                 spatial_filter: str = "per_channel",
                 pooling: str = "sum",
                 # Fuzzy branch knobs
                 fuzzy_init_kind: str = "ramp",
                 fuzzy_t_norm: str = "min",
                 fuzzy_t_conorm: str = "max",
                 # Hybrid knobs
                 hybrid_alpha_init: float = 0.05):
        super().__init__()
        self.d = d
        self.arities = arities
        # HSiKAN multi-arity branch.
        self.hsikan = HSiKANVisionLayer(
            d_in=d, d_out=d, H=H, W=W, arities=arities, m=m,
            tie_we=tie_we, spatial_filter=spatial_filter,
            pooling=pooling,
        )
        # Fuzzy branch — single-arity FuzzySignatureLayer at the first
        # arity (multi-arity fuzzy would double the cost without
        # meaningful diversity gain at this stage; the HSiKAN side
        # already multi-aritises).
        kernel, stride = arities[0]
        self.fuzzy = FuzzySignatureLayer(
            d=d, H=H, W=W, kernel=kernel, stride=stride, m=m,
            t_norm_kind=fuzzy_t_norm, t_conorm_kind=fuzzy_t_conorm,
            cr_input_scale="raw",          # accept unbounded x
            residual_kind="additive_centered",  # HSiKAN-relaxed
            init_kind=fuzzy_init_kind,
            gate_center_learnable=True,
        )
        # Learnable per-channel α. Init: σ⁻¹(hybrid_alpha_init).
        p = max(min(hybrid_alpha_init, 1.0 - 1e-6), 1e-6)
        alpha_raw_init = torch.logit(torch.tensor(p))
        self.alpha_raw = nn.Parameter(
            torch.full((d,), alpha_raw_init.item())
        )

    @property
    def alpha_eff(self) -> torch.Tensor:
        """Effective per-channel α = σ(α_raw) ∈ (0, 1)."""
        return torch.sigmoid(self.alpha_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, d) → (B, V, d)."""
        # HSiKAN branch — no built-in residual, gives the L(x) delta.
        h_hsikan = self.hsikan(x)                       # (B, V, d)
        # Fuzzy branch with additive_centered residual: out = x + (h_v - 0.5).
        # Extract the fuzzy delta = (h_v - 0.5) by subtracting x.
        fuzzy_delta = self.fuzzy(x) - x                  # (B, V, d)
        alpha = self.alpha_eff                           # (d,)
        return x + (1.0 - alpha) * h_hsikan + alpha * fuzzy_delta


# ─── HybridHSiKANFuzzyPoseModel ────────────────────────────────────


class HybridHSiKANFuzzyPoseModel(nn.Module):
    """Hybrid HSiKAN + Fuzzy pose model — both inductive biases
    available, per-channel α-mix per layer chooses the balance.

    Stacks ``HybridHSiKANFuzzyLayer`` × ``n_layers`` between an embed
    and a per-pixel pose head with soft-argmax. The per-channel α
    starts HSiKAN-dominant (``α ≈ 0.05``) and gradient descent moves
    it where the fuzzy branch helps. The model thus achieves
    HSiKAN-quality on tasks HSiKAN already solves (blob detection on
    synthetic pose) while admitting fuzzy contribution on tasks where
    it matters (occlusion, real-image clutter, uncertainty
    calibration).
    """

    def __init__(self, H: int, W: int, n_keypoints: int,
                 *, d: int = 16, n_layers: int = 8,
                 arities: list[tuple[int, int]] | None = None,
                 m: int = 8,
                 # HSiKAN branch
                 tie_we: bool = False,
                 spatial_filter: str = "per_channel",
                 pooling: str = "sum",
                 # Fuzzy branch
                 fuzzy_init_kind: str = "ramp",
                 fuzzy_t_norm: str = "min",
                 fuzzy_t_conorm: str = "max",
                 # Hybrid mix
                 hybrid_alpha_init: float = 0.05,
                 # Pose head
                 soft_argmax_beta: float = 1.0):
        super().__init__()
        if n_keypoints < 1:
            raise ValueError(f"n_keypoints must be >= 1; got {n_keypoints}")
        if soft_argmax_beta <= 0:
            raise ValueError(
                f"soft_argmax_beta must be > 0; got {soft_argmax_beta}"
            )
        if arities is None:
            arities = [(3, 1), (5, 2)]
        self.H, self.W = H, W
        self.n_keypoints = n_keypoints
        self.d = d
        self.n_layers = n_layers
        self.arities = arities
        self.soft_argmax_beta = soft_argmax_beta

        self.embed = nn.Linear(1, d)
        self.layers = nn.ModuleList([
            HybridHSiKANFuzzyLayer(
                d=d, H=H, W=W, arities=arities, m=m,
                tie_we=tie_we, spatial_filter=spatial_filter,
                pooling=pooling,
                fuzzy_init_kind=fuzzy_init_kind,
                fuzzy_t_norm=fuzzy_t_norm,
                fuzzy_t_conorm=fuzzy_t_conorm,
                hybrid_alpha_init=hybrid_alpha_init,
            )
            for _ in range(n_layers)
        ])
        self.pose_head = nn.Linear(d, n_keypoints)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)                                # (B, V, d) unbounded
        for L in self.layers:
            v = L(v)
        scores = self.pose_head(v)                       # (B, V, n_kp)
        heatmaps = scores.view(B, self.H, self.W, self.n_keypoints)
        heatmaps = heatmaps.permute(0, 3, 1, 2).contiguous()
        return soft_argmax_2d(heatmaps, beta=self.soft_argmax_beta)

    def heatmaps(self, x_img: torch.Tensor) -> torch.Tensor:
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)
        for L in self.layers:
            v = L(v)
        scores = self.pose_head(v)
        return scores.view(B, self.H, self.W, self.n_keypoints) \
            .permute(0, 3, 1, 2).contiguous()


# ─── TinyCNNPoseModel (standard external baseline) ─────────────────


class _TinyCNNBlock(nn.Module):
    """One Conv-BN-GELU residual block; ``out = x + GELU(BN(Conv(x)))``.

    The block preserves channels and spatial size (Conv2d with stride 1,
    padding 1). Skip connection makes the stack composable to any depth
    without normalisation tricks beyond BN.
    """

    def __init__(self, d: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(d, d, kernel_size=kernel, padding=pad)
        self.bn = nn.BatchNorm2d(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.bn(self.conv(x)))


class TinyCNNPoseModel(nn.Module):
    """Tiny CNN pose model — external baseline for the 3-way (or 4-way)
    pose comparison.

    Standard architecture intentionally chosen for honest comparison:
      1. ``Conv2d(1, d, 1)`` — input embed (1×1, matches the
         ``Linear(1, d)`` embed of the fuzzy / HSiKAN / Gömb models).
      2. ``L`` residual ``Conv2d(d, d, kernel=3, padding=1) + BN + GELU``
         blocks with skip connections. Matches the HSiKAN-vision /
         fuzzy-signature L-deep residual pattern but with vanilla
         neural primitives (no fuzzy memberships, no signed branches,
         no Catmull-Rom splines, no αₖ mixer, no Gömb shells).
      3. ``Conv2d(d, n_keypoints, 1)`` — per-pixel pose head
         (functionally identical to the ``Linear(d, n_keypoints)`` of
         the other models, just written as Conv2d for image-tensor
         convenience).
      4. Soft-argmax over the per-keypoint heatmaps → ``(B, n_kp, 2)``.

    Parameter count at d=12, L=8, n_kp=8:
      embed:    1·1·1·12 + 12 = 24
      blocks:   L · (3·3·d·d + d + 2d_bn) = 8 · (1296 + 12 + 24) = 8·1332 = 10,656
      head:     1·1·12·8 + 8 = 104
      Total:    ≈ 10,784 params (closely matches fuzzy rev-6's 9,576).

    This is the "what would a normal CNN of comparable size do?"
    baseline that reviewers will demand for any framework comparison.
    """

    def __init__(self, H: int, W: int, n_keypoints: int,
                 *, d: int = 12, n_layers: int = 8, kernel: int = 3,
                 soft_argmax_beta: float = 1.0):
        super().__init__()
        if n_keypoints < 1:
            raise ValueError(f"n_keypoints must be >= 1; got {n_keypoints}")
        if soft_argmax_beta <= 0:
            raise ValueError(
                f"soft_argmax_beta must be > 0; got {soft_argmax_beta}"
            )
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError(
                f"kernel must be a positive odd integer; got {kernel}"
            )
        self.H, self.W = H, W
        self.n_keypoints = n_keypoints
        self.d = d
        self.n_layers = n_layers
        self.kernel = kernel
        self.soft_argmax_beta = soft_argmax_beta

        self.embed = nn.Conv2d(1, d, kernel_size=1)
        self.blocks = nn.ModuleList([
            _TinyCNNBlock(d=d, kernel=kernel) for _ in range(n_layers)
        ])
        self.head = nn.Conv2d(d, n_keypoints, kernel_size=1)

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """x_img: (B, 1, H, W) → (B, n_keypoints, 2) predicted coords."""
        v = self.embed(x_img)                            # (B, d, H, W)
        for blk in self.blocks:
            v = blk(v)
        heatmaps = self.head(v)                          # (B, n_kp, H, W)
        return soft_argmax_2d(heatmaps, beta=self.soft_argmax_beta)

    def heatmaps(self, x_img: torch.Tensor) -> torch.Tensor:
        v = self.embed(x_img)
        for blk in self.blocks:
            v = blk(v)
        return self.head(v)


# ─── Vision-Gömb head (3-shell V1/V4/IT analogue for pose readout) ──


class _VisionGombOuterShell(nn.Module):
    """Outer shell (V1 analogue) — M parallel refinement banks.

    Vision adaptation of ``hymeko_gomb.OuterFIRShell``. The signed-
    graph cycle aggregation is replaced by a depth-wise 3×3
    neighborhood-mean pool (the standard V1-cortex local receptive
    field). Each bank has its own pre-projection so different banks
    see different feature subspaces — analogous to OuterFIRShell's
    staggered Clifford coefficients across banks.

    Args:
        d_in: input per-vertex feature dim.
        d_bank: output dim per bank.
        M: number of parallel banks.
        H, W: spatial dimensions of the vertex grid (for the 3×3 pool).
        fuzzy: when True, replace ``GELU`` activation with an
            Atanassov-CR pair (μ⁺ and μ⁻ via two CR activations clamped
            to [0,1] with ramp init, mixed by ``g·μ⁺ + (1−g)·μ⁻`` where
            ``g = σ(x − ½)`` per channel). The output is in [0,1] per
            bank (per-pixel fuzzy membership). When False (default) the
            shell is the crisp version. This is the "outer" position
            in the ``gomb_fuzzy_at`` localization axis.

    Forward: (B, V, d_in) → (B, V, M·d_bank).
    """

    def __init__(self, d_in: int, d_bank: int, M: int,
                 H: int, W: int, fuzzy: bool = False,
                 cr_m: int = 8, ramp_strength: float = 1.5):
        super().__init__()
        self.M = M
        self.d_bank = d_bank
        self.H = H
        self.W = W
        self.fuzzy = fuzzy
        self.banks = nn.ModuleList([
            nn.Linear(d_in, d_bank) for _ in range(M)
        ])
        with torch.no_grad():
            for m, bank in enumerate(self.banks):
                bank.bias.fill_((m + 1) / (M + 1) - 0.5)
        if fuzzy:
            # Atanassov μ⁺/μ⁻ pair per bank. Each pair is a 2-branch
            # CRActivation (branch 0 = μ⁺, branch 1 = μ⁻). Ramp init
            # to avoid the all-0.5 collapse documented in rev-3 → rev-5.
            self.mu_pairs = nn.ModuleList([
                CRActivation(channels=d_bank, n_branches=2, m=cr_m,
                              init_scale=0.05, clamp_to_01=True)
                for _ in range(M)
            ])
            grid = torch.linspace(-3.0, 3.0, cr_m)
            with torch.no_grad():
                for pair in self.mu_pairs:
                    # branch 0 = μ⁺ (ramp up)
                    pair.cpts.data[0] = (
                        grid.view(1, -1).expand(d_bank, -1)
                        * ramp_strength
                    )
                    # branch 1 = μ⁻ (ramp down)
                    pair.cpts.data[1] = (
                        -grid.view(1, -1).expand(d_bank, -1)
                        * ramp_strength
                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, V, _ = x.shape
        outs = []
        for m_idx, bank in enumerate(self.banks):
            y = bank(x)                                 # (B, V, d_bank)
            # Reshape to (B, d_bank, H, W), depth-wise 3×3 avg pool,
            # back to (B, V, d_bank).
            y_img = y.transpose(1, 2).reshape(B, self.d_bank,
                                                self.H, self.W)
            y_img = F.avg_pool2d(y_img, kernel_size=3, stride=1,
                                   padding=1)
            y = y_img.reshape(B, self.d_bank, V).transpose(1, 2)
            if self.fuzzy:
                # Atanassov mix: g = σ(x). μ⁺ and μ⁻ via the CR pair.
                mu_plus = self.mu_pairs[m_idx](y, branch_idx=0)
                mu_minus = self.mu_pairs[m_idx](y, branch_idx=1)
                gate = torch.sigmoid(y)                  # gate centred at 0
                outs.append(gate * mu_plus + (1.0 - gate) * mu_minus)
            else:
                outs.append(F.gelu(y))                   # crisp default
        return torch.cat(outs, dim=-1)                  # (B, V, M·d_bank)


class _VisionGombMiddleShell(nn.Module):
    """Middle shell (V4 analogue) — Catmull-Rom spline aggregator.

    Mirrors ``hymeko_gomb.MiddleHSiKAN`` for vision: pre-project the
    outer-shell output, apply a per-channel CR spline (the
    nonlinearity the outer FIR-like shell lacks), produce refined
    per-vertex features.

    Forward: (B, V, d_in) → (B, V, d_layer).
    """

    def __init__(self, d_in: int, d_layer: int, m: int = 8,
                 init_scale: float = 0.05, fuzzy: bool = False,
                 ramp_strength: float = 1.5):
        super().__init__()
        self.pre_proj = nn.Linear(d_in, d_layer)
        self.fuzzy = fuzzy
        if fuzzy:
            # Atanassov μ⁺/μ⁻ pair on the projected feature. clamp_to_01
            # turns each branch into a fuzzy membership; mix via the
            # sigmoid gate. Ramp init.
            self.activation = CRActivation(
                channels=d_layer, n_branches=2, m=m,
                init_scale=init_scale, clamp_to_01=True,
            )
            grid = torch.linspace(-3.0, 3.0, m)
            with torch.no_grad():
                self.activation.cpts.data[0] = (
                    grid.view(1, -1).expand(d_layer, -1) * ramp_strength
                )
                self.activation.cpts.data[1] = (
                    -grid.view(1, -1).expand(d_layer, -1) * ramp_strength
                )
        else:
            # Crisp default: single-branch CR without [0,1] clamp.
            self.activation = CRActivation(
                channels=d_layer, n_branches=1, m=m,
                init_scale=init_scale, clamp_to_01=False,
            )

    def forward(self, x_outer: torch.Tensor) -> torch.Tensor:
        x_proj = self.pre_proj(x_outer)                  # (B, V, d_layer)
        if self.fuzzy:
            mu_plus = self.activation(x_proj, branch_idx=0)
            mu_minus = self.activation(x_proj, branch_idx=1)
            gate = torch.sigmoid(x_proj)
            return gate * mu_plus + (1.0 - gate) * mu_minus
        return self.activation(x_proj, branch_idx=0)     # crisp default


class _VisionGombInnerShell(nn.Module):
    """Inner shell (IT analogue) — per-pixel pose readout.

    Vision adaptation of the inner ``CPMLCore`` readout: replaces the
    link-prediction tier-stratified output with a per-pixel
    keypoint-heatmap readout. The "tier" structure is collapsed (no
    cycle tiers in vision) but the inner-shell *role* — final
    decision readout — is preserved.

    Forward: (B, V, d_in) → (B, n_kp, H, W) heatmaps.
    """

    def __init__(self, d_in: int, n_keypoints: int,
                 H: int, W: int, fuzzy: bool = False):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.H = H
        self.W = W
        self.fuzzy = fuzzy
        self.readout = nn.Linear(d_in, n_keypoints)

    def forward(self, x_mid: torch.Tensor) -> torch.Tensor:
        B = x_mid.shape[0]
        scores = self.readout(x_mid)                     # (B, V, n_kp)
        if self.fuzzy:
            # TSK-style defuzzification at the readout: scores become
            # fuzzy firing strengths in [0,1] via σ before reshape +
            # soft-argmax. Each pixel's contribution to a keypoint
            # heatmap is now an explicit fuzzy membership.
            scores = torch.sigmoid(scores)
        heatmaps = scores.view(B, self.H, self.W, self.n_keypoints)
        return heatmaps.permute(0, 3, 1, 2).contiguous()  # (B, n_kp, H, W)


class HSiKANGombPoseModel(nn.Module):
    """Deep-narrow HSiKAN backbone + 3-shell vision-Gömb pose head.

    User direction 2026-05-31:
      *"my plan is to have a hsikan backbone (deep, narrow) with a
        gömb head detecting poses"*

    Pipeline:
      1. **Backbone**: deep-narrow HSiKAN (default d=8, L=12, single
         arity) — the depth-narrow Pareto from 2026-05-30.
      2. **Outer shell (V1)**: M parallel refinement banks +
         depth-wise 3×3 pool → (B, V, M·d_bank).
      3. **Middle shell (V4)**: Linear + Catmull-Rom spline →
         (B, V, d_mid).
      4. **Inner shell (IT)**: per-pixel Linear → heatmaps.
      5. **Soft-argmax tail**: same as the other pose models →
         (B, n_kp, 2).

    Total params at d=8, L=12, M=4, d_bank=8, d_mid=16, n_kp=8:
      ≈ 2.8k (smaller than both fuzzy rev-6 and vanilla HSiKAN at
      h=16/L=8).

    Shares the soft-argmax tail with the other pose models so paired
    comparison is at iso-everything-except-internals.
    """

    def __init__(self, H: int, W: int, n_keypoints: int,
                 *, d: int = 8, n_layers: int = 12,
                 arities: list[tuple[int, int]] | None = None,
                 m: int = 8,
                 # vanilla-HSiKAN backbone defaults
                 tie_we: bool = False,
                 spatial_filter: str = "per_channel",
                 pooling: str = "sum",
                 # Gömb head dims
                 gomb_M: int = 4,
                 gomb_d_bank: int = 8,
                 gomb_d_mid: int = 16,
                 # Gömb fuzzy-localization axis (per-shell flag).
                 # Subset of {"outer", "middle", "inner"}. Each named
                 # shell uses the Atanassov-CR pair instead of the
                 # crisp variant. Empty tuple = fully-crisp G\"omb.
                 gomb_fuzzy_at: tuple[str, ...] = (),
                 # pose head
                 soft_argmax_beta: float = 1.0):
        super().__init__()
        if n_keypoints < 1:
            raise ValueError(f"n_keypoints must be >= 1; got {n_keypoints}")
        if soft_argmax_beta <= 0:
            raise ValueError(
                f"soft_argmax_beta must be > 0; got {soft_argmax_beta}"
            )
        if gomb_M < 1:
            raise ValueError(f"gomb_M must be >= 1; got {gomb_M}")
        if arities is None:
            arities = [(3, 1)]  # single-arity default — narrow-deep regime
        # Validate fuzzy-localization axis: only "outer", "middle",
        # "inner" are recognised. The axis is a *subset*, not a
        # categorical pick, so ablations can independently flip each
        # shell's fuzziness.
        valid_loc = {"outer", "middle", "inner"}
        bad = set(gomb_fuzzy_at) - valid_loc
        if bad:
            raise ValueError(
                f"gomb_fuzzy_at must be a subset of "
                f"{sorted(valid_loc)}; got unknown {sorted(bad)}"
            )
        self.H, self.W = H, W
        self.n_keypoints = n_keypoints
        self.d = d
        self.n_layers = n_layers
        self.arities = arities
        self.gomb_fuzzy_at = tuple(sorted(set(gomb_fuzzy_at)))
        self.soft_argmax_beta = soft_argmax_beta

        # ── deep-narrow HSiKAN backbone ──
        self.embed = nn.Linear(1, d)
        self.layers = nn.ModuleList([
            HSiKANVisionLayer(
                d_in=d, d_out=d, H=H, W=W, arities=arities, m=m,
                tie_we=tie_we, spatial_filter=spatial_filter,
                pooling=pooling,
            )
            for _ in range(n_layers)
        ])

        # ── vision-Gömb head (3 shells) ──
        # Per-shell fuzziness gated by ``gomb_fuzzy_at``.
        self.outer = _VisionGombOuterShell(
            d_in=d, d_bank=gomb_d_bank, M=gomb_M, H=H, W=W,
            fuzzy=("outer" in self.gomb_fuzzy_at),
        )
        self.middle = _VisionGombMiddleShell(
            d_in=gomb_M * gomb_d_bank, d_layer=gomb_d_mid, m=m,
            fuzzy=("middle" in self.gomb_fuzzy_at),
        )
        self.inner = _VisionGombInnerShell(
            d_in=gomb_d_mid, n_keypoints=n_keypoints, H=H, W=W,
            fuzzy=("inner" in self.gomb_fuzzy_at),
        )

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        """x_img: (B, 1, H, W) → (B, n_keypoints, 2) predicted coords."""
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)
        for L in self.layers:
            v = L(v) + v                                  # HSiKAN residual
        # Gömb head: outer → middle → inner.
        v = self.outer(v)
        v = self.middle(v)
        heatmaps = self.inner(v)                          # (B, n_kp, H, W)
        return soft_argmax_2d(heatmaps, beta=self.soft_argmax_beta)

    def heatmaps(self, x_img: torch.Tensor) -> torch.Tensor:
        B = x_img.shape[0]
        v = x_img.view(B, -1, 1)
        v = self.embed(v)
        for L in self.layers:
            v = L(v) + v
        v = self.outer(v)
        v = self.middle(v)
        return self.inner(v)


# ─── Synthetic pose-data generator (Gaussian-blob keypoints) ─────────


def _gauss_blob(H: int, W: int, cx: float, cy: float, sigma: float,
                device: torch.device | str = "cpu") -> torch.Tensor:
    """Render a 2D Gaussian centred at (cx, cy) with std sigma on a
    H×W grid. Amplitude 1 at the peak."""
    ys = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    return torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


def make_synthetic_pose_sample(
    H: int = 32, W: int = 32, n_keypoints: int = 8,
    blob_sigma: float = 1.5, bg_noise: float = 0.1,
    rng: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a single synthetic pose sample.

    Keypoints follow a coarse kinematic-chain prior on a 32×32 canvas:
      kp 0 (head):       (8 .. W-8, 4 .. 10)         — top of image
      kp 1 (l_shoulder): (head_x + (-7..-3), head_y + 5..9)
      kp 2 (r_shoulder): (head_x + (3..7),   head_y + 5..9)
      kp 3 (l_elbow):    (l_shoulder + offset)
      kp 4 (r_elbow):    (r_shoulder + offset)
      kp 5 (l_hand):     (l_elbow + offset)
      kp 6 (r_hand):     (r_elbow + offset)
      kp 7 (mid_hip):    (head_x, head_y + 16..22)

    Returns:
        image: (1, H, W) tensor with the n_kp Gaussian blobs additively
            composed plus uniform-noise background.
        coords: (n_kp, 2) tensor of (x, y) ground-truth coords in
            pixel space.
    """
    if rng is None:
        rng = torch.Generator()
        rng.manual_seed(0)

    def uni(lo, hi):
        return torch.rand(1, generator=rng).item() * (hi - lo) + lo

    head_x = uni(8, W - 8)
    head_y = uni(4, 10)
    l_sh_x = head_x + uni(-7, -3)
    l_sh_y = head_y + uni(5, 9)
    r_sh_x = head_x + uni(3, 7)
    r_sh_y = head_y + uni(5, 9)
    l_el_x = l_sh_x + uni(-5, -1)
    l_el_y = l_sh_y + uni(2, 6)
    r_el_x = r_sh_x + uni(1, 5)
    r_el_y = r_sh_y + uni(2, 6)
    l_ha_x = l_el_x + uni(-4, 0)
    l_ha_y = l_el_y + uni(2, 5)
    r_ha_x = r_el_x + uni(0, 4)
    r_ha_y = r_el_y + uni(2, 5)
    mid_hip_x = head_x + uni(-2, 2)
    mid_hip_y = head_y + uni(16, 22)

    # Clamp to image bounds (avoid keypoints outside).
    eps = blob_sigma + 1.0
    coords_list = [
        (head_x, head_y),
        (l_sh_x, l_sh_y), (r_sh_x, r_sh_y),
        (l_el_x, l_el_y), (r_el_x, r_el_y),
        (l_ha_x, l_ha_y), (r_ha_x, r_ha_y),
        (mid_hip_x, mid_hip_y),
    ]
    coords_list = [
        (max(eps, min(W - 1 - eps, cx)),
         max(eps, min(H - 1 - eps, cy)))
        for (cx, cy) in coords_list[:n_keypoints]
    ]
    coords = torch.tensor(coords_list, dtype=torch.float32)

    # Render image: each blob added in, plus background noise.
    img = bg_noise * torch.rand((H, W), generator=rng)
    for (cx, cy) in coords_list:
        img = img + _gauss_blob(H, W, cx, cy, blob_sigma)
    img = img.clamp(0.0, None).unsqueeze(0)                       # (1, H, W)
    return img, coords


class SyntheticPoseDataset(torch.utils.data.Dataset):
    """Synthetic keypoint pose dataset with full ground-truth control.
    Each item returns (image, coords): (1, H, W) and (n_kp, 2).
    """

    def __init__(self, n_samples: int, H: int = 32, W: int = 32,
                 n_keypoints: int = 8, blob_sigma: float = 1.5,
                 bg_noise: float = 0.1, seed: int = 0):
        self.n_samples = n_samples
        self.H = H
        self.W = W
        self.n_keypoints = n_keypoints
        self.blob_sigma = blob_sigma
        self.bg_noise = bg_noise
        self.seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Deterministic per-index sampling: seed = self.seed * 1_000_000 + idx.
        rng = torch.Generator()
        rng.manual_seed(self.seed * 1_000_000 + idx)
        return make_synthetic_pose_sample(
            H=self.H, W=self.W, n_keypoints=self.n_keypoints,
            blob_sigma=self.blob_sigma, bg_noise=self.bg_noise, rng=rng,
        )


__all__ = [
    "soft_argmax_2d",
    "FuzzySignaturePoseModel",
    "HSiKANPoseModel",
    "HSiKANGombPoseModel",
    "HybridHSiKANFuzzyLayer",
    "HybridHSiKANFuzzyPoseModel",
    "TinyCNNPoseModel",
    "make_synthetic_pose_sample",
    "SyntheticPoseDataset",
]
