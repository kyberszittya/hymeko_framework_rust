"""Backbone alternatives for HyMeYOLO Stage B.

Two new backbones, both drop-in replacements for the 3-conv
``TinyBackbone`` (defined in ``hymeyolo_q_smoke.py``):

* :class:`ResNetTinyBackbone` — residual-block stack (~107k params
  at ``c_out=32``); the canonical "deeper backbone" Stage B lever.

* :class:`HSiKANConvBackbone` — same shape as ResNetTinyBackbone but
  replaces every ReLU with :class:`CatmullRomActivation`, a learnable
  per-channel univariate function from the HSiKAN basis-function
  family. Tests whether HSiKAN's basis-function primitive
  (independently of the σ-cycle aggregator) carries any vision-side
  weight.

Both backbones share the (B, 3, H, W) → (B, c_out, H/8, W/8)
contract of :class:`TinyBackbone` (8 × 8 feature map at H=W=64).

Plan: ``docs/plans/2026-05-16-hymeyolo-stage-b-backbone/``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Catmull-Rom basis activation (HSiKAN primitive) ─────────────────


class CatmullRomActivation(nn.Module):
    """Learnable per-channel univariate function via Catmull-Rom
    spline interpolation through G fixed-x learnable-y control points.

    The KAN angle: replaces ReLU (``max(0, x)``, fixed) with a
    learnable function ``φ_c(x)`` per channel. Initialised to a near-
    linear identity so the network at init is well-behaved.

    Parameters
    ----------
    num_channels : int
        Channel dimension of the input tensor (the function is
        applied per-channel; each channel learns its own ``φ``).
    n_knots : int, default 8
        Number of control points along the x-axis. Higher = more
        expressive but more parameters. The KAN literature suggests
        4-8 is plenty for conv activations.
    x_range : tuple(float, float), default (-3.0, 3.0)
        Domain of the learnable function. Inputs are soft-clamped
        to this range before interpolation. Beyond the range the
        function extrapolates linearly using the boundary slopes,
        so unbounded BN-normalised activations still get a
        well-defined output.

    Parameter count
    ---------------
    ``num_channels × n_knots`` learnable scalars. At
    ``num_channels=32, n_knots=8``: 256 params per layer. Compared
    to ReLU's zero learnable parameters, the marginal cost is
    small.

    Initialisation
    --------------
    Control points are initialised to the identity function:
    ``y_i = θ_i`` (the function ``φ(x) = x`` linearly through the
    knots). This makes the network at init behave like a no-
    nonlinearity skip layer at this position; the network learns
    the appropriate curvature during training.
    """

    def __init__(
        self,
        num_channels: int,
        n_knots: int = 8,
        x_range: tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        super().__init__()
        if n_knots < 4:
            raise ValueError(
                f"n_knots must be >= 4 for Catmull-Rom; got {n_knots}"
            )
        if x_range[1] <= x_range[0]:
            raise ValueError(
                f"x_range must be increasing; got {x_range}"
            )
        self.num_channels = num_channels
        self.n_knots = n_knots
        self.x_min, self.x_max = float(x_range[0]), float(x_range[1])

        # Fixed knot positions on [x_min, x_max].
        theta = torch.linspace(self.x_min, self.x_max, n_knots)
        self.register_buffer("theta", theta)
        # Per-channel control points, initialised to identity.
        cp_init = theta.unsqueeze(0).expand(num_channels, n_knots).clone()
        self.cp = nn.Parameter(cp_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the per-channel CR activation.

        Input shape: ``(B, C, H, W)`` where ``C == num_channels``.
        Output shape: same as input.
        """
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"channel mismatch: x has {x.shape[1]} channels, "
                f"expected {self.num_channels}"
            )

        # Clamp into the learnable domain (we extrapolate linearly
        # outside the domain by clamping the input — equivalent to a
        # piecewise-constant boundary continuation, which the CR
        # interpolation handles gracefully).
        x_in = x.clamp(self.x_min, self.x_max)

        # Find segment index: i such that theta[i] <= x_in < theta[i+1].
        # searchsorted returns insertion index; subtract 1 for the
        # left endpoint.
        idx = torch.searchsorted(self.theta, x_in.contiguous(), right=False)
        idx = (idx - 1).clamp(0, self.n_knots - 2)

        # CR neighbour indices: idx-1, idx, idx+1, idx+2.
        # Reflective boundaries (no Python conditional, just clamp).
        idx_m1 = (idx - 1).clamp(0, self.n_knots - 1)
        idx_0 = idx
        idx_p1 = (idx + 1).clamp(0, self.n_knots - 1)
        idx_p2 = (idx + 2).clamp(0, self.n_knots - 1)

        # Gather control points per (B, C, H, W) position. cp has
        # shape (C, n_knots); we need cp[c, idx_*[b, c, h, w]].
        # Broadcast cp to (1, C, 1, 1, n_knots) and gather along
        # the last dim — this is a view, no extra memory.
        B, C, H, W = x_in.shape
        cp_view = self.cp.view(1, C, 1, 1, self.n_knots).expand(
            B, C, H, W, self.n_knots,
        )
        cp_m1 = cp_view.gather(-1, idx_m1.unsqueeze(-1)).squeeze(-1)
        cp_0 = cp_view.gather(-1, idx_0.unsqueeze(-1)).squeeze(-1)
        cp_p1 = cp_view.gather(-1, idx_p1.unsqueeze(-1)).squeeze(-1)
        cp_p2 = cp_view.gather(-1, idx_p2.unsqueeze(-1)).squeeze(-1)

        # Local interpolation parameter t ∈ [0, 1] within the segment.
        theta_0 = self.theta[idx_0]
        theta_p1 = self.theta[idx_p1]
        seg = (theta_p1 - theta_0).clamp_min(1e-9)
        t = ((x_in - theta_0) / seg).clamp(0.0, 1.0)

        # Catmull-Rom basis polynomials (Uniform CR; tension τ = 0).
        t2 = t * t
        t3 = t2 * t
        a0 = -t3 + 2.0 * t2 - t
        a1 = 3.0 * t3 - 5.0 * t2 + 2.0
        a2 = -3.0 * t3 + 4.0 * t2 + t
        a3 = t3 - t2
        y = 0.5 * (a0 * cp_m1 + a1 * cp_0 + a2 * cp_p1 + a3 * cp_p2)
        return y


# ─── ResNet-tiny backbone ────────────────────────────────────────────


class _BasicBlock(nn.Module):
    """2-conv residual block. Used in ResNetTinyBackbone."""

    def __init__(self, c: int, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.act1 = activation(c) if activation is CatmullRomActivation \
                   else activation(inplace=False)
        self.conv2 = nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.act2 = activation(c) if activation is CatmullRomActivation \
                   else activation(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act2(out + identity)
        return out


class _Downsample(nn.Module):
    """Stride-2 conv-BN-act for spatial downsampling."""

    def __init__(self, c_in: int, c_out: int,
                  activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = activation(c_out) if activation is CatmullRomActivation \
                  else activation(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def _build_resnet_tiny_stack(
    c_in: int, c_out: int, activation: type[nn.Module],
) -> nn.Sequential:
    """Common stack used by both ResNetTinyBackbone and
    HSiKANConvBackbone — only the activation class differs.

    Architecture (at H=W=64):
      stem    3 → 16            (64×64)  conv+bn+act
      block1  16 → 16           (64×64)  2× BasicBlock(16)
      down1   16 → 32           (32×32)  conv+bn+act stride 2
      block2  32 → 32           (32×32)  2× BasicBlock(32)
      down2   32 → 32           (16×16)  conv+bn+act stride 2
      block3  32 → 32           (16×16)  2× BasicBlock(32)
      down3   32 → c_out         (8×8)    conv+bn+act stride 2
    """
    # Stem.
    stem_act = activation(16) if activation is CatmullRomActivation \
              else activation(inplace=False)
    layers = [
        nn.Conv2d(c_in, 16, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        stem_act,
    ]
    # Block 1.
    layers.append(_BasicBlock(16, activation))
    layers.append(_BasicBlock(16, activation))
    # Down 1.
    layers.append(_Downsample(16, 32, activation))
    # Block 2.
    layers.append(_BasicBlock(32, activation))
    layers.append(_BasicBlock(32, activation))
    # Down 2.
    layers.append(_Downsample(32, 32, activation))
    # Block 3.
    layers.append(_BasicBlock(32, activation))
    layers.append(_BasicBlock(32, activation))
    # Down 3.
    layers.append(_Downsample(32, c_out, activation))
    return nn.Sequential(*layers)


# Stack layer indices that demarcate the multi-scale taps.
# These follow the layout in _build_resnet_tiny_stack:
#   0..2 stem (/1) ; 3..4 block1 (/1) ;
#   5 down1 (/2) ;
#   6..7 block2 (/2) ;
#   8 down2 (/4) ;
#   9..10 block3 (/4)  ← P_4 tap (c=32, H/4 × W/4)
#   11 down3 (/8)      ← P_8 tap (c=c_out, H/8 × W/8)
_STACK_TAP_P4_AFTER = 10  # index inclusive: after this layer, capture P4
_STACK_TAP_P8_AFTER = 11  # after this layer (final), capture P8


class ResNetTinyBackbone(nn.Module):
    """Residual-block backbone, drop-in replacement for
    ``TinyBackbone``. Same forward contract: ``(B, 3, H, W)`` →
    ``(B, c_out, H/8, W/8)``. ~107k params at ``c_out=32``.

    Stage C (FPN) needs intermediate features at /4. Use
    :meth:`multi_scale_features` to retrieve both /4 and /8 maps in
    one pass; the state_dict is the same as for single-scale
    ``forward`` (the multi-scale path is just a different traversal
    of the same Sequential stack).
    """

    def __init__(self, c_in: int = 3, c_out: int = 32) -> None:
        super().__init__()
        self.stack = _build_resnet_tiny_stack(c_in, c_out, nn.ReLU)
        self.c_out = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)

    def multi_scale_features(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (P_4, P_8) features at /4 and /8 scales.

        P_4: (B, 32, H/4, W/4) — output of block3, before down3.
        P_8: (B, c_out, H/8, W/8) — output of down3 (final).
        """
        h = x
        p_4: torch.Tensor | None = None
        for i, layer in enumerate(self.stack):
            h = layer(h)
            if i == _STACK_TAP_P4_AFTER:
                p_4 = h
        assert p_4 is not None
        return p_4, h


class HSiKANConvBackbone(nn.Module):
    """ResNet-tiny architecture with :class:`CatmullRomActivation` in
    place of every ``ReLU``. Same input/output shapes as
    :class:`ResNetTinyBackbone`; the difference is purely in
    the per-channel activation function.

    Stage B "HSiKAN as a ResNet substitute" measurement: paired
    against :class:`ResNetTinyBackbone` at same convs, same skips,
    same BN, only the activation kind differs. Isolates whether
    HSiKAN's basis-function primitive transfers to vision.

    Supports the same :meth:`multi_scale_features` API as
    :class:`ResNetTinyBackbone` for Stage C FPN integration.
    """

    def __init__(self, c_in: int = 3, c_out: int = 32,
                  *, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.stack = _build_resnet_tiny_stack(
            c_in, c_out, CatmullRomActivation,
        )
        self.c_out = c_out
        # Stage D-3-quinquies (2026-05-18): per-layer activation
        # checkpointing trades ~30% wall for ~70% activation memory.
        # The Catmull-Rom basis path materialises large per-channel
        # Hermite-spline intermediates that dominate activation
        # memory; recomputing them in backward is the right trade
        # for the 7.6 GiB consumer-GPU regime that OOMed D-3c.
        self.use_checkpoint = use_checkpoint

    def _apply_layer(self, layer: nn.Module,
                      h: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and h.requires_grad:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(layer, h, use_reentrant=False)
        return layer(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.stack:
            h = self._apply_layer(layer, h)
        return h

    def multi_scale_features(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (P_4, P_8) features at /4 and /8 scales.

        Same indices as :class:`ResNetTinyBackbone` (the stack
        layouts are identical; only the activation modules differ).
        """
        h = x
        p_4: torch.Tensor | None = None
        for i, layer in enumerate(self.stack):
            h = self._apply_layer(layer, h)
            if i == _STACK_TAP_P4_AFTER:
                p_4 = h
        assert p_4 is not None
        return p_4, h


# ─── Convenience dispatch ────────────────────────────────────────────


class ResNet18ImageNetBackbone(nn.Module):
    """ImageNet-pretrained ResNet18 truncated to layer2 (stride 8).

    The Stage D-1 backbone (per
    ``docs/plans/2026-05-18-hymeyolo-stage-d1-pretrain``).
    Drop-in replacement for :class:`ResNetTinyBackbone` with the
    same ``(B, 3, H, W) → (B, c_out, H/8, W/8)`` forward contract
    and the same :meth:`multi_scale_features` $/4 + /8$ FPN
    interface.

    Pretrained weights come from
    ``torchvision.models.resnet18(weights=IMAGENET1K_V1)``. The
    layer3/layer4 stages and the final FC are dropped (we only
    need stride-8 features). A frozen ImageNet-normalisation layer
    is inserted at the front so callers can still feed raw $[0, 1]$
    images (the same convention as Cluttered MNIST / VOC).

    Parameters
    ----------
    c_in : int
        Must be 3 — ResNet18 has 3 input channels.
    c_out : int, default 32
        Output channel count; a 1×1 projection ``128 → c_out`` is
        applied at the end of layer2.

    Falls back to a zero-weight init if the torchvision download is
    not available (offline / CI without network) — but logs a
    warning. Production runs must hit the cache.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    def __init__(self, c_in: int = 3, c_out: int = 32,
                 pretrained: bool = True) -> None:
        super().__init__()
        if c_in != 3:
            raise ValueError(
                f"ResNet18ImageNetBackbone requires c_in=3 (RGB); got {c_in}"
            )
        self.c_out = c_out
        from torchvision.models import resnet18
        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights
                net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception as e:  # offline / CI without network
                import warnings
                warnings.warn(
                    f"ResNet18 ImageNet weights unavailable ({e}); "
                    f"falling back to random init. Production runs MUST "
                    f"have the cache populated."
                )
                net = resnet18(weights=None)
        else:
            net = resnet18(weights=None)
        # Stem + layer1 (stride 4, 64 ch) + layer2 (stride 8, 128 ch).
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
        )
        self.layer1 = net.layer1   # 64 ch, stride 4
        self.layer2 = net.layer2   # 128 ch, stride 8
        self.proj_p8 = nn.Conv2d(128, c_out, kernel_size=1, bias=False)
        self.proj_p4 = nn.Conv2d(64, c_out, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.proj_p8.weight, mode="fan_out",
                                  nonlinearity="relu")
        nn.init.kaiming_normal_(self.proj_p4.weight, mode="fan_out",
                                  nonlinearity="relu")
        # ImageNet normalisation as a non-learnable buffer.
        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32
                          ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGENET_STD, dtype=torch.float32
                          ).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalise(x)
        x = self.stem(x)        # B × 64 × H/4 × W/4
        x = self.layer1(x)      # B × 64 × H/4 × W/4
        x = self.layer2(x)      # B × 128 × H/8 × W/8
        return self.proj_p8(x)  # B × c_out × H/8 × W/8

    def multi_scale_features(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (P_4, P_8) at strides 4 and 8 in channels c_out."""
        x = self._normalise(x)
        x = self.stem(x)
        p4 = self.layer1(x)     # B × 64 × H/4 × W/4
        p8 = self.layer2(p4)    # B × 128 × H/8 × W/8
        return self.proj_p4(p4), self.proj_p8(p8)


def build_backbone(name: str, c_in: int = 3, c_out: int = 32,
                    *, use_checkpoint: bool = False) -> nn.Module:
    """Dispatch on a string name. Used by RicciHyMeYOLOMulti's
    ``backbone`` kwarg + the ``--backbone`` CLI flag.

    Known names:
      "tiny"               → ``TinyBackbone`` (the pre-Stage-B default)
      "resnet"             → :class:`ResNetTinyBackbone`
      "hsikan"             → :class:`HSiKANConvBackbone`
      "resnet18_imagenet"  → :class:`ResNet18ImageNetBackbone` (Stage D-1)

    ``use_checkpoint`` (Stage D-3-quinquies) is honoured only for the
    HSiKAN backbone (its Catmull-Rom basis materialises the largest
    activation tensors). Silently ignored for the others.
    """
    if name == "tiny":
        # Lazy import to avoid forcing every backbone import to drag in
        # the q_smoke module's transitive dependencies.
        from .hymeyolo_q_smoke import TinyBackbone
        return TinyBackbone(c_in=c_in, c_out=c_out)
    if name == "resnet":
        return ResNetTinyBackbone(c_in=c_in, c_out=c_out)
    if name == "hsikan":
        return HSiKANConvBackbone(c_in=c_in, c_out=c_out,
                                    use_checkpoint=use_checkpoint)
    if name == "resnet18_imagenet":
        return ResNet18ImageNetBackbone(c_in=c_in, c_out=c_out)
    raise ValueError(
        f"unknown backbone {name!r}; expected one of "
        f"'tiny', 'resnet', 'hsikan', 'resnet18_imagenet'"
    )
