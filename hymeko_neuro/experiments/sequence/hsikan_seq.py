"""HSiKANSeqWindow — windowed σ-cycle aggregator for sequences.

Lifts the HSiKAN signed-graph cycle primitive to a 1D sequence: a
length-K sliding window over positions {t-K+1, …, t} is a path graph
of length K, and the per-position sign labels σ_{t-k} ∈ {-1, 0, +1}
define a "cycle σ-product"

    π_t = Π_{k=0}^{K-1} σ_{t-k}    with σ=0 treated as +1 (no-sign)

The aggregator pools the K multivector samples into a per-sign mean
(positives and negatives separately, per channel), projects each
pool by a learnable Cl(2,0) tap, then mixes channels via a
multivector channel-mixer.

v2 (2026-05-17): multi-channel support via ``n_channels`` kwarg.
At ``n_channels=1`` the path is byte-identical to v1 (the channel
mixer is identity-initialised and trivially passes the single
channel through).

Plan: docs/plans/2026-05-17-sequence-multichannel-v2/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .clifford import CL2_DIM, geometric_product


class HSiKANSeqWindow(nn.Module):
    """Windowed σ-cycle aggregator over a 1D multivector sequence.

    Parameters
    ----------
    K : int
        Window length. ``π_t = Π_{k=0}^{K-1} σ_{t-k}``.
    n_channels : int, default 1
        Number of multivector channels in the input/output stream.
        When ``n_channels = 1`` the layer is byte-identical to v1.
    init_scale : float, default 0.1
        Std for Gaussian init of the per-sign output taps.

    Input shapes:
        x : (B, L, 4) when ``n_channels=1`` or (B, L, C, 4) otherwise.
        σ : (B, L)    sign stream in {-1, 0, +1}; one sign per token
                       (independent of channel; v2 design choice).
    Output shape:
        same as input.
    """

    def __init__(
        self, K: int,
        n_channels: int = 1,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")
        if n_channels < 1:
            raise ValueError(f"n_channels must be >= 1; got {n_channels}")
        self.K = int(K)
        self.n_channels = int(n_channels)
        # Per-channel per-sign output taps. Shape (C, 4) each; channel
        # index broadcasts at the geometric_product step. Init: scalar-
        # identity (1, 0, 0, 0) + small Gaussian noise on the non-scalar
        # components.
        tap_pos = torch.randn(self.n_channels, CL2_DIM) * float(init_scale)
        tap_pos[:, 0] = 1.0
        tap_neg = torch.randn(self.n_channels, CL2_DIM) * float(init_scale)
        tap_neg[:, 0] = 1.0
        self.tap_pos = nn.Parameter(tap_pos)
        self.tap_neg = nn.Parameter(tap_neg)
        # Channel mixer: (c_out, c_in, 4) multivector per channel-pair.
        # The geometric-product mixer routes information between
        # channels. Init: identity-scalar on the diagonal, near-zero
        # elsewhere → at init, output channel c equals input channel c
        # (no cross-channel mixing).
        mixer = torch.zeros(self.n_channels, self.n_channels, CL2_DIM)
        for c in range(self.n_channels):
            mixer[c, c, 0] = 1.0
        self.channel_mixer = nn.Parameter(mixer)
        # Parity-gate embedding: 3 learnable scalars indexed by π ∈ {-1, 0, +1}.
        # Order is [neg, zero, pos] (index 0, 1, 2). Initialised to identity
        # (1.0) so the gate at init is multiplicatively neutral.
        self.parity_gate = nn.Parameter(torch.ones(3))

    def _window_unfold(self, x: torch.Tensor) -> torch.Tensor:
        """Left-pad zero + unfold along L → windows of size K.

        Input (B, L, ...trailing) → output (B, L, K, ...trailing).
        Convention: windows[..., 0, :] = x_t (most recent).
        """
        pad_shape = list(x.shape)
        pad_shape[1] = self.K - 1
        pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
        x_pad = torch.cat([pad, x], dim=1)
        windows = x_pad.unfold(dimension=1, size=self.K, step=1)
        # Move the new K-axis (currently at the end) to position 2.
        if windows.dim() == 4:
            # (B, L, 4, K) → (B, L, K, 4)
            windows = windows.permute(0, 1, 3, 2).contiguous()
        elif windows.dim() == 5:
            # (B, L, C, 4, K) → (B, L, K, C, 4)
            windows = windows.permute(0, 1, 4, 2, 3).contiguous()
        elif windows.dim() == 3:
            # (B, L, K) — sigma stream, already correct
            pass
        else:
            raise ValueError(f"unfold produced unexpected shape {windows.shape}")
        # Flip K so index 0 = most recent.
        windows = windows.flip(dims=(2,) if windows.dim() >= 4 else (-1,))
        return windows

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor,
    ) -> torch.Tensor:
        squeezed = False
        if x.dim() == 3:
            if self.n_channels != 1:
                raise ValueError(
                    f"input is (B, L, 4) but n_channels={self.n_channels}; "
                    f"call with shape (B, L, C, 4)"
                )
            x = x.unsqueeze(2)  # (B, L, 1, 4)
            squeezed = True
        if x.dim() != 4 or x.shape[-1] != CL2_DIM:
            raise ValueError(
                f"x must be (B, L, C, 4); got {tuple(x.shape)}"
            )
        if sigma.dim() != 2 or sigma.shape != x.shape[:2]:
            raise ValueError(
                f"sigma must be (B, L) matching x's leading dims; "
                f"got {tuple(sigma.shape)} for x={tuple(x.shape)}"
            )
        B, L, C, _ = x.shape
        if C != self.n_channels:
            raise ValueError(
                f"input has {C} channels; layer expects {self.n_channels}"
            )
        # Window the multivector stream and sign stream identically.
        x_windows = self._window_unfold(x)        # (B, L, K, C, 4)
        sig_windows = self._window_unfold(sigma)  # (B, L, K)

        # σ-masked pooling: per-token mask, broadcast across channels.
        pos_mask = (sig_windows > 0.5).float()    # (B, L, K)
        neg_mask = (sig_windows < -0.5).float()
        cnt_pos = pos_mask.sum(dim=2).clamp(min=1.0)  # (B, L)
        cnt_neg = neg_mask.sum(dim=2).clamp(min=1.0)
        # Broadcast over channel + multivector dims at pool step.
        pos_e = pos_mask.unsqueeze(-1).unsqueeze(-1)  # (B, L, K, 1, 1)
        neg_e = neg_mask.unsqueeze(-1).unsqueeze(-1)
        cnt_pos_e = cnt_pos.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
        cnt_neg_e = cnt_neg.unsqueeze(-1).unsqueeze(-1)
        a_pos = (x_windows * pos_e).sum(dim=2) / cnt_pos_e  # (B, L, C, 4)
        a_neg = (x_windows * neg_e).sum(dim=2) / cnt_neg_e
        # Per-channel per-sign output projection.
        tap_pos_e = self.tap_pos.view(1, 1, self.n_channels, CL2_DIM).expand_as(a_pos)
        tap_neg_e = self.tap_neg.view(1, 1, self.n_channels, CL2_DIM).expand_as(a_neg)
        y_pos = geometric_product(tap_pos_e, a_pos)  # (B, L, C, 4)
        y_neg = geometric_product(tap_neg_e, a_neg)
        y_per_chan = y_pos + y_neg                    # (B, L, C, 4)

        # Channel mixer: (c_out, c_in, 4) ⊗ (B, L, c_in, 4)
        # y_mixed[b, l, c_out] = Σ_{c_in} mixer[c_out, c_in] ⊗ y_per_chan[b, l, c_in]
        # Implementation: broadcast both, geometric_product, sum over c_in.
        y_in_e = y_per_chan.unsqueeze(2)              # (B, L, 1, C_in, 4)
        mix_e = self.channel_mixer.view(1, 1, self.n_channels, self.n_channels, CL2_DIM)
        prods = geometric_product(mix_e, y_in_e)      # (B, L, C_out, C_in, 4)
        y_mixed = prods.sum(dim=3)                    # (B, L, C_out, 4)

        # Parity gate: π_t = Π σ_{t-k}.
        sig_nonzero = torch.where(
            sig_windows.abs() > 0.5, sig_windows,
            torch.ones_like(sig_windows),
        )
        parity = sig_nonzero.sign().prod(dim=2)       # (B, L)
        gate_idx = (parity.long() + 1).clamp(0, 2)
        gate = self.parity_gate[gate_idx]             # (B, L)
        y = y_mixed * gate.unsqueeze(-1).unsqueeze(-1)  # (B, L, C, 4)

        if squeezed:
            y = y.squeeze(2)                          # (B, L, 4)
        return y

    def receptive_field(self) -> int:
        return self.K
