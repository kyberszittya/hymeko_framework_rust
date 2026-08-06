"""DualPathSeqBlock — per-position routing between CliffordFIR (signal
branch) and HSiKANSeqWindow (information branch).

Position gate:
    g_t = sigmoid(W_g · [ |x_t|, |σ_t|, LocalSpec_t ])

LocalSpec_t is a 4-bin energy of the multivector-norm over a small
context (positions t-2..t+2) → 4 scalars. With |x_t| (1) + |σ_t| (1)
= 6 features fed to a linear → 1 head.

Output:
    y_t = g_t · y^sig_t + (1 - g_t) · y^info_t
    σ_t passed through unchanged (the network does not modify the
    sign stream layer-to-layer in v1 — flagged as a v2 lever in the
    plan).

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/ §3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clifford import CL2_DIM, multivector_norm
from .clifford_fir import CliffordFIR
from .hsikan_seq import HSiKANSeqWindow


def _local_spectrum(x_norm: torch.Tensor, n_bins: int = 4) -> torch.Tensor:
    """Cheap local "spectral" features per position: causal moving-
    average energies at K-power-of-two windows.

    For each bin k=0..n_bins-1, compute the mean of |x_norm| over the
    last 2^k positions (causal). Returns (B, L, n_bins).

    Not a DCT; a simple temporal-pyramid feature. Plan §3 calls for
    a 4-bin DCT; this is a faster CPU-friendly substitute that the
    router can still learn against.
    """
    B, L = x_norm.shape
    out = []
    for k in range(n_bins):
        win = 2 ** k
        if win == 1:
            avg = x_norm
        else:
            pad = torch.zeros(B, win - 1, dtype=x_norm.dtype, device=x_norm.device)
            xp = torch.cat([pad, x_norm], dim=1)            # (B, L+win-1)
            # Causal moving average.
            kernel = torch.full((1, 1, win), 1.0 / win,
                                dtype=x_norm.dtype, device=x_norm.device)
            avg = F.conv1d(xp.unsqueeze(1), kernel, padding=0).squeeze(1)  # (B, L)
        out.append(avg)
    return torch.stack(out, dim=-1)                          # (B, L, n_bins)


class PositionRouter(nn.Module):
    """Per-position sigmoid gate. Feature vector per position is
    (mean-channel-||x_t||, |σ_t|, local_spec_t[0..3]) — 6 scalars.

    v2: accepts both (B, L, 4) [single channel] and (B, L, C, 4)
    [multi-channel] inputs. For multi-channel input, the
    ||x_t|| feature is averaged across channels.
    """

    N_LOCAL_BINS = 4

    def __init__(self) -> None:
        super().__init__()
        n_features = 2 + self.N_LOCAL_BINS
        self.proj = nn.Linear(n_features, 1)
        # Bias init to 0 (gate=0.5 at start), weights to small Gaussian.
        nn.init.normal_(self.proj.weight, std=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, L, 4) or (B, L, C, 4); σ: (B, L); returns g: (B, L)."""
        if x.dim() == 4:
            # (B, L, C, 4) → channel-mean norm
            x_norm = multivector_norm(x).mean(dim=-1)        # (B, L)
        else:
            x_norm = multivector_norm(x)                     # (B, L)
        sigma_mag = sigma.abs()                              # (B, L)
        local = _local_spectrum(x_norm, n_bins=self.N_LOCAL_BINS)
        feats = torch.cat([
            x_norm.unsqueeze(-1),
            sigma_mag.unsqueeze(-1),
            local,
        ], dim=-1)                                            # (B, L, 6)
        g = torch.sigmoid(self.proj(feats).squeeze(-1))      # (B, L)
        return g


class DualPathSeqBlock(nn.Module):
    """A single dual-path sequential block: CliffordFIR (signal) +
    HSiKANSeqWindow (info) + PositionRouter mix.

    Parameters
    ----------
    K : int
        Window length for both branches. Default 4 (plan §5 default).
    n_channels : int, default 1
        Multivector channel count. At ``n_channels=1`` the block is
        byte-identical (in the v2 channel_mixer-at-identity init) to
        v1 except for the small additional channel-mixer params.

    Input / output:
        x : (B, L, 4) when ``n_channels=1`` or (B, L, C, 4) otherwise.
        σ : (B, L)    sign stream (unchanged through this block)
        returns (y, σ_out) with y matching x's shape and σ_out == σ.
    """

    def __init__(self, K: int = 4, n_channels: int = 1) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.fir = CliffordFIR(K=K, c_in=n_channels, c_out=n_channels)
        self.hsi = HSiKANSeqWindow(K=K, n_channels=n_channels)
        self.router = PositionRouter()
        self.K = K

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_sig = self.fir(x)                                  # matches x's shape
        y_info = self.hsi(x, sigma)
        g = self.router(x, sigma)                            # (B, L)
        # Reshape g to broadcast over channel + multivector dims.
        if x.dim() == 4:
            g_e = g.unsqueeze(-1).unsqueeze(-1)              # (B, L, 1, 1)
        else:
            g_e = g.unsqueeze(-1)                            # (B, L, 1)
        y = g_e * y_sig + (1.0 - g_e) * y_info
        return y, sigma

    def gate_load_balance_loss(
        self, x: torch.Tensor, sigma: torch.Tensor,
        target: float = 0.5,
    ) -> torch.Tensor:
        """Auxiliary load-balancing loss to keep mean gate near `target`.

        Standard MoE trick (plan §11 risk mitigation): prevents the
        router from collapsing to g_t ≡ 0 or g_t ≡ 1.
        """
        g = self.router(x, sigma)
        return (g.mean() - target).pow(2)
