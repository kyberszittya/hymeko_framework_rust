"""One FSR block. Residual geometry is selectable (Gömb sphere vs standard pre-norm)."""
from __future__ import annotations

import torch
import torch.nn as nn

from hymeko_lm.channel_mixer import HSiKANChannelMixer
from hymeko_lm.config import FSRConfig, ResidualMode
from hymeko_lm.sequence_mixer import FiberSpikeRotorMixer
from hymeko_lm.sphere import spherical_residual


class FSRBlock(nn.Module):
    """Fiber-Spike-Rotor sequence mix + HSiKAN channel mix.

    ``SPHERE``: each sublayer retracts the stream to the unit sphere (Gömb). ``PRENORM``: standard
    pre-LayerNorm additive residual (the control).

    # Preconditions ``h: (B,T,cfg.d_model)``; ``T <= cfg.max_seq_len``.
    # Postconditions output ``(B,T,cfg.d_model)``.
    """

    def __init__(self, cfg: FSRConfig) -> None:
        super().__init__()
        self.mode = cfg.residual_mode
        self.mixer = FiberSpikeRotorMixer(cfg.n_blocks, cfg.max_seq_len, cfg.gate_rank,
                                          cfg.gate_mode, cfg.spike_k)
        self.channel = HSiKANChannelMixer(cfg.d_model, cfg.channel_mult, cfg.activation, cfg.grid)
        if self.mode is ResidualMode.PRENORM:
            self.norm1 = nn.LayerNorm(cfg.d_model)
            self.norm2 = nn.LayerNorm(cfg.d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode is ResidualMode.SPHERE:
            h = spherical_residual(h, self.mixer(h))
            h = spherical_residual(h, self.channel(h))
        else:
            h = h + self.mixer(self.norm1(h))
            h = h + self.channel(self.norm2(h))
        return h
