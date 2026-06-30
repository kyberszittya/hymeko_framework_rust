"""HSiKAN channel mixer — replaces the transformer FFN.

A per-token KAN cell ``Linear -> edge activation -> Linear`` whose nonlinearity is the
canonical CR-Chebyshev cell from ``signed_kan`` (``make_activation('cr_cheby')``):
train-CR / deploy-Chebyshev, GPU-friendly. No new basis code (the cell already exists).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from signed_kan.splines import make_activation

from hymeko_lm.config import Activation


class HSiKANChannelMixer(nn.Module):
    """Per-token signed-KAN channel mix. ``(B,T,d) -> (B,T,d)``.

    # Preconditions ``d_model >= 1``, ``channel_mult >= 1``; ``grid >= 8`` for ``cr_cheby``.
    # Postconditions output shape equals input shape.
    """

    def __init__(self, d_model: int, channel_mult: int, activation: Activation, grid: int) -> None:
        super().__init__()
        hidden = d_model * channel_mult
        self.up = nn.Linear(d_model, hidden)
        self.act = make_activation(activation.value, hidden, grid=grid)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down(self.act(self.up(h)))
        return out
