"""FFNBlock — post-attention feed-forward variants.

* IdentityFFN  -- no FFN (back-compat).
* StandardFFN  -- transformer-style ``d -> ff_hidden -> d`` with
                  activation, dropout, residual + LayerNorm.
* BottleneckFFN -- speedup #3: smaller ratio (e.g. 1x or 2x instead
                   of the transformer-standard 4x) to cut FLOPS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..config import AcHsikanConfig


class FFNBlock(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class IdentityFFN(FFNBlock):
    def forward(self, x): return x


class StandardFFN(FFNBlock):
    def __init__(self, d_model: int, ff_hidden: int, dropout: float = 0.1):
        super().__init__()
        if ff_hidden < 1:
            raise ValueError(f"ff_hidden must be >= 1; got {ff_hidden}")
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(self.mlp(x)))


class BottleneckFFN(FFNBlock):
    """Lower-expansion FFN: ``ff_hidden = max(1, d_model // ratio)``.

    Typical use: ``ratio = 1`` (no expansion, identity-ish) or
    ``ratio = 2`` (half-width hidden). At ``ratio=1`` the block still
    has a GELU non-linearity which can help, but the parameter count
    is much smaller than the transformer-standard 4× expansion.
    """
    def __init__(self, d_model: int, ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        if ratio < 1:
            raise ValueError(f"BottleneckFFN ratio must be >= 1; got {ratio}")
        bottleneck = max(1, d_model // ratio)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(self.mlp(x)))


def build_ffn_block(cfg: "AcHsikanConfig") -> FFNBlock:
    bottleneck_ratio = getattr(cfg, "ffn_bottleneck_ratio", 0)
    if bottleneck_ratio > 0:
        return BottleneckFFN(
            d_model=cfg.d_model, ratio=bottleneck_ratio, dropout=cfg.dropout,
        )
    if cfg.ffn_hidden > 0:
        return StandardFFN(
            d_model=cfg.d_model, ff_hidden=cfg.ffn_hidden, dropout=cfg.dropout,
        )
    return IdentityFFN()
