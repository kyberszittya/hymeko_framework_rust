"""DualPathSequenceModel — stacks L_depth DualPathSeqBlocks +
attention-pool + linear head for sequence classification/regression.

Input contract:
    raw : (B, L, F)   F input scalar features per position (e.g.,
                       a multivariate time-series)

Inside, the model
  1. Encodes raw → x ∈ Cl(2,0) multivector stream (B, L, 4) via a
     small linear projection.
  2. Encodes raw → σ ∈ {-1, 0, +1} sign stream (B, L) via a
     straight-through `sign(tanh(W_σ · raw))` head (the v1 default).
  3. Stacks `depth` DualPathSeqBlocks producing (B, L, 4) features.
  4. Pools across L with a learned attention pool → (B, 4).
  5. Linear → (B, n_classes).

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/ §3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clifford import CL2_DIM
from .dual_router import DualPathSeqBlock


class _STESign(torch.autograd.Function):
    """Straight-through estimator for `sign`. Forward is hard sign;
    backward passes the upstream gradient through unchanged."""

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return _STESign.apply(x)


class SeqInputEncoder(nn.Module):
    """Encodes raw features (B, L, F) into (multivector_stream, sign_stream).

    Both encoders are tiny linear maps; sign is rounded via a
    straight-through estimator so the layer is end-to-end trainable.
    """

    def __init__(self, in_features: int, supervised_sign: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.supervised_sign = supervised_sign
        # Multivector projection: F → 4 (Cl(2,0)).
        self.to_mv = nn.Linear(in_features, CL2_DIM)
        # Sign projection: F → 1.
        if not supervised_sign:
            self.to_sign = nn.Linear(in_features, 1)
            nn.init.normal_(self.to_sign.weight, std=0.5)

    def forward(
        self,
        raw: torch.Tensor,
        sigma_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if raw.dim() != 3 or raw.shape[-1] != self.in_features:
            raise ValueError(
                f"raw must be (B, L, {self.in_features}); got {tuple(raw.shape)}"
            )
        x = self.to_mv(raw)                                   # (B, L, 4)
        if self.supervised_sign:
            if sigma_override is None:
                raise ValueError(
                    "supervised_sign=True requires sigma_override"
                )
            sigma = sigma_override.to(dtype=x.dtype)
        else:
            sigma = ste_sign(torch.tanh(self.to_sign(raw).squeeze(-1)))
        return x, sigma


class AttentionPool(nn.Module):
    """Learned-attention pool over the L axis. Scores via Linear → softmax;
    output is the attention-weighted sum of the per-position multivectors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Linear(CL2_DIM, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 4) → (B, 4)"""
        logits = self.score(x).squeeze(-1)                    # (B, L)
        w = F.softmax(logits, dim=-1)                         # (B, L)
        return (x * w.unsqueeze(-1)).sum(dim=1)               # (B, 4)


class DualPathSequenceModel(nn.Module):
    """Stack of L_depth dual-path blocks + attention-pool + linear head.

    Parameters
    ----------
    in_features : int
        Per-position scalar count of the raw input.
    n_classes : int
        Output dim (binary classification = 2).
    depth : int, default 3
        Number of stacked DualPathSeqBlocks.
    K : int, default 4
        Window length within each block.
    supervised_sign : bool, default False
        If True, the model takes the sign stream from a `sigma_override`
        kwarg in forward() (used during synthetic-benchmark ground-truth
        supervision). If False, the input encoder learns the signs end-to-end.

    Parameter count at defaults (in_features=4, n_classes=2, depth=3, K=4):
      encoder (to_mv + to_sign): 4*4+4 + 4+1 = 25
      block * 3:
        CliffordFIR: 1*1*4*4 = 16
        HSiKANSeqWindow: 4+4+3 = 11
        PositionRouter: 6+1 = 7
        per-block: 34
      depth=3: 102
      attention pool: 4+1 = 5
      head: 4*n_classes + n_classes
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        depth: int = 3,
        K: int = 4,
        supervised_sign: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = SeqInputEncoder(in_features, supervised_sign=supervised_sign)
        self.blocks = nn.ModuleList([
            DualPathSeqBlock(K=K) for _ in range(depth)
        ])
        self.pool = AttentionPool()
        self.head = nn.Linear(CL2_DIM, n_classes)
        self.depth = depth
        self.K = K
        self.supervised_sign = supervised_sign

    def forward(
        self,
        raw: torch.Tensor,
        sigma_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, sigma = self.encoder(raw, sigma_override=sigma_override)
        for block in self.blocks:
            x, sigma = block(x, sigma)
        pooled = self.pool(x)                                 # (B, 4)
        logits = self.head(pooled)                            # (B, n_classes)
        return logits

    def gate_load_balance_loss(
        self,
        raw: torch.Tensor,
        sigma_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean of per-block gate-balance losses; add to the training
        objective with a small weight (plan §11)."""
        x, sigma = self.encoder(raw, sigma_override=sigma_override)
        losses = []
        for block in self.blocks:
            losses.append(block.gate_load_balance_loss(x, sigma))
            x, sigma = block(x, sigma)
        return torch.stack(losses).mean()
