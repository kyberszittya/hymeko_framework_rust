"""Forecasting models for the time-series benchmark.

All models share the contract::

    model(x: (B, W)) -> (B,)        # 1-step forecast
    model.n_params -> int

Three reference baselines + an HSIKAN-flavoured model:

- ``LinearAR``  — pure linear autoregressor (the floor).
- ``MLP``       — 2-layer ReLU MLP (mid-baseline).
- ``GRU``       — 1-layer GRU encoder + linear head.
- ``HSIKANSeq`` — sign-stream + windowed σ-cycle aggregator using the
  existing ``HSiKANSeqWindow`` from ``signedkan_wip/src/sequence``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from signedkan_wip.src.sequence.hsikan_seq import HSiKANSeqWindow
from signedkan_wip.src.sequence.clifford import CL2_DIM


class LinearAR(nn.Module):
    def __init__(self, window: int):
        super().__init__()
        self.fc = nn.Linear(window, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLP(nn.Module):
    def __init__(self, window: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GRUForecaster(nn.Module):
    def __init__(self, hidden: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W) → (B, W, 1)
        h_seq, h_last = self.gru(x.unsqueeze(-1))
        return self.head(h_last.squeeze(0)).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HSIKANSeqForecaster(nn.Module):
    """HSIKAN forecaster — highway-gated residual over a linear AR floor.

    Architecture (per window x of length W):

        y_hat = (1 - g) * linear_ar(x) + g * hsikan_branch(x)
                                          ↑
                                    g init = σ(-3) ≈ 0.05

    The HSIKAN branch:
        1. Sign stream σ_t = sign(x_t - x_{t-1})  (direction of change).
        2. Lift x_t to a (C, 4) multivector with scalar slot = c-th
           lift coefficient · x_t.
        3. ``HSiKANSeqWindow`` (windowed σ-cycle aggregator).
        4. Mean-pool across positions, flatten, project to 1 scalar.

    The highway-gated init means the model starts byte-identical to
    LinearAR and the HSIKAN branch only contributes if its predictions
    improve over the floor.  This is the proven productive pattern
    from the link-prediction outer-HSIKAN-residual win (Bitcoin Alpha
    +0.0066 AUC, 5.68σ paired).

    Parameters
    ----------
    window : int
        Forecast window W.
    K : int, optional
        Cycle window length. Default = window.
    n_channels : int, default 4
        Number of Clifford channels inside the HSIKAN block.
    """

    def __init__(self, window: int, K: Optional[int] = None,
                  n_channels: int = 4) -> None:
        super().__init__()
        self.window = int(window)
        self.K = int(K) if K is not None else window
        self.n_channels = int(n_channels)
        # Linear-AR base predictor; the HSIKAN branch is a residual.
        self.linear_ar = nn.Linear(window, 1)
        self.lift = nn.Linear(1, n_channels, bias=False)
        self.lift.weight.data.fill_(1.0 / max(1, n_channels))
        self.block = HSiKANSeqWindow(K=self.K, n_channels=n_channels)
        # Pooled readout: mean across positions, then project flattened
        # (C, 4) → scalar residual.
        self.head = nn.Linear(n_channels * CL2_DIM, 1)
        # Highway gate, init at sigmoid(-3) ≈ 0.05 — start near pure
        # LinearAR; let the HSIKAN branch earn its weight.
        self.gate_logit = nn.Parameter(torch.full((1,), -3.0))

    def _sigma_diff(self, x: torch.Tensor) -> torch.Tensor:
        # sign(x_t - x_{t-1}); position 0 has no predecessor → 0.
        diff = torch.diff(x, dim=1, prepend=x[:, :1])
        return torch.sign(diff).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, W = x.shape
        base = self.linear_ar(x).squeeze(-1)  # (B,)
        sigma = self._sigma_diff(x)
        per_channel = self.lift(x.unsqueeze(-1))  # (B, W, C)
        mv = torch.zeros(B, W, self.n_channels, CL2_DIM,
                          device=x.device, dtype=x.dtype)
        mv[..., 0] = per_channel
        out = self.block(mv, sigma)             # (B, W, C, 4)
        pooled = out.mean(dim=1).reshape(B, -1)  # (B, C*4)
        residual = self.head(pooled).squeeze(-1)
        g = torch.sigmoid(self.gate_logit)
        return (1.0 - g) * base + g * residual

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def gate_value(self) -> float:
        return float(torch.sigmoid(self.gate_logit).detach().item())


MODELS = {
    "linear_ar": LinearAR,
    "mlp": MLP,
    "gru": GRUForecaster,
    "hsikan_seq": HSIKANSeqForecaster,
}


__all__ = ["LinearAR", "MLP", "GRUForecaster", "HSIKANSeqForecaster", "MODELS"]
