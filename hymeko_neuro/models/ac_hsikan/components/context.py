"""ContextEncoder — pre-attention contextualisation.

* IdentityContext     -- raw embedding (v1 default).
* CliffordFIRContext  -- causal multivector FIR filter from the Gömb
                          architecture (hymeko_neuro/experiments/sequence/
                          clifford_fir.py). Provides each position with
                          a window of geometric-product mixed history.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..config import AcHsikanConfig


class ContextEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class IdentityContext(ContextEncoder):
    def forward(self, x): return x


class CliffordFIRContext(ContextEncoder):
    """Clifford-FIR pre-attention context. The synaptic-glial substrate.

    On CUDA the forward + backward use the fused Triton kernel from
    ``hymeko_neuro.kernels.triton_kernels.clifford_fir`` (4-5× forward,
    2-5× fwd+bwd vs PyTorch reference depending on L). On CPU falls
    back to the PyTorch reference implementation.
    """
    def __init__(self, d_model: int, K: int):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(
                f"CliffordFIRContext requires d_model % 4 == 0; "
                f"got {d_model}"
            )
        from hymeko_neuro.experiments.sequence.clifford_fir import CliffordFIR
        n_c = d_model // 4
        # Keep the nn.Module for parameter registration (taps); the
        # forward path dispatches on device below.
        self.fir = CliffordFIR(K=K, c_in=n_c, c_out=n_c)
        self.d_model = int(d_model)
        self.n_c = int(n_c)

    def forward(self, x):
        B, L, _ = x.shape
        x_mv = x.view(B, L, self.n_c, 4)
        if x.is_cuda:
            from hymeko_neuro.kernels.triton_kernels.clifford_fir import (
                clifford_fir_triton,
            )
            x_ctx = clifford_fir_triton(x_mv, self.fir.taps)
        else:
            x_ctx = self.fir(x_mv)
        x_ctx = x_ctx.view(B, L, self.d_model)
        return x + x_ctx   # residual: FIR learns a perturbation


def build_context_encoder(cfg: "AcHsikanConfig") -> ContextEncoder:
    if cfg.use_clifford_fir_context:
        return CliffordFIRContext(d_model=cfg.d_model, K=cfg.clifford_fir_K)
    return IdentityContext()
