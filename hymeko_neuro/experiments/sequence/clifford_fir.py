"""CliffordFIR — finite-impulse-response filter with multivector taps.

For a 1D multivector sequence x_t ∈ Cl(2,0) (shape (B, L, 4)), the
output at position t is

    y_t = Σ_{k=0}^{K-1} b_k ⊗ x_{t-k}     (geometric product)

with b_k ∈ Cl(2,0) learnable multivector taps and zero-padding at
positions t-k < 0 (causal FIR; output at t depends on inputs at
≤ t). Parameter count: K × 4 scalar coefficients per channel.

Multi-channel variant: input (B, L, C_in, 4), output (B, L, C_out, 4)
with filter (C_out, C_in, K, 4). The output channel c is

    y_t[c] = Σ_{c'} Σ_k b_k[c, c'] ⊗ x_{t-k}[c']

For the dual-path block's v1, C_in = C_out = 1 (a single multivector
stream) — the multi-channel API is documented but not used yet.

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/ §3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clifford import CL2_DIM, geometric_product


class CliffordFIR(nn.Module):
    """Causal multivector-tap FIR filter over a 1D sequence.

    Parameters
    ----------
    K : int
        Filter length (number of taps). Output at position t depends
        on inputs at positions {t, t-1, …, t-K+1}.
    c_in : int, default 1
        Number of input multivector channels. Each channel is itself
        a 4-vector in Cl(2,0).
    c_out : int, default 1
        Number of output multivector channels.
    init_scale : float, default 0.1
        Standard deviation for Gaussian init of the taps. Kept small
        so the filter starts near-identity-like.

    Input shape:  (B, L, c_in, 4)   or   (B, L, 4) when c_in == 1
    Output shape: (B, L, c_out, 4)  or   (B, L, 4) when c_out == 1
    """

    def __init__(
        self, K: int, c_in: int = 1, c_out: int = 1,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")
        self.K = int(K)
        self.c_in = int(c_in)
        self.c_out = int(c_out)
        # Filter taps: (c_out, c_in, K, 4).
        taps = torch.randn(c_out, c_in, K, CL2_DIM) * float(init_scale)
        # Initialise the first tap of each (c_out=c_in) diagonal as
        # the identity multivector (1, 0, 0, 0), so the layer starts
        # close to "output = input" (helpful for deep stacks).
        if c_out == c_in:
            for c in range(c_out):
                taps[c, c, 0, 0] = 1.0
                taps[c, c, 0, 1:] = 0.0
        self.taps = nn.Parameter(taps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal FIR with multivector taps.

        Accepts (B, L, 4) when c_in == 1 == c_out (the dual-path
        single-channel case) and (B, L, c_in, 4) otherwise.
        """
        squeezed = False
        if x.dim() == 3:
            if self.c_in != 1:
                raise ValueError(
                    f"input is (B, L, 4) but c_in={self.c_in}; "
                    f"call with shape (B, L, c_in, 4)"
                )
            x = x.unsqueeze(2)
            squeezed = True
        if x.dim() != 4 or x.shape[-1] != CL2_DIM:
            raise ValueError(
                f"expected (B, L, c_in, 4); got {tuple(x.shape)}"
            )
        B, L, C_in, _ = x.shape
        if C_in != self.c_in:
            raise ValueError(
                f"input has c_in={C_in} channels; layer expects "
                f"{self.c_in}"
            )
        # Left-pad with K-1 zeros along L so output at t includes
        # zeros for t-k < 0 (causal).
        pad = torch.zeros(
            B, self.K - 1, C_in, CL2_DIM,
            dtype=x.dtype, device=x.device,
        )
        x_pad = torch.cat([pad, x], dim=1)            # (B, L+K-1, C_in, 4)
        # Unfold along the L dim into windows of size K, oldest first.
        # After unfold the new dim ordering is (B, L, C_in, 4, K),
        # where window[..., k] corresponds to input position t-k+K-1.
        windows = x_pad.unfold(dimension=1, size=self.K, step=1)
        # windows: (B, L, C_in, 4, K). Permute the K dim before the 4-dim
        # to match the tap layout (c_out, c_in, K, 4).
        windows = windows.permute(0, 1, 2, 4, 3).contiguous()
        # The unfold lays out windows with the OLDEST element first
        # at index 0 of the K-dim (sliding from start). Our filter's
        # tap b_0 multiplies the MOST RECENT element x_t per the
        # convention "y_t = Σ b_k ⊗ x_{t-k}". So flip the K-dim:
        windows = windows.flip(dims=(3,))             # now windows[..., k, :] = x_{t-k}
        # windows: (B, L, C_in, K, 4). Broadcast against taps (c_out, c_in, K, 4):
        # expand to (B, L, c_out, c_in, K, 4) and apply geometric_product.
        windows_e = windows.unsqueeze(2)               # (B, L, 1, c_in, K, 4)
        taps_e = self.taps.view(1, 1, self.c_out, self.c_in, self.K, CL2_DIM)
        prods = geometric_product(taps_e, windows_e)   # (B, L, c_out, c_in, K, 4)
        # Sum over c_in and K to get the per-c_out output.
        y = prods.sum(dim=(3, 4))                       # (B, L, c_out, 4)
        if squeezed and self.c_out == 1:
            y = y.squeeze(2)                            # back to (B, L, 4)
        return y

    def receptive_field(self) -> int:
        """Number of input positions that influence each output position."""
        return self.K
