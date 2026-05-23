"""Clifford-rotor positional encoding for Sequential HSiKAN text models.

In Cl(2,0), a rotation in the e_1-e_2 plane is implemented by a rotor

    R_t = cos(θ_t / 2) + sin(θ_t / 2) e_{12}

acting on a vector v ∈ {e_1, e_2}-plane by conjugation:

    v ↦ R_t v R_t^{-1}.

For a multivector channel x_t = (x_0, x_1, x_2, x_{12}) ∈ Cl(2,0),
the rotor action via conjugation rotates the (e_1, e_2) part by
θ_t while preserving the scalar and bivector parts. This is the
multivector generalisation of complex-valued rotary positional
encoding (RoPE; Su et al. 2024).

Properties:
  * Norm-preserving: ||R_t x_t R_t^{-1}|| = ||x_t||.
  * Relative-position aware: R_t R_s^{-1} = R_{t-s}.
  * O(1) per position per channel; no learnable parameters by
    default (θ_t is a fixed schedule).

Plan: docs/plans/2026-05-17-text-encoder-decoder-contest/.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .clifford import CL2_DIM


def _rotor_apply(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply the Clifford rotor R = cos(θ/2) + sin(θ/2) e_{12} to a
    multivector x = (x_0, x_1, x_2, x_{12}) via conjugation R x R^{-1}.

    For Cl(2,0), the conjugation rotates the (e_1, e_2) part by angle
    θ while leaving the scalar and bivector parts invariant. Closed
    form:

        x'_0    = x_0
        x'_1    =  cos(θ) x_1 + sin(θ) x_2
        x'_2    = -sin(θ) x_1 + cos(θ) x_2
        x'_{12} = x_{12}

    Parameters
    ----------
    x : (..., 4) tensor in Cl(2,0).
    theta : (...,) angle tensor, broadcastable to x's leading dims.

    Returns
    -------
    (..., 4) tensor with the rotated multivector.
    """
    if x.shape[-1] != CL2_DIM:
        raise ValueError(
            f"_rotor_apply: x trailing dim must be {CL2_DIM}; got {x.shape[-1]}"
        )
    c = torch.cos(theta)
    s = torch.sin(theta)
    x0, x1, x2, x12 = x.unbind(dim=-1)
    out1 =  c * x1 + s * x2
    out2 = -s * x1 + c * x2
    return torch.stack([x0, out1, out2, x12], dim=-1)


class CliffordRotorPositional(nn.Module):
    """Adds Clifford-rotor positional encoding to a multivector
    sequence.

    Parameters
    ----------
    max_len : int
        Maximum sequence length supported (pre-computed θ_t cache).
    n_channels : int, default 1
        Multivector channel count.
    base : float, default 10000.0
        Frequency base, in the spirit of sinusoidal positional
        encoding. The angle at position t for channel c is
        ``θ_{t, c} = t / base^{c / n_channels}``.
    learnable_base : bool, default False
        If True, ``base`` becomes a learnable scalar; otherwise it
        is fixed at construction.

    Input shape:  (B, L, C, 4) or (B, L, 4) when C=1.
    Output shape: same.
    """

    def __init__(
        self,
        max_len: int = 1024,
        n_channels: int = 1,
        base: float = 10000.0,
        learnable_base: bool = False,
    ) -> None:
        super().__init__()
        self.max_len = int(max_len)
        self.n_channels = int(n_channels)
        if learnable_base:
            self.log_base = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(base)).item())))
        else:
            self.register_buffer(
                "log_base", torch.tensor(float(torch.log(torch.tensor(base)).item())),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = False
        if x.dim() == 3:
            x = x.unsqueeze(2)
            squeezed = True
        if x.dim() != 4 or x.shape[-1] != CL2_DIM:
            raise ValueError(
                f"CliffordRotorPositional expects (B, L, C, 4); got {tuple(x.shape)}"
            )
        B, L, C, _ = x.shape
        if C != self.n_channels:
            raise ValueError(
                f"input channel dim {C} doesn't match n_channels={self.n_channels}"
            )
        if L > self.max_len:
            raise ValueError(
                f"sequence length {L} exceeds max_len={self.max_len}"
            )
        # Construct per-position, per-channel angle theta_{t, c}.
        base = torch.exp(self.log_base)
        t = torch.arange(L, dtype=x.dtype, device=x.device).unsqueeze(-1)
        # Per-channel scale: 1 / base^(c / C).
        c_idx = torch.arange(C, dtype=x.dtype, device=x.device).unsqueeze(0)
        scale = torch.pow(base, -c_idx / max(1, C))
        theta = t * scale  # (L, C)
        theta_e = theta.unsqueeze(0)  # (1, L, C)
        out = _rotor_apply(x, theta_e)
        if squeezed:
            out = out.squeeze(2)
        return out
