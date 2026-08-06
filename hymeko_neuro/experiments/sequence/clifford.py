"""Clifford-algebra primitives for the Sequential HSiKAN + CliffordFIR
dual-path architecture.

Default algebra is Cl(2,0): basis {1, e₁, e₂, e₁₂} with e₁²=e₂²=+1
and e₁·e₂ = -e₂·e₁ = e₁₂. A multivector is a 4-component vector
(scalar, e₁, e₂, e₁₂) carried in the trailing dim of any tensor.

We treat each "multivector channel" as the last dim of size 4 and
support broadcasting in earlier dims (B, L, C, …). The geometric
product is implemented as a closed-form 4-component scalar
expression — no Clifford-table lookup — so it autograds cleanly
and fits any tensor shape.

Multiplication table for Cl(2,0):
                  1      e₁     e₂     e₁₂
        1     |   1     e₁     e₂     e₁₂
        e₁    |  e₁      1    e₁₂     e₂
        e₂    |  e₂    -e₁₂    1     -e₁
        e₁₂   |  e₁₂   -e₂    e₁    -1

So for a = (a0, a1, a2, a12), b = (b0, b1, b2, b12):

    (a·b)_scalar  = a0·b0 + a1·b1 + a2·b2 - a12·b12
    (a·b)_e1      = a0·b1 + a1·b0 - a2·b12 + a12·b2
    (a·b)_e2      = a0·b2 + a1·b12 + a2·b0 - a12·b1
    (a·b)_e12     = a0·b12 + a1·b2 - a2·b1 + a12·b0

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/.
"""
from __future__ import annotations

import torch


# Sentinel: the dimension we keep multivector components in.
# 4 = (scalar, e_1, e_2, e_12) for Cl(2, 0).
CL2_DIM = 4


def geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geometric product of two Cl(2,0) multivectors.

    Both inputs must have a trailing dim of 4 holding the components
    (scalar, e₁, e₂, e₁₂); leading dims broadcast.

    Returns a tensor of the broadcasted shape with trailing dim 4.
    """
    if a.shape[-1] != CL2_DIM or b.shape[-1] != CL2_DIM:
        raise ValueError(
            f"geometric_product expects trailing dim {CL2_DIM} for "
            f"Cl(2,0); got a.shape[-1]={a.shape[-1]}, b.shape[-1]={b.shape[-1]}"
        )
    a0, a1, a2, a12 = a.unbind(dim=-1)
    b0, b1, b2, b12 = b.unbind(dim=-1)
    out_scalar = a0 * b0 + a1 * b1 + a2 * b2 - a12 * b12
    out_e1     = a0 * b1 + a1 * b0 - a2 * b12 + a12 * b2
    out_e2     = a0 * b2 + a1 * b12 + a2 * b0 - a12 * b1
    out_e12    = a0 * b12 + a1 * b2 - a2 * b1 + a12 * b0
    return torch.stack([out_scalar, out_e1, out_e2, out_e12], dim=-1)


def multivector_norm(a: torch.Tensor) -> torch.Tensor:
    """Euclidean norm over the multivector components.

    Returns a tensor with the trailing 4-component dim collapsed. Used
    by the position router (the |x_t| feature). For Cl(2,0), the
    "Clifford norm" sqrt(<a~·a>_0) equals the Euclidean norm of the
    component vector under the (+,+,+,-) signature — but we want a
    fixed positive scalar for gating, so we use the plain L2 norm of
    the 4-vector. Documented choice.
    """
    if a.shape[-1] != CL2_DIM:
        raise ValueError(
            f"multivector_norm expects trailing dim {CL2_DIM}; got "
            f"{a.shape[-1]}"
        )
    return torch.linalg.norm(a, dim=-1)


def scalar_multivector(s: torch.Tensor) -> torch.Tensor:
    """Lift a scalar tensor s of shape (..., ) into a (..., 4)
    multivector (s, 0, 0, 0)."""
    zeros = torch.zeros_like(s)
    return torch.stack([s, zeros, zeros, zeros], dim=-1)
