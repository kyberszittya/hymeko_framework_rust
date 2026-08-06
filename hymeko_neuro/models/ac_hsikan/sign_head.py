"""SignHead — learned edge-sign computation for the position graph.

Inputs:  x ∈ ℝ^{B × L × d}
Outputs: s ∈ {-1, +1}^{B × L × L} (hard at eval) or its soft surrogate
         tanh(...) ∈ [-1, +1] (training).

Mechanism: a small bilinear projection
    score_{ij} = w_b^T (x_i ⊙ x_j) + linear(x_i, x_j)
followed by tanh(score / T). For hard-sign inference we use either
plain sign() or sign + STE (straight-through estimator):
    forward:   hard_sign(score)
    backward:  identity on tanh-surrogate gradient
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _STESign(torch.autograd.Function):
    """Straight-through-estimator hard-sign: forward = sign(x), backward = identity."""
    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_sign(x: torch.Tensor) -> torch.Tensor:
    return _STESign.apply(x)


class SignHead(nn.Module):
    """Bilinear edge-sign head.

    Parameters
    ----------
    d_model : int
        Input feature dimension.
    hidden : int
        Hidden bilinear rank (smaller = fewer params; default 16).
    temperature : float
        tanh sharpness; smaller -> sharper sign approximation
        (closer to hard).
    use_ste : bool
        If True, forward emits hard sign (+ STE backward); else
        soft tanh (training-friendly).
    """

    def __init__(
        self,
        d_model: int,
        hidden: int = 16,
        temperature: float = 1.0,
        use_ste: bool = False,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.temperature = float(temperature)
        self.use_ste = bool(use_ste)
        # Bilinear projection: W_q @ x ⊙ W_k @ x → scalar per (i, j)
        self.W_q = nn.Linear(d_model, hidden, bias=False)
        self.W_k = nn.Linear(d_model, hidden, bias=False)
        # Small linear correction on the sum (i, j)
        self.W_b = nn.Linear(2 * d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute (B, L, L) edge signs.

        Returns soft tanh in [-1, 1] when ``self.use_ste`` is False
        (training default); hard sign in {-1, +1} with STE backward
        when True.
        """
        if x.dim() != 3:
            raise ValueError(f"SignHead expects (B, L, d_model); got {x.shape}")
        B, L, d = x.shape

        q = self.W_q(x)     # (B, L, h)
        k = self.W_k(x)     # (B, L, h)
        # bilinear score: (B, L, L)
        bilinear = torch.einsum("bld,bmd->blm", q, k)

        # linear correction: (B, L, L)
        # cheap broadcast: f(x_i + x_j) via concat
        xi = x.unsqueeze(2).expand(B, L, L, d)
        xj = x.unsqueeze(1).expand(B, L, L, d)
        lin = self.W_b(torch.cat([xi, xj], dim=-1)).squeeze(-1)

        score = (bilinear + lin) / self.temperature
        soft = torch.tanh(score)
        if self.use_ste:
            return ste_sign(soft)
        return soft


class QuaternionSignHead(nn.Module):
    """Hamilton-product real-part scoring for edge signs.

    Mirrors ``hymeko_neuro/models/mixed_arity_signedkan/attention.py::
    _QuaternionAttentionM_e`` (graph context, 2026-05-08 production
    HSiKAN ``HSIKAN_ATTENTION_M_E=quaternion``) but applied to the
    sequence position-pair attention.

    Scoring: split the (W_q · x), (W_k · x) projection into
    ``n_quat = hidden / 4`` independent quaternions; compute the
    real part of the Hamilton product per quaternion and sum:

        score(i, j) = Σ_q ( q_a·k_a − q_b·k_b − q_c·k_c − q_d·k_d )

    The negative sign on the (i, j, k) imaginary components is the
    point: anti-aligned imaginary axes SUBTRACT from the score
    (vs scalar attention where every dimension contributes
    positively). On sequences, this gives an asymmetric scoring
    that mirrors natural language asymmetry (subject vs object,
    modifier vs head, negation scope direction).
    """

    def __init__(
        self,
        d_model: int,
        hidden: int = 16,
        temperature: float = 1.0,
        use_ste: bool = False,
    ) -> None:
        super().__init__()
        if hidden % 4 != 0:
            raise ValueError(
                f"QuaternionSignHead requires hidden % 4 == 0; got {hidden}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        self.temperature = float(temperature)
        self.use_ste = bool(use_ste)
        self.hidden = int(hidden)
        self.n_quat = hidden // 4
        self.W_q = nn.Linear(d_model, hidden, bias=False)
        self.W_k = nn.Linear(d_model, hidden, bias=False)
        # Init small so initial scores ~0 -> tanh ~0 -> soft sign ~0.
        with torch.no_grad():
            self.W_q.weight.mul_(0.1)
            self.W_k.weight.mul_(0.1)
        # 1 / sqrt(n_quat) scale -- analogue of 1 / sqrt(d_k) in cosine attn.
        self.scale = 1.0 / max(self.n_quat ** 0.5, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"QuaternionSignHead expects (B, L, d_model); got {x.shape}"
            )
        B, L, d = x.shape
        q = self.W_q(x).view(B, L, self.n_quat, 4)
        k = self.W_k(x).view(B, L, self.n_quat, 4)
        # Compute (B, L, L) via Hamilton-product real-part:
        #   per-quaternion: q_a·k_a − q_b·k_b − q_c·k_c − q_d·k_d
        #   summed across n_quat then × scale.
        # Vectorised: stack the 4 sign coefficients and dot.
        # signs: (1, 1, 1, 1, 4) — broadcast against quaternion axis.
        coeffs = torch.tensor(
            [1.0, -1.0, -1.0, -1.0], device=x.device, dtype=x.dtype
        )
        # q: (B, L, n_quat, 4); k: (B, L, n_quat, 4)
        # we want score[b, i, j] = sum_q sum_a coeffs[a] * q[b, i, q, a] * k[b, j, q, a]
        # Re-shape to (B, L, hidden) with the coeff sign baked in.
        q_signed = (q * coeffs).reshape(B, L, self.hidden)
        k_flat = k.reshape(B, L, self.hidden)
        scores = torch.einsum("bld,bmd->blm", q_signed, k_flat) * self.scale
        scores = scores / self.temperature
        soft = torch.tanh(scores)
        if self.use_ste:
            return ste_sign(soft)
        return soft


def build_sign_head(
    kind: str,
    d_model: int,
    hidden: int,
    temperature: float,
    use_ste: bool,
) -> nn.Module:
    """Factory: returns SignHead or QuaternionSignHead per ``kind``."""
    if kind == "bilinear":
        return SignHead(d_model, hidden, temperature, use_ste)
    if kind == "quaternion":
        return QuaternionSignHead(d_model, hidden, temperature, use_ste)
    raise ValueError(
        f"sign_head_kind must be 'bilinear' or 'quaternion'; got {kind!r}"
    )
