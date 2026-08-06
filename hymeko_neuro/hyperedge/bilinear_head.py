"""Bilinear classifier head for SignedKAN edge prediction.

Adds a learnable bilinear form on the edge endpoints' node
embeddings, scoring an edge $e = (u, v)$ as
    $$\\text{score}_{\\mathrm{bil}}(e) = \\mathbf{h}_u^\\top \\mathbf{W} \\mathbf{h}_v + b$$
to the existing per-edge linear logit
    $$\\text{score}_{\\mathrm{lin}}(e) = \\mathbf{w}^\\top \\mathbf{h}_e + b'$$
where $\\mathbf{h}_e$ is the triad-mean-pooled edge embedding.

Total per-edge logit is the sum:
    $$\\text{logit}(e) = \\text{score}_{\\mathrm{lin}}(e) + \\text{score}_{\\mathrm{bil}}(e).$$

The linear head keeps the sign-aware triadic signal; the bilinear
head adds a direct pairwise endpoint coupling that the optimiser
can use to bypass the (lossy) mean-pool when it helps.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BilinearHead(nn.Module):
    """Full-rank bilinear edge scorer: $\\hat y = \\gamma \\, h_u^\\top W h_v + b$
    with a LayerScale gate $\\gamma$ initialised to ``gamma_init``.

    $W$ is Xavier-initialised; $\\gamma$ starts small (default $10^{-3}$)
    so the bilinear path is effectively dead at training start and the
    spline pathway trains unimpeded. The optimiser scales $\\gamma$ up
    only if the bilinear coupling actually helps.
    """

    def __init__(self, d: int, gamma_init: float = 1e-3):
        super().__init__()
        scale = 1.0 / max(d, 1) ** 0.5
        self.W = nn.Parameter(torch.randn(d, d) * scale)
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        """``h_u, h_v: (E, d) → out: (E,)``."""
        return self.gamma * ((h_u @ self.W) * h_v).sum(dim=-1) + self.bias


class LowRankBilinearHead(nn.Module):
    """Low-rank bilinear: $W = U V^\\top$, rank $r$; $r d$ params instead of $d^2$.

    $U$ initialised small ($1/\\sqrt{d}$ scaled), $V$ initialised to zero
    so $W = U V^\\top = 0$ at init — same dead-start trick as
    ``BilinearHead``.
    """

    def __init__(self, d: int, rank: int = 4,
                 init_scale: float | None = None,
                 gamma_init: float = 1e-3):
        super().__init__()
        scale = init_scale if init_scale is not None else 1.0 / max(d, 1) ** 0.5
        self.U = nn.Parameter(torch.randn(d, rank) * scale)
        self.V = nn.Parameter(torch.randn(d, rank) * scale)
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        # (E, d) @ (d, r) = (E, r); element-wise multiply, sum.
        return self.gamma * ((h_u @ self.U) * (h_v @ self.V)).sum(dim=-1) + self.bias


class ComplexDiagonalHead(nn.Module):
    r"""ComplEx-style head (Trouillon 2016): treat ``h`` as ``d/2`` complex pairs
    (first half = real, second = imaginary) and score with a learned **complex
    diagonal** weight $w$:
    $$\phi(u,v)=\mathrm{Re}\sum_k w_k\,u_k\,\bar v_k .$$
    The complex analogue of the real bilinear matched filter — phase-aware, and
    asymmetric iff $\mathrm{Im}(w)\neq0$ (so it can model directed/antisymmetric
    sign structure that DistMult cannot). Params $= d$ (a structured, algebra-
    shared factorisation of a full $d\times d$ bilinear).

    # Preconditions ``h_u, h_v: (E, d)`` with ``d`` even.
    # Postconditions returns ``(E,)``; dead-start ``gamma`` (off at init).
    """

    def __init__(self, d: int, gamma_init: float = 1e-3) -> None:
        super().__init__()
        if d % 2 != 0:
            raise ValueError(f"ComplexDiagonalHead needs even d; got {d}")
        self.d2 = d // 2
        scale = 1.0 / max(d, 1) ** 0.5
        self.w_re = nn.Parameter(torch.randn(self.d2) * scale)
        self.w_im = nn.Parameter(torch.randn(self.d2) * scale)
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        ur, ui = h_u[:, :self.d2], h_u[:, self.d2:]
        vr, vi = h_v[:, :self.d2], h_v[:, self.d2:]
        # Re(w · u · conj(v)) = w_re(ur·vr + ui·vi) − w_im(ui·vr − ur·vi).
        real = self.w_re * (ur * vr + ui * vi) - self.w_im * (ui * vr - ur * vi)
        return self.gamma * real.sum(dim=-1) + self.bias
