"""Edge activations (the KAN nonlinearity) for the signed-KAN core — a Strategy over spline kinds.

The Catmull-Rom spline is the ``CR`` in HSiKAN. This module is the single home for it (the RL backbone and,
after phase 3, the ``hymeko_neuro`` line both use it); a parity test pins it to ``hymeko_neuro``'s canonical
``_catmull_rom_eval``. B-spline / Kochanek-Bartels register here when the phase-3 migration lands.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def catmull_rom(coef: torch.Tensor, x: torch.Tensor, grid: int) -> torch.Tensor:
    """Uniform cubic Catmull-Rom interpolating spline on ``[-1, 1]`` (the KAN ``CR`` activation).

    **The single canonical Catmull-Rom evaluator** for the whole repo: ``hymeko_rl`` (via the re-export in
    ``policy.py``) and ``hymeko_neuro`` (which wraps this in its ``torch.compile`` fast-path) both call it.
    The gather form below is the ``hymeko_neuro`` canonical body verbatim — generalised over arbitrary leading
    dims on ``coef`` (so it serves both the RL ``(C, G)`` and the vision ``(..., C, G)`` coefficient layouts).

    # Preconditions ``coef`` is ``(..., C, grid)`` (or ``(C, grid)``) per-channel control points; ``x`` is
    ``(..., C)``; ``grid >= 4``.
    # Postconditions returns ``(..., C)``; ``x`` is clamped into ``[-1, 1]`` before evaluation.
    """
    g = grid
    x = x.clamp(min=-1.0, max=1.0)
    u = (x + 1.0) * 0.5 * (g - 1)
    i = u.floor().long().clamp(max=g - 2)
    t = u - i.to(x.dtype)
    t2 = t * t
    t3 = t2 * t
    w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
    w_0 = 0.5 * (3.0 * t3 - 5.0 * t2 + 2.0)
    w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
    w_p2 = 0.5 * (t3 - t2)
    # Four separate gathers (one per control point) — never materialise a ``(..., C, 4)`` int64 stack.
    idx_m1 = (i - 1).clamp(min=0, max=g - 1).unsqueeze(-1)
    idx_0 = i.unsqueeze(-1)
    idx_p1 = (i + 1).clamp(min=0, max=g - 1).unsqueeze(-1)
    idx_p2 = (i + 2).clamp(min=0, max=g - 1).unsqueeze(-1)
    coef_b = coef.expand(*x.shape, g)
    p_m1 = coef_b.gather(-1, idx_m1).squeeze(-1)
    p_0 = coef_b.gather(-1, idx_0).squeeze(-1)
    p_p1 = coef_b.gather(-1, idx_p1).squeeze(-1)
    p_p2 = coef_b.gather(-1, idx_p2).squeeze(-1)
    return w_m1 * p_m1 + w_0 * p_0 + w_p1 * p_p1 + w_p2 * p_p2


class EdgeActivation(nn.Module, ABC):
    """A learnable per-channel univariate edge function (the KAN activation). ``forward: (..., C) -> (..., C)``."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class CatmullRomActivation(EdgeActivation):
    """Learnable per-channel Catmull-Rom spline activation — the KAN nonlinearity (``CR``).

    ``grid`` control points per channel, ``C^1``-continuous and interpolatory. Input is softly bounded to
    ``[-1, 1]`` (the spline domain).

    # Preconditions ``n_channels >= 1``, ``grid >= 4`` (a cubic needs 4 knots).
    """

    def __init__(self, n_channels: int, *, grid: int = 5, init_scale: float = 0.1) -> None:
        super().__init__()
        if n_channels < 1 or grid < 4:
            raise ValueError(f"n_channels >= 1 and grid >= 4 required; got {n_channels}, {grid}")
        self.grid = grid
        self.coef = nn.Parameter(torch.randn(n_channels, grid) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return catmull_rom(self.coef, x, self.grid)


def make_uniform_knots(grid: int, k: int, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """Uniform knot vector on ``[lo, hi]`` padded with ``k`` extra knots each end (clamped extension).
    Verbatim from ``hymeko_neuro`` so the B-spline basis is identical."""
    inner = torch.linspace(lo, hi, grid)
    step = (hi - lo) / (grid - 1)
    left_pad = torch.tensor([lo - (k - i) * step for i in range(k)])
    right_pad = torch.tensor([hi + (i + 1) * step for i in range(k)])
    return torch.cat([left_pad, inner, right_pad])


def cox_de_boor(x: torch.Tensor, knots: torch.Tensor, k: int) -> torch.Tensor:
    """Cox--de Boor B-spline basis (vectorised). ``x: (B,)``, ``knots: (G+2k+1,)`` → ``(B, G+k-1)``.
    Verbatim from ``hymeko_neuro`` (the OTC benchmark's spline) so re-homing is bit-faithful."""
    x = x.unsqueeze(-1)
    left = knots[:-1]
    right = knots[1:]
    basis = ((x >= left) & (x < right)).to(x.dtype)
    last_mask = (x.squeeze(-1) == knots[-1])
    if last_mask.any():
        basis[last_mask, -1] = 1.0
    x_flat = x.squeeze(-1)
    for order in range(1, k + 1):
        n_basis_o = knots.shape[0] - 1 - order
        knots_l = knots[:n_basis_o]
        knots_lp = knots[1:1 + n_basis_o]
        knots_ro = knots[order:order + n_basis_o]
        knots_rop = knots[order + 1:order + 1 + n_basis_o]
        denom_left = (knots_ro - knots_l).clamp(min=1e-12)
        denom_right = (knots_rop - knots_lp).clamp(min=1e-12)
        x_b = x_flat.unsqueeze(-1)
        term_left = (x_b - knots_l) / denom_left * basis[..., :n_basis_o]
        term_right = (knots_rop - x_b) / denom_right * basis[..., 1:1 + n_basis_o]
        basis = term_left + term_right
    return basis


class BSplineActivation(EdgeActivation):
    """Learnable cubic B-spline activation with per-channel coefficients — the legacy vision-line spline,
    re-homed here so a single architecture can A/B Catmull-Rom vs B-spline. ``(B, C) -> (B, C)``.

    # Preconditions ``n_channels >= 1``, ``grid >= 1``, ``k >= 1``.
    """

    def __init__(self, n_channels: int, *, grid: int = 5, k: int = 3, init_scale: float = 0.1) -> None:
        super().__init__()
        if n_channels < 1 or grid < 1 or k < 1:
            raise ValueError(f"n_channels/grid/k must be >= 1; got {n_channels}/{grid}/{k}")
        self.grid = grid
        self.k = k
        self.register_buffer("knots", make_uniform_knots(grid, k), persistent=False)
        self.coef = nn.Parameter(torch.randn(n_channels, grid + k - 1) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=-1.0, max=1.0)
        shape = x.shape
        knots = self.knots
        assert isinstance(knots, torch.Tensor)   # registered buffer (invariant: set in __init__)
        basis = cox_de_boor(x.reshape(-1), knots, self.k).view(*shape, -1)   # (..., C, n_basis)
        out: torch.Tensor = (basis * self.coef).sum(dim=-1)
        return out


class ChebyshevCRActivation(EdgeActivation):
    """CR-Chebyshev (Catmull-Rom through a Chebyshev lens): a CR spline whose ``grid`` control points are generated
    by a learned degree-``(k-1)`` Chebyshev expansion through a CACHED knot-basis, ``P = coef · T_knots^{T}``
    (``T_knots`` = the Chebyshev polynomials evaluated at the fixed knots — a constant). The control points become a
    \\emph{band-limited} (smooth) family instead of free knots. Wins: ``k`` coefficients/channel (param-efficient
    when ``k < grid``), a smoother fit on smooth targets, latency parity with CR, and --- because the function lies
    on a degree-``k`` Chebyshev --- :meth:`chebyshev_forward` evaluates it via the cheap matmul (no gather; ~3x at
    ``B=1``) as a deploy-time fast path. The default :meth:`forward` is the CR gather (train quality).

    # Preconditions ``n_channels >= 1``, ``grid >= 4``, ``2 <= k <= grid`` (``k`` defaults to ``min(5, grid)``)."""

    t_knots: torch.Tensor

    def __init__(self, n_channels: int, *, grid: int = 12, k: int | None = None, init_scale: float = 0.1) -> None:
        super().__init__()
        k = min(5, grid) if k is None else k
        if n_channels < 1 or grid < 4 or not (2 <= k <= grid):
            raise ValueError(f"need n_channels>=1, grid>=4, 2<=k<=grid; got {n_channels}, {grid}, {k}")
        self.grid, self.k = grid, k
        knots = torch.linspace(-1.0, 1.0, grid)
        terms = [torch.ones_like(knots), knots]
        for _ in range(2, k):
            terms.append(2 * knots * terms[-1] - terms[-2])
        self.register_buffer("t_knots", torch.stack(terms[:k], dim=-1), persistent=False)   # (grid, k) cached
        self.coef = nn.Parameter(torch.randn(n_channels, k) * init_scale)                    # (C, k) Chebyshev coeffs
        self.deploy = False     # train-CR / deploy-Chebyshev: when True, forward() takes the fast Chebyshev path

    def control_points(self) -> torch.Tensor:
        """The band-limited spline control points ``(C, grid) = coef · T_knots^{T}``."""
        return self.coef @ self.t_knots.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:                                 # deploy fast path (cheap matmul, ~3x at B=1; approx CR)
            return self.chebyshev_forward(x)
        return catmull_rom(self.control_points(), x, self.grid)

    def chebyshev_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deploy-time fast path: the Chebyshev evaluated directly (no gather; ~3x at ``B=1``). An approximation of
        :meth:`forward` (CR interpolation of the Chebyshev-sampled points) — validate the tolerance before swapping."""
        x = x.clamp(min=-1.0, max=1.0)
        terms = [torch.ones_like(x), x]
        for _ in range(2, self.k):
            terms.append(2 * x * terms[-1] - terms[-2])
        tk = torch.stack(terms[:self.k], dim=-1)        # (..., C, k)
        return (tk * self.coef).sum(-1)


def set_deploy_mode(model: nn.Module, deploy: bool = True) -> int:
    """Switch every :class:`ChebyshevCRActivation` in ``model`` to its Chebyshev deploy fast path (or back to the
    CR train path). Returns the count switched. The deploy path *approximates* the CR forward — validate the
    tolerance for accuracy-critical use; it is exact-enough for the inference-latency win (~3x at ``B=1``)."""
    n = 0
    for m in model.modules():
        if isinstance(m, ChebyshevCRActivation):
            m.deploy = deploy
            n += 1
    return n


def make_activation(kind: str, n_channels: int, *, grid: int = 5) -> nn.Module:
    """Edge-activation Strategy: ``"cr"`` (the Catmull-Rom KAN spline that defines HSiKAN), ``"cr_cheby"`` (the
    CR-Chebyshev cell — band-limited control points), ``"bspline"`` (the legacy vision spline, for CR-vs-B-spline
    A/B), or ``"relu"`` / ``"tanh"`` for ablation (a non-KAN signed-GCN). # Errors ``ValueError`` on an unknown
    kind."""
    if kind == "cr":
        return CatmullRomActivation(n_channels, grid=grid)
    if kind == "cr_cheby":
        return ChebyshevCRActivation(n_channels, grid=max(grid, 8))   # band-limit needs a few knots
    if kind == "bspline":
        return BSplineActivation(n_channels, grid=grid)
    if kind == "relu":
        return nn.ReLU()
    if kind == "tanh":
        return nn.Tanh()
    raise ValueError(f"activation must be 'cr', 'cr_cheby', 'bspline', 'relu', or 'tanh'; got {kind!r}")
