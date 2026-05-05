"""SignedKAN — Phase 2.1–2.5: Cox--de Boor B-spline activation.

Vectorised cubic (k=3) B-spline evaluation on a fixed uniform knot
grid. Differentiable through autograd. The activation function form is

    f(x) = Σ_i c_i · B_i^k(x)

where B_i^k are the B-spline basis functions of order k on the knot
sequence, and c_i are learnable per-(vertex, edge) coefficients
populated upstream by the SignedKAN layer.

Defaults from DECISIONS.md:
  - k = 3 (cubic)
  - G = 5 (knot grid resolution on [-1, 1])
  - knots are uniform: t_0 = -1, ..., t_{G-1} = +1 with order-k padding
"""
from __future__ import annotations

import torch
import torch.nn as nn


def cox_de_boor(x: torch.Tensor, knots: torch.Tensor, k: int) -> torch.Tensor:
    """Cox--de Boor recursion. Vectorised over an arbitrary leading
    batch shape.

    Parameters
    ----------
    x      : (B,) input values, in the active knot range
    knots  : (G + 2k + 1,) padded knot vector
    k      : spline order (cubic = 3)

    Returns
    -------
    B : (B, G + k - 1) basis function values
        Each row sums to 1 within the active region.
    """
    # Order-0: B_i^0(x) = 1 if t_i ≤ x < t_{i+1} else 0.
    # We use [t_i ≤ x ≤ t_{i+1}] inclusive on the right boundary
    # to keep gradients defined at the right edge.
    x = x.unsqueeze(-1)                       # (B, 1)
    n_basis = knots.shape[0] - 1
    left = knots[:-1]                          # (n_basis,)
    right = knots[1:]                          # (n_basis,)
    B = ((x >= left) & (x < right)).to(x.dtype)  # (B, n_basis)
    # Edge case: include the right-most knot point so x = +1 evaluates.
    # (We attach it to the last interval.)
    last_mask = (x.squeeze(-1) == knots[-1])
    if last_mask.any():
        B[last_mask, -1] = 1.0

    # Cox-de Boor recursion, vectorised over the basis-index loop.
    # The original implementation had a Python `for i in range(...)`
    # over basis indices, firing ~k * n_basis sequential CUDA launches.
    # Here all index slices are computed in one shot with tensor slices;
    # the only remaining loop is the order-recursion (k iterations,
    # cubic = 3) which is intrinsic to the algorithm.
    x_flat = x.squeeze(-1)                           # (B,)
    for order in range(1, k + 1):
        n_basis_o = knots.shape[0] - 1 - order
        knots_l   = knots[:n_basis_o]                # (n_basis_o,)
        knots_lp  = knots[1:1 + n_basis_o]           # (n_basis_o,)
        knots_ro  = knots[order:order + n_basis_o]   # (n_basis_o,)
        knots_rop = knots[order + 1:order + 1 + n_basis_o]  # (n_basis_o,)
        denom_left  = (knots_ro  - knots_l ).clamp(min=1e-12)
        denom_right = (knots_rop - knots_lp).clamp(min=1e-12)
        # Broadcast x_flat: (B,) → (B, 1) against the (n_basis_o,) knot
        # slices to produce (B, n_basis_o).
        x_b = x_flat.unsqueeze(-1)                   # (B, 1)
        term_left  = (x_b - knots_l ) / denom_left  * B[..., :n_basis_o]
        term_right = (knots_rop - x_b) / denom_right * B[..., 1:1 + n_basis_o]
        B = term_left + term_right
    return B


def make_uniform_knots(grid: int, k: int,
                       lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """Uniform knot vector on [lo, hi] padded with k extra knots on
    each end (clamped uniform extension)."""
    inner = torch.linspace(lo, hi, grid)
    step = (hi - lo) / (grid - 1)
    left_pad = torch.tensor([lo - (k - i) * step for i in range(k)])
    right_pad = torch.tensor([hi + (i + 1) * step for i in range(k)])
    return torch.cat([left_pad, inner, right_pad])


class BatchedBSplineActivation(nn.Module):
    """Batched B-spline activation: $S$ parallel spline pairs sharing
    the same Cox--de Boor basis but with separate coefficient tensors,
    contracted in a single ``einsum``.

    Mathematically:
        $$f^{(s)}_c(x) \;=\; \sum_n c^{(s)}_{c,n} \, B_n(x)$$

    where $B_n(x)$ is the $n$-th basis function (shared across $s$),
    $c^{(s)}_{c,n}$ is a per-branch per-channel coefficient, and the
    output stacks the $S$ branches along a new axis.

    Replaces $S$ separate ``BSplineActivation`` calls with a single
    basis evaluation followed by one ``einsum``. Used in
    ``SignedKANLayer`` to fuse the inner-positive / inner-negative
    spline pair (shared input) and the outer-positive /
    outer-negative spline pair (per-sign input but still one
    fused call via index selection).

    input: ``(B, n_channels)``, in the active range $[-1, 1]$
    output: ``(B, n_branches, n_channels)``
    """
    def __init__(self, n_branches: int, n_channels: int, grid: int = 5,
                 k: int = 3, init_scale: float = 0.1):
        super().__init__()
        self.n_branches = n_branches
        self.n_channels = n_channels
        self.grid = grid
        self.k = k
        knots = make_uniform_knots(grid, k)
        self.register_buffer("knots", knots, persistent=False)
        n_basis = grid + k - 1
        # Coefficients: one set per (branch, channel).
        self.coef = nn.Parameter(
            torch.randn(n_branches, n_channels, n_basis) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, C) → out: (B, S, C)``."""
        x = x.clamp(min=-1.0, max=1.0)
        B_, C = x.shape
        # Single basis evaluation, shared across the $S$ branches.
        B_basis = cox_de_boor(x.reshape(-1), self.knots, self.k)
        B_basis = B_basis.view(B_, C, -1)              # (B, C, n_basis)
        # Branch-conditioned contraction:
        #   out[b, s, c] = sum_n  B_basis[b, c, n] * coef[s, c, n]
        return torch.einsum('bcn,scn->bsc', B_basis, self.coef)


class DiagonalBatchedBSplineActivation(nn.Module):
    """Per-branch diagonal spline: input row $s$ is processed by
    branch $s$'s coefficients only. No off-diagonal compute.

    Mathematically: for input $x \in \mathbb{R}^{B \times S \times C}$
    and coefficients $c^{(s)}_{c,n}$,
        $$f(x)_{b, s, c} \;=\; \sum_n c^{(s)}_{c,n} \, B_n(x_{b, s, c}).$$

    Compared to ``BatchedBSplineActivation`` followed by a diagonal
    selection across the branch axis, this saves a factor of $S$ on
    the einsum (skips computing $S^2$ outputs and discarding $S^2-S$).

    Used for the outer spline pair in ``SignedKANLayer``: each per-sign
    aggregate is fed only into its own sign's outer spline.

    input: ``(B, S, C)``, in the active range $[-1, 1]$
    output: ``(B, S, C)``
    """
    def __init__(self, n_branches: int, n_channels: int, grid: int = 5,
                 k: int = 3, init_scale: float = 0.1):
        super().__init__()
        self.n_branches = n_branches
        self.n_channels = n_channels
        self.grid = grid
        self.k = k
        knots = make_uniform_knots(grid, k)
        self.register_buffer("knots", knots, persistent=False)
        n_basis = grid + k - 1
        self.coef = nn.Parameter(
            torch.randn(n_branches, n_channels, n_basis) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, S, C) → out: (B, S, C)``."""
        x = x.clamp(min=-1.0, max=1.0)
        B_, S, C = x.shape
        B_basis = cox_de_boor(x.reshape(-1), self.knots, self.k)
        B_basis = B_basis.view(B_, S, C, -1)            # (B, S, C, n_basis)
        # Per-branch diagonal contraction:
        #   out[b, s, c] = sum_n  B_basis[b, s, c, n] * coef[s, c, n]
        return torch.einsum('bscn,scn->bsc', B_basis, self.coef)


class BSplineActivation(nn.Module):
    """Learnable cubic B-spline activation with per-channel coefficients.

    Used as the inner+outer spline pair in a SignedKAN layer. Shape:
      input: (B, n_channels)  in the active range [-1, 1]
      output: (B, n_channels)
    """
    def __init__(self, n_channels: int, grid: int = 5, k: int = 3,
                 init_scale: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.grid = grid
        self.k = k
        knots = make_uniform_knots(grid, k)
        self.register_buffer("knots", knots, persistent=False)
        n_basis = grid + k - 1
        # Learnable coefficients: one set per channel.
        self.coef = nn.Parameter(
            torch.randn(n_channels, n_basis) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_channels)  →  out: (B, n_channels)."""
        # Clamp x to the active knot range so basis values are nonzero.
        x = x.clamp(min=-1.0, max=1.0)
        B, C = x.shape
        # Evaluate basis for all channels at once: flatten (B, C) → (B*C,)
        # then reshape back. Knots are shared across channels.
        B_basis = cox_de_boor(x.reshape(-1), self.knots, self.k)
        # B_basis: (B*C, n_basis). coef: (C, n_basis).
        # We need per-channel coefficients applied to each row.
        n_basis = B_basis.shape[-1]
        B_basis = B_basis.view(B, C, n_basis)
        # Output: (B, C) = sum over n_basis of B_basis[B,C,:] * coef[C,:]
        return (B_basis * self.coef).sum(dim=-1)


# ─── Catmull-Rom activations ─────────────────────────────────────────


import os as _os


def _maybe_compile(fn):
    """Wrap fn in ``torch.compile`` when the env flag is set.
    Default off; enable with ``HSIKAN_TORCH_COMPILE=1``.

    Mode is chosen via ``HSIKAN_COMPILE_MODE`` (default
    ``reduce-overhead`` which uses CUDA graphs to fuse per-call kernel-
    launch overhead — the dominant inference-latency cost on cuda).
    Set to ``default`` for the conservative inductor path if cudagraphs
    cause problems (graph captures require static shapes; recompilation
    on shape change is more expensive than ``default`` mode).

    Decoupled from default so eager fallbacks remain available when
    torch.compile encounters incompatible ops on a given PyTorch
    version."""
    if _os.environ.get("HSIKAN_TORCH_COMPILE", "0") != "1":
        return fn
    mode = _os.environ.get("HSIKAN_COMPILE_MODE", "reduce-overhead")
    try:
        return torch.compile(fn, dynamic=False, fullgraph=False, mode=mode)
    except Exception:                                              # pragma: no cover
        return fn


# Catmull-Rom basis matrix construction is inlined inside
# _catmull_rom_eval — torch.compile constant-folds it into the graph,
# while eager pays a ~1ms-per-call torch.tensor build that's tolerable
# given the spline call only happens 4× per forward. Earlier we cached
# the matrix in a module-level dict, but that aliasing breaks
# torch.compile's `mode='reduce-overhead'` (cudagraphs) — the cached
# tensor's storage gets reported as overwritten between graph runs.
@_maybe_compile
def _catmull_rom_eval(coef: torch.Tensor, x: torch.Tensor,
                      grid: int) -> torch.Tensor:
    """Uniform Catmull-Rom cubic interpolating spline on $[-1, 1]$.

    coef: (..., C, G) per-channel control points.
    x   : (..., C)    inputs in $[-1, 1]$.
    Returns the spline value of shape ``(..., C)``.

    Memory-friendly form (post-2026-05-04 rewrite):
      - Basis weights computed via the closed-form CR polynomial
        ($w_{-1} = (-t^3 + 2t^2 - t)/2$ etc.).  No intermediate
        ``t_powers`` stack and no $4{\\times}4$ matmul — each
        weight is a single fused expression in ``t``.
      - Four separate gathers (one per control point) instead of
        a stacked-index single gather.  Trades one kernel launch
        for a $\\sim 4\\times$ smaller index tensor (no
        ``(\\ldots, C, 4)`` int64 stack), which is the dominant
        autograd-retained intermediate at large $T$.

    The compiled (lever 2) form sits one level up: the
    ``BatchedCatmullRomActivation`` modules wrap the call in
    ``torch.compile`` once at module construction; the lower
    intermediate count here lets ``torch.compile`` fuse more
    aggressively in the cudagraph capture.
    """
    G = grid
    x = x.clamp(min=-1.0, max=1.0)
    u = (x + 1.0) * 0.5 * (G - 1)
    i = u.floor().long().clamp(max=G - 2)
    t = u - i.to(x.dtype)
    t2 = t * t
    t3 = t2 * t

    # Closed-form Catmull-Rom blend weights (each (..., C)):
    #   w_m1 = 0.5 * (-t^3 + 2t^2 - t)
    #   w_0  = 0.5 * (3t^3 - 5t^2 + 2)
    #   w_p1 = 0.5 * (-3t^3 + 4t^2 + t)
    #   w_p2 = 0.5 * (t^3 - t^2)
    w_m1 = 0.5 * (-t3 + 2.0 * t2 - t)
    w_0  = 0.5 * ( 3.0 * t3 - 5.0 * t2 + 2.0)
    w_p1 = 0.5 * (-3.0 * t3 + 4.0 * t2 + t)
    w_p2 = 0.5 * ( t3 - t2)

    # Gather indices — int64 but kept as separate (..., C) tensors,
    # so we never materialise a (..., C, 4) int64 stack
    # (which is 8 bytes/element × 4 = the dominant memory chunk in
    # the prior path).
    idx_m1 = (i - 1).clamp(min=0, max=G - 1).unsqueeze(-1)        # (..., C, 1)
    idx_0  = i.unsqueeze(-1)
    idx_p1 = (i + 1).clamp(min=0, max=G - 1).unsqueeze(-1)
    idx_p2 = (i + 2).clamp(min=0, max=G - 1).unsqueeze(-1)
    coef_b = coef.expand(*x.shape, G)

    P_m1 = coef_b.gather(-1, idx_m1).squeeze(-1)                  # (..., C)
    P_0  = coef_b.gather(-1, idx_0).squeeze(-1)
    P_p1 = coef_b.gather(-1, idx_p1).squeeze(-1)
    P_p2 = coef_b.gather(-1, idx_p2).squeeze(-1)

    return w_m1 * P_m1 + w_0 * P_0 + w_p1 * P_p1 + w_p2 * P_p2


class CatmullRomActivation(nn.Module):
    """Learnable cubic Catmull-Rom interpolating spline activation.

    $G$ control points per channel, $C^1$ continuous, interpolatory
    (the curve passes through the control points). Compared to
    ``BSplineActivation`` at the same $G$ this uses $G$ parameters
    per channel instead of $G + k - 1$, at the cost of weaker local
    control (changing one $P_i$ tilts neighbouring tangents).

    input : ``(B, n_channels)`` in $[-1, 1]$
    output: ``(B, n_channels)``
    """
    def __init__(self, n_channels: int, grid: int = 5,
                 init_scale: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.grid = grid
        self.coef = nn.Parameter(
            torch.randn(n_channels, grid) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, C) → out: (B, C)``."""
        # coef expand: (B, C, G) for gather alignment.
        coef = self.coef.unsqueeze(0).expand(x.shape[0], -1, -1)
        return _catmull_rom_eval(coef, x, self.grid)


class BatchedCatmullRomActivation(nn.Module):
    """$S$ parallel Catmull-Rom splines sharing the same per-channel
    input but with separate per-branch control points.

    input : ``(B, n_channels)`` in $[-1, 1]$
    output: ``(B, n_branches, n_channels)``
    """
    def __init__(self, n_branches: int, n_channels: int, grid: int = 5,
                 init_scale: float = 0.1):
        super().__init__()
        self.n_branches = n_branches
        self.n_channels = n_channels
        self.grid = grid
        self.coef = nn.Parameter(
            torch.randn(n_branches, n_channels, grid) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, C) → out: (B, S, C)``."""
        B_, C = x.shape
        S = self.n_branches
        # Broadcast x and coef to a common (B, S, C) leading shape:
        # x → (B, 1, C) → (B, S, C); coef (S, C, G) → (1, S, C, G)
        # → (B, S, C, G).
        x_b = x.unsqueeze(1).expand(B_, S, C)
        coef_b = self.coef.unsqueeze(0).expand(B_, S, C, self.grid)
        return _catmull_rom_eval(coef_b, x_b, self.grid)


class DiagonalBatchedCatmullRomActivation(nn.Module):
    """Per-branch diagonal Catmull-Rom: input row $s$ uses branch
    $s$'s control points only.

    input : ``(B, S, n_channels)`` in $[-1, 1]$
    output: ``(B, S, n_channels)``
    """
    def __init__(self, n_branches: int, n_channels: int, grid: int = 5,
                 init_scale: float = 0.1):
        super().__init__()
        self.n_branches = n_branches
        self.n_channels = n_channels
        self.grid = grid
        self.coef = nn.Parameter(
            torch.randn(n_branches, n_channels, grid) * init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, S, C) → out: (B, S, C)``."""
        B_, S, C = x.shape
        coef_b = self.coef.unsqueeze(0).expand(B_, S, C, self.grid)
        return _catmull_rom_eval(coef_b, x, self.grid)


# ─── Kochanek-Bartels activations ────────────────────────────────────


def _kb_eval(coef: torch.Tensor,
             tcb_raw: torch.Tensor,
             x: torch.Tensor,
             grid: int) -> torch.Tensor:
    """Kochanek-Bartels (TCB) cubic spline on $[-1, 1]$.

    coef    : (..., C, G)     per-channel control points.
    tcb_raw : (..., C, G, 3)  unconstrained learnable triples; mapped
                              to (t, c, b) ∈ (-1, 1) via tanh.
    x       : (..., C)        inputs in $[-1, 1]$.
    Returns the spline value of shape ``(..., C)``.

    With $t = c = b = 0$ this reduces exactly to Catmull-Rom. For a
    pair of control points $P_i, P_{i+1}$ the in/out tangents are
    given by the standard KB blend:

      $$
      \\mathbf{D}_i^{out} = \\tfrac{(1-t_i)(1+c_i)(1+b_i)}{2}(P_i\\!-\\!P_{i-1})
                          + \\tfrac{(1-t_i)(1-c_i)(1-b_i)}{2}(P_{i+1}\\!-\\!P_i)
      $$
      $$
      \\mathbf{D}_{i+1}^{in} = \\tfrac{(1-t_{i+1})(1-c_{i+1})(1+b_{i+1})}{2}
                                  (P_{i+1}\\!-\\!P_i)
                              + \\tfrac{(1-t_{i+1})(1+c_{i+1})(1-b_{i+1})}{2}
                                  (P_{i+2}\\!-\\!P_{i+1})
      $$

    Then standard cubic Hermite interpolation between $P_i$ and
    $P_{i+1}$ with these two tangents.
    """
    G = grid
    x = x.clamp(min=-1.0, max=1.0)
    u = (x + 1.0) * 0.5 * (G - 1)
    i = u.floor().long().clamp(max=G - 2)
    s = u - i.to(x.dtype)
    s2 = s * s
    s3 = s2 * s
    # Cubic Hermite basis (standard form, segment parameter ∈ [0, 1]):
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2

    # Gather P_{i-1}, P_i, P_{i+1}, P_{i+2}.
    idx_m1 = (i - 1).clamp(min=0, max=G - 1)
    idx_0  = i
    idx_p1 = (i + 1).clamp(min=0, max=G - 1)
    idx_p2 = (i + 2).clamp(min=0, max=G - 1)
    coef_b = coef.expand(*x.shape, G)
    Pm1 = coef_b.gather(-1, idx_m1.unsqueeze(-1)).squeeze(-1)
    P0  = coef_b.gather(-1, idx_0 .unsqueeze(-1)).squeeze(-1)
    P1  = coef_b.gather(-1, idx_p1.unsqueeze(-1)).squeeze(-1)
    P2  = coef_b.gather(-1, idx_p2.unsqueeze(-1)).squeeze(-1)

    # Gather (t, c, b) at the two endpoints of the active segment,
    # mapping unconstrained learnable params through tanh.
    tcb = torch.tanh(tcb_raw)                          # (..., C, G, 3)
    # tcb has shape (..., C, G, 3); we need (t, c, b) at i and at i+1.
    # We gather along the G axis using idx_0 and idx_p1.
    # Expand tcb to broadcast against the leading shape of x (..., C).
    tcb_b = tcb.expand(*x.shape, G, 3)
    idx_g_i  = idx_0 .unsqueeze(-1).unsqueeze(-1).expand(*x.shape, 1, 3)
    idx_g_ip = idx_p1.unsqueeze(-1).unsqueeze(-1).expand(*x.shape, 1, 3)
    tcb_i  = tcb_b.gather(-2, idx_g_i ).squeeze(-2)    # (..., C, 3)
    tcb_ip = tcb_b.gather(-2, idx_g_ip).squeeze(-2)    # (..., C, 3)
    t_i,  c_i,  b_i  = tcb_i .unbind(-1)
    t_ip, c_ip, b_ip = tcb_ip.unbind(-1)

    # KB out-tangent at P_i.
    w_left  = (1.0 - t_i)  * (1.0 + c_i)  * (1.0 + b_i)  * 0.5
    w_right = (1.0 - t_i)  * (1.0 - c_i)  * (1.0 - b_i)  * 0.5
    D_i_out = w_left * (P0 - Pm1) + w_right * (P1 - P0)

    # KB in-tangent at P_{i+1}.
    w_left  = (1.0 - t_ip) * (1.0 - c_ip) * (1.0 + b_ip) * 0.5
    w_right = (1.0 - t_ip) * (1.0 + c_ip) * (1.0 - b_ip) * 0.5
    D_ip_in = w_left * (P1 - P0) + w_right * (P2 - P1)

    return h00 * P0 + h10 * D_i_out + h01 * P1 + h11 * D_ip_in


class KochanekBartelsActivation(nn.Module):
    """Learnable cubic Kochanek-Bartels (TCB) spline activation.

    $G$ control points $\\times$ 3 hyperparameters per channel
    (tension $t$, continuity $c$, bias $b$, all in $(-1, 1)$ via
    tanh). At $t=c=b=0$ reduces to Catmull-Rom exactly.

    Compared to ``CatmullRomActivation`` this adds $3G$ parameters
    per channel to give per-control-point tangent shaping; the
    optimiser can independently bend, sharpen, or skew the curve at
    each knot without moving the curve's value.

    input : ``(B, n_channels)`` in $[-1, 1]$
    output: ``(B, n_channels)``
    """
    def __init__(self, n_channels: int, grid: int = 5,
                 init_scale: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.grid = grid
        self.coef = nn.Parameter(
            torch.randn(n_channels, grid) * init_scale
        )
        # Default-init at zero so the spline starts as Catmull-Rom.
        self.tcb = nn.Parameter(torch.zeros(n_channels, grid, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, C) → out: (B, C)``."""
        coef = self.coef.unsqueeze(0).expand(x.shape[0], -1, -1)
        tcb  = self.tcb .unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        return _kb_eval(coef, tcb, x, self.grid)


class BatchedKochanekBartelsActivation(nn.Module):
    """$S$ parallel KB splines sharing per-channel input.

    input : ``(B, n_channels)``
    output: ``(B, n_branches, n_channels)``
    """
    def __init__(self, n_branches: int, n_channels: int, grid: int = 5,
                 init_scale: float = 0.1):
        super().__init__()
        self.n_branches = n_branches
        self.n_channels = n_channels
        self.grid = grid
        self.coef = nn.Parameter(
            torch.randn(n_branches, n_channels, grid) * init_scale
        )
        self.tcb = nn.Parameter(torch.zeros(n_branches, n_channels, grid, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, C = x.shape
        S = self.n_branches
        x_b   = x.unsqueeze(1).expand(B_, S, C)
        coef_b = self.coef.unsqueeze(0).expand(B_, S, C, self.grid)
        tcb_b  = self.tcb .unsqueeze(0).expand(B_, S, C, self.grid, 3)
        return _kb_eval(coef_b, tcb_b, x_b, self.grid)


class DiagonalBatchedKochanekBartelsActivation(nn.Module):
    """Per-branch diagonal KB: input row $s$ uses branch $s$'s
    (control points, tangent triples) only.

    input : ``(B, S, n_channels)``
    output: ``(B, S, n_channels)``
    """
    def __init__(self, n_branches: int, n_channels: int, grid: int = 5,
                 init_scale: float = 0.1):
        super().__init__()
        self.n_branches = n_branches
        self.n_channels = n_channels
        self.grid = grid
        self.coef = nn.Parameter(
            torch.randn(n_branches, n_channels, grid) * init_scale
        )
        self.tcb = nn.Parameter(torch.zeros(n_branches, n_channels, grid, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, S, C = x.shape
        coef_b = self.coef.unsqueeze(0).expand(B_, S, C, self.grid)
        tcb_b  = self.tcb .unsqueeze(0).expand(B_, S, C, self.grid, 3)
        return _kb_eval(coef_b, tcb_b, x, self.grid)


# ─── Unit tests (Phase 2.4–2.5) ──────────────────────────────────────


def _test_partition_of_unity():
    """B-spline basis sums to 1 within the active region."""
    knots = make_uniform_knots(5, 3)
    x = torch.linspace(-0.95, 0.95, 200)
    B = cox_de_boor(x, knots, 3)
    sums = B.sum(dim=-1)
    err = (sums - 1.0).abs().max().item()
    assert err < 1e-5, f"partition-of-unity violated: max err = {err}"
    return err


def _test_zero_coef_zero_output():
    """Zero coefficients ⇒ zero output identically."""
    act = BSplineActivation(n_channels=8, grid=5, k=3)
    with torch.no_grad():
        act.coef.zero_()
    x = torch.randn(32, 8)
    y = act(x)
    err = y.abs().max().item()
    assert err < 1e-7, f"zero-coef zero-output failed: max |y| = {err}"
    return err


def _test_gradient_flow():
    """Gradient flows through the spline coefficients and the input."""
    act = BSplineActivation(n_channels=4, grid=5, k=3)
    x = torch.randn(16, 4, requires_grad=True)
    y = act(x).sum()
    y.backward()
    assert act.coef.grad is not None
    assert (act.coef.grad.abs() > 0).any(), "coefficient gradient is identically zero"
    assert x.grad is not None
    return float(act.coef.grad.norm())


def main() -> None:
    print("Phase 2 spline tests:")
    err1 = _test_partition_of_unity()
    print(f"  partition_of_unity: max err = {err1:.2e}  ✓")
    err2 = _test_zero_coef_zero_output()
    print(f"  zero_coef_zero_out: max err = {err2:.2e}  ✓")
    grad = _test_gradient_flow()
    print(f"  gradient_flow:      ‖∇c‖ = {grad:.4f}  ✓")
    print("\nAll Phase 2 spline tests passed.")


if __name__ == "__main__":
    main()
