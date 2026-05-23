"""K_g-class continuous-weight aggregator for weighted hyperedges.

Generalises the HSiKAN $\\sigma$-masked aggregator from binary
$\\sigma \\in \\{-1, +1\\}$ per-arc signs to per-arc continuous
weights $w(v, e) \\in \\mathbb{R}$ with $K_g$ soft sign classes.

For a hyperedge $e$ with arity $k$ and per-arc weights
$\\{w_i\\}_{i=1}^k$ and per-arc features $\\{h_i\\}_{i=1}^k$:

  g_{k,i} = phi_gate^{(k)}(w_i)          (soft per-class gate)
  a_k     = sum_i g_{k,i} * phi_in^{(k)}(h_i)   (per-class pool)
  z       = sum_k phi_out^{(k)}(a_k)             (output stack)

Each $\\phi_\\text{gate}^{(k)}$, $\\phi_\\text{in}^{(k)}$, and
$\\phi_\\text{out}^{(k)}$ is a learnable univariate function (we
parameterise as a Catmull-Rom spline through fixed-x learnable-y
control points, matching the HSiKAN basis from
``signedkan_wip/src/vision/hymeyolo_backbones.py``).

Binary special case: $K_g = 2$ with $\\phi_\\text{gate}^{(0)} =
\\indicator{w < 0}$ and $\\phi_\\text{gate}^{(1)} =
\\indicator{w > 0}$ recovers the existing HSiKAN $\\sigma$-mask.
The soft / smooth Catmull-Rom version approaches this in the
limit of high knot density.

Plan: ``docs/plans/2026-05-17-general-weighted-hyperedges/plan.tex``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _catmull_rom_eval(
    y_knots: torch.Tensor,
    x: torch.Tensor,
    x_min: float,
    x_max: float,
) -> torch.Tensor:
    """Cubic Catmull-Rom interpolation of $\\varphi(x)$ through
    learnable control values ``y_knots`` at uniformly-spaced fixed-x
    knots in ``[x_min, x_max]``.

    Parameters
    ----------
    y_knots : (G,) tensor of learnable control values.
    x : tensor of arbitrary leading shape; query points in
        $[x_\\min, x_\\max]$; values outside the range are linearly
        clamped (extrapolation is the boundary slope direction).
    x_min, x_max : float, knot range.

    Returns
    -------
    Tensor of same shape as ``x``.
    """
    G = int(y_knots.shape[0])
    if G < 4:
        raise ValueError(f"need >= 4 knots for cubic Catmull-Rom; got {G}")
    # Map x to grid coordinates in [0, G-1].
    t = (x.clamp(min=x_min, max=x_max) - x_min) / (x_max - x_min) * (G - 1)
    # Interval index i in [1, G - 3] so we have y[i-1..i+2] available.
    i = t.long().clamp(min=1, max=G - 3)
    local_t = t - i.to(t.dtype)
    # Gather the four control values per query.
    y_m1 = y_knots[i - 1]
    y_0  = y_knots[i]
    y_1  = y_knots[i + 1]
    y_2  = y_knots[i + 2]
    # Catmull-Rom matrix:
    #  [ 0   2    0   0 ]
    #  [-1   0    1   0 ] * 0.5 then dotted with (1, t, t^2, t^3)
    #  [ 2  -5    4  -1 ]
    #  [-1   3   -3   1 ]
    t2 = local_t * local_t
    t3 = t2 * local_t
    a = 0.5 * (-y_m1 + 3.0 * y_0 - 3.0 * y_1 + y_2)
    b = 0.5 * (2.0 * y_m1 - 5.0 * y_0 + 4.0 * y_1 - y_2)
    c = 0.5 * (-y_m1 + y_1)
    d = y_0
    return a * t3 + b * t2 + c * local_t + d


class WeightedHSiKANAggregator(nn.Module):
    """$K_g$-class continuous-weight aggregator.

    Parameters
    ----------
    d_in : int
        Per-arc feature dimension.
    d_hidden : int
        Output and intermediate-pool dimension.
    K_g : int, default 2
        Number of soft sign classes. The binary special case is
        $K_g = 2$ with the natural negative / positive gate split.
    n_gate_knots : int, default 7
        Number of control points for each Catmull-Rom gate.
    w_range : tuple, default (-1.0, +1.0)
        Range of the weight domain. Bitcoin ratings divided by 10
        sit in $[-1, +1]$; Reddit sentiment scores too.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 32,
        K_g: int = 2,
        n_gate_knots: int = 7,
        w_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        if K_g < 2:
            raise ValueError(f"K_g must be >= 2; got {K_g}")
        if n_gate_knots < 4:
            raise ValueError(f"n_gate_knots must be >= 4; got {n_gate_knots}")
        self.d_in = int(d_in)
        self.d_hidden = int(d_hidden)
        self.K_g = int(K_g)
        self.n_gate_knots = int(n_gate_knots)
        self.w_range = (float(w_range[0]), float(w_range[1]))

        # Per-class Catmull-Rom gate, parameterised by y_knots ∈ R^G.
        # Init: K_g = 2 → gate 0 ≈ "negative-leaning" (slope down across w),
        # gate 1 ≈ "positive-leaning" (slope up). For K_g > 2 we init
        # gates to be smooth probabilistic indicators of equal-width
        # intervals of [w_min, w_max].
        gates_init = torch.zeros(self.K_g, self.n_gate_knots)
        for k in range(self.K_g):
            # Centre of the k-th class in [w_min, w_max]:
            centre = w_range[0] + (k + 0.5) / self.K_g * (w_range[1] - w_range[0])
            for g in range(self.n_gate_knots):
                w_g = w_range[0] + g / (self.n_gate_knots - 1) * (w_range[1] - w_range[0])
                # Gaussian-shaped initial gate for class k centred on `centre`.
                sigma = (w_range[1] - w_range[0]) / (2.0 * self.K_g)
                gates_init[k, g] = float(torch.exp(
                    torch.tensor(-((w_g - centre) ** 2) / (2.0 * sigma * sigma))
                ).item())
        self.gate_knots = nn.Parameter(gates_init)

        # Per-class input and output projections. Each is a Linear;
        # we use simple linear projections rather than full Catmull-Rom
        # activations to keep the v1 parameter count tight.
        self.phi_in = nn.ModuleList([
            nn.Linear(d_in, d_hidden) for _ in range(self.K_g)
        ])
        self.phi_out = nn.ModuleList([
            nn.Linear(d_hidden, d_hidden) for _ in range(self.K_g)
        ])
        for layer in list(self.phi_in) + list(self.phi_out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def gate_values(self, w: torch.Tensor) -> torch.Tensor:
        """Compute the per-class gate values for a batch of per-arc
        weights.

        Parameters
        ----------
        w : tensor of shape (..., k) — per-arc weights for k arcs.

        Returns
        -------
        g : tensor of shape (K_g, ..., k) with non-negative gate
            values. We do NOT normalise across classes (the per-class
            pools are summed with the class-output projections, not
            averaged; if normalisation is desired, divide by the
            class sum externally).
        """
        gates = []
        for k in range(self.K_g):
            gates.append(_catmull_rom_eval(
                self.gate_knots[k], w,
                x_min=self.w_range[0], x_max=self.w_range[1],
            ))
        return torch.stack(gates, dim=0)

    def forward(
        self, h_corners: torch.Tensor, w_corners: torch.Tensor,
    ) -> torch.Tensor:
        """K_g-class continuous-weight aggregator forward.

        Parameters
        ----------
        h_corners : tensor (B, k, d_in)
            Per-arc features in the hyperedge.
        w_corners : tensor (B, k)
            Per-arc weights $w(v_i, e)$.

        Returns
        -------
        z : tensor (B, d_hidden)
            Per-hyperedge embedding.
        """
        if h_corners.dim() != 3:
            raise ValueError(
                f"h_corners must be (B, k, d_in); got {tuple(h_corners.shape)}"
            )
        if w_corners.dim() != 2 or w_corners.shape != h_corners.shape[:2]:
            raise ValueError(
                f"w_corners must be (B, k) matching h_corners; "
                f"got {tuple(w_corners.shape)} for h={tuple(h_corners.shape)}"
            )
        B, k, _ = h_corners.shape
        # Per-class gates: (K_g, B, k)
        gates = self.gate_values(w_corners)
        out_total = h_corners.new_zeros(B, self.d_hidden)
        for kc in range(self.K_g):
            g_k = gates[kc].unsqueeze(-1)            # (B, k, 1)
            u = torch.tanh(self.phi_in[kc](h_corners))  # (B, k, d_hidden)
            a_k = (u * g_k).sum(dim=1)               # (B, d_hidden)
            # Normalise by gate sum to keep magnitudes comparable.
            denom = g_k.sum(dim=1).clamp(min=1e-6)   # (B, 1)
            a_k = a_k / denom
            out_total = out_total + torch.tanh(self.phi_out[kc](a_k))
        return out_total

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
