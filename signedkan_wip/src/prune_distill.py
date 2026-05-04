"""SignedKAN pruning + symbolic distillation.

Phase A: prune low-activity splines (zero out coefficient vectors below
an L2-norm threshold).
Phase B: for surviving splines, sample the activation curve and fit a
symbolic form by least squares against a small library
(linear, quadratic, cubic, sine, exp, tanh).
Phase C: re-evaluate the model with symbolic-substituted splines.

Pruning criterion is the L2 norm of the per-(branch, channel)
coefficient vector. Distillation criterion is residual MSE on a
$200$-point grid sampled in $[-1, 1]$.

Greedy per-spline (no joint optimisation) — sufficient at this scale,
since the per-channel contributions to the downstream loss are weakly
coupled. A B&B refinement is straightforward to add later if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit

from .splines import (BatchedBSplineActivation,
                      DiagonalBatchedBSplineActivation,
                      BatchedCatmullRomActivation,
                      DiagonalBatchedCatmullRomActivation,
                      BatchedKochanekBartelsActivation,
                      DiagonalBatchedKochanekBartelsActivation,
                      cox_de_boor, _catmull_rom_eval, _kb_eval)


# ─── Symbolic library ────────────────────────────────────────────────


def _zero(x):                    return np.zeros_like(x)
def _linear(x, a, b):             return a * x + b
def _quadratic(x, a, b, c):       return a * x**2 + b * x + c
def _cubic(x, a, b, c, d):        return a * x**3 + b * x**2 + c * x + d
def _sine(x, a, w, p, c):         return a * np.sin(w * x + p) + c
def _exp(x, a, b, c):             return a * np.exp(np.clip(b * x, -8, 8)) + c
def _tanh(x, a, b, c):            return a * np.tanh(b * x) + c


SYMBOLIC_LIBRARY: dict[str, tuple[Callable, int, list]] = {
    # name        function     n_params  initial guess for curve_fit
    "linear":    (_linear,    2, [1.0, 0.0]),
    "quadratic": (_quadratic, 3, [0.0, 1.0, 0.0]),
    "cubic":     (_cubic,     4, [0.0, 0.0, 1.0, 0.0]),
    "sine":      (_sine,      4, [1.0, 1.0, 0.0, 0.0]),
    "exp":       (_exp,       3, [1.0, 1.0, 0.0]),
    "tanh":      (_tanh,      3, [1.0, 1.0, 0.0]),
}


# ─── Spline-output sampling ──────────────────────────────────────────


def _sample_b_spline(coef_vec: torch.Tensor, knots: torch.Tensor,
                     k: int, x: torch.Tensor) -> torch.Tensor:
    """Forward one (branch, channel) of a B-spline at x samples.

    coef_vec : (n_basis,)   coefficient vector
    knots    : (n_basis+k+1,) padded knot vector
    """
    B = cox_de_boor(x, knots, k)            # (n_samples, n_basis)
    return (B * coef_vec.unsqueeze(0)).sum(dim=-1)


def _sample_catmull_rom(coef_vec: torch.Tensor, grid: int,
                        x: torch.Tensor) -> torch.Tensor:
    """Forward one (branch, channel) of a Catmull-Rom spline at x samples."""
    # _catmull_rom_eval expects coef shape (..., C, G) and x shape (..., C).
    # Make a single-channel batch.
    coef = coef_vec.view(1, 1, -1)            # (1, 1, G)
    coef_b = coef.expand(x.shape[0], 1, grid) # (n_samples, 1, G)
    x_b    = x.unsqueeze(-1)                  # (n_samples, 1)
    out = _catmull_rom_eval(coef_b, x_b, grid)  # (n_samples, 1)
    return out.squeeze(-1)


def _sample_kochanek_bartels(coef_vec: torch.Tensor, tcb_vec: torch.Tensor,
                             grid: int, x: torch.Tensor) -> torch.Tensor:
    """Forward one (branch, channel) of a Kochanek-Bartels spline at x.

    coef_vec : (G,)        per-control-point coefficient
    tcb_vec  : (G, 3)      per-control-point (t, c, b) triple
    """
    coef = coef_vec.view(1, 1, -1)                # (1, 1, G)
    coef_b = coef.expand(x.shape[0], 1, grid)     # (n_samples, 1, G)
    tcb = tcb_vec.view(1, 1, grid, 3)             # (1, 1, G, 3)
    tcb_b = tcb.expand(x.shape[0], 1, grid, 3)    # (n_samples, 1, G, 3)
    x_b = x.unsqueeze(-1)                         # (n_samples, 1)
    out = _kb_eval(coef_b, tcb_b, x_b, grid)      # (n_samples, 1)
    return out.squeeze(-1)


def sample_spline_activation(activation: nn.Module,
                             branch: int, channel: int,
                             n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, y)`` arrays for one (branch, channel) of a batched
    spline activation, sampled on $[-1, 1]$."""
    x = torch.linspace(-1.0, 1.0, n_samples, dtype=torch.float32)
    coef = activation.coef.detach().cpu()
    if isinstance(activation, (BatchedBSplineActivation,
                                DiagonalBatchedBSplineActivation)):
        knots = activation.knots.detach().cpu()
        y = _sample_b_spline(coef[branch, channel], knots,
                              activation.k, x)
    elif isinstance(activation, (BatchedCatmullRomActivation,
                                  DiagonalBatchedCatmullRomActivation)):
        y = _sample_catmull_rom(coef[branch, channel], activation.grid, x)
    elif isinstance(activation, (BatchedKochanekBartelsActivation,
                                  DiagonalBatchedKochanekBartelsActivation)):
        tcb = activation.tcb.detach().cpu()
        y = _sample_kochanek_bartels(coef[branch, channel],
                                      tcb[branch, channel],
                                      activation.grid, x)
    else:
        raise TypeError(f"unsupported activation: {type(activation).__name__}")
    return x.numpy(), y.numpy()


# ─── Pruning ─────────────────────────────────────────────────────────


def measure_activity(activation: nn.Module) -> torch.Tensor:
    """Per-(branch, channel) L2-norm of coefficient vector."""
    coef = activation.coef.detach()           # (S, C, G or n_basis)
    return coef.pow(2).sum(dim=-1).sqrt()     # (S, C)


def prune_inactive(activation: nn.Module, threshold: float) -> int:
    """Zero out coefficient vectors with L2-norm < threshold.
    Returns the number of pruned (branch, channel) splines."""
    with torch.no_grad():
        norms = measure_activity(activation)            # (S, C)
        mask = norms < threshold                        # (S, C)
        # Broadcast mask to coef shape (S, C, ...).
        mask_b = mask.unsqueeze(-1).expand_as(activation.coef)
        activation.coef.data[mask_b] = 0.0
    return int(mask.sum().item())


# ─── Symbolic fitting ────────────────────────────────────────────────


@dataclass
class SymbolicFit:
    form: str                 # "zero" / "linear" / ... / "tanh"
    params: list[float]       # fitted parameters (empty for "zero")
    residual: float           # MSE on the sampled grid


def fit_symbolic(x: np.ndarray, y: np.ndarray,
                 prune_threshold_y: float = 1e-3) -> SymbolicFit:
    """Try each form in ``SYMBOLIC_LIBRARY``, return the best fit
    (lowest residual MSE).

    If $\\max |y| < \\text{prune\\_threshold\\_y}$ returns ``zero``.
    """
    if np.max(np.abs(y)) < prune_threshold_y:
        return SymbolicFit("zero", [], float(np.mean(y * y)))
    best = SymbolicFit("zero", [], float("inf"))
    for name, (func, _, p0) in SYMBOLIC_LIBRARY.items():
        try:
            popt, _ = curve_fit(func, x, y, p0=p0, maxfev=2000)
            r = float(np.mean((func(x, *popt) - y) ** 2))
            if r < best.residual:
                best = SymbolicFit(name, [float(p) for p in popt], r)
        except Exception:
            continue
    return best


def distill_activation(activation: nn.Module) -> list[list[SymbolicFit]]:
    """Distill every (branch, channel) of a batched-spline activation
    into a symbolic form. Returns ``fits[s][c]`` = ``SymbolicFit``."""
    S = activation.coef.shape[0]
    C = activation.coef.shape[1]
    fits: list[list[SymbolicFit]] = []
    for s in range(S):
        row: list[SymbolicFit] = []
        for c in range(C):
            x, y = sample_spline_activation(activation, s, c)
            row.append(fit_symbolic(x, y))
        fits.append(row)
    return fits


def fit_summary(fits: list[list[SymbolicFit]]) -> dict[str, int]:
    """Histogram of symbolic forms across all (branch, channel)."""
    h: dict[str, int] = {}
    for row in fits:
        for f in row:
            h[f.form] = h.get(f.form, 0) + 1
    return h


# ─── Symbolic-substitution forward pass ──────────────────────────────


def evaluate_symbolic(x: np.ndarray, fit: SymbolicFit) -> np.ndarray:
    """Evaluate a fitted symbolic form on x samples (numpy)."""
    if fit.form == "zero":
        return np.zeros_like(x)
    func, _, _ = SYMBOLIC_LIBRARY[fit.form]
    return func(x, *fit.params)
