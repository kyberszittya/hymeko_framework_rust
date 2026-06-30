"""Train (fit) Chebyshev-CR coefficients to a target timbre.

The Chebyshev-CR cell is differentiable, so its single-cycle waveform can be fit
by gradient descent to a target — a *tune* timbre (saw / square / organ / vowel)
or a *drum* body (tom / metallic). This turns a procedural wavetable into a
**learned patch** matching a chosen tone. Loss = time-domain MSE + a spectral
(rFFT-magnitude) term so the harmonic content matches, not just the shape.

Fit a target array directly (``fit_chebyshev``) — so a cycle extracted from a
real sample drops in unchanged — or a named analytic target (``TARGETS``).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from signed_kan import ChebyshevCRActivation

from signedkan_wip.examples.cheby_music.oscillator import Wavetable, chebyshev_wavetable


def _norm(w: np.ndarray) -> np.ndarray:
    w = w - w.mean()
    peak = np.abs(w).max()
    return (w / peak if peak > 0 else w).astype(np.float32)


# Target single-cycle waveforms over the phase grid x ∈ [-1, 1].
TARGETS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "saw": lambda x: _norm(x),
    "square": lambda x: _norm(np.sign(x)),
    "triangle": lambda x: _norm(1.0 - 2.0 * np.abs(x)),
    "organ": lambda x: _norm(np.sin(np.pi * x) + 0.5 * np.sin(2 * np.pi * x)
                             + 0.3 * np.sin(3 * np.pi * x)),
    "vowel_a": lambda x: _norm(np.sin(np.pi * x) + 0.7 * np.sin(3 * np.pi * x)
                               + 0.5 * np.sin(4 * np.pi * x) + 0.2 * np.sin(7 * np.pi * x)),
    "tom_body": lambda x: _norm(np.sin(np.pi * x) + 0.4 * np.sin(2.7 * np.pi * x)
                                + 0.2 * np.sin(5.1 * np.pi * x)),
    "metallic": lambda x: _norm(sum(np.sin(np.pi * k * x) / k for k in (3, 5, 7, 9, 11))),
}


def target_array(name: str, size: int = 512) -> np.ndarray:
    """Sample a named target waveform over one period (``size`` points)."""
    if name not in TARGETS:
        raise ValueError(f"unknown target {name!r}; choose {sorted(TARGETS)}")
    x = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return TARGETS[name](x)


def fit_chebyshev(target: np.ndarray, *, degree: int = 14, grid: int = 24,
                  iters: int = 600, lr: float = 0.05, spectral_weight: float = 0.5,
                  seed: int = 0) -> tuple[ChebyshevCRActivation, list[float]]:
    """Fit a 1-channel Chebyshev-CR cell to ``target`` (one period, any length).

    Preconditions: ``target`` 1-D with >= 8 samples; ``2 <= degree <= grid``.
    Postconditions: returns the trained cell and the per-iteration loss history
    (monotone-ish, decreasing); the cell's curve over ``[-1,1]`` approximates the
    (band-limited) target.
    """
    tgt = torch.as_tensor(_norm(np.asarray(target)), dtype=torch.float32)
    if tgt.ndim != 1 or tgt.numel() < 8:
        raise ValueError("target must be a 1-D array with >= 8 samples")
    torch.manual_seed(seed)
    cell = ChebyshevCRActivation(1, grid=grid, k=degree)
    x = torch.linspace(-1.0, 1.0, tgt.numel()).unsqueeze(-1)
    tgt_mag = torch.fft.rfft(tgt).abs()
    opt = torch.optim.Adam(cell.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(iters):
        opt.zero_grad()
        out = cell(x).squeeze(-1)
        time_loss = torch.mean((out - tgt) ** 2)
        spec_loss = torch.mean((torch.fft.rfft(out).abs() - tgt_mag) ** 2) / tgt.numel()
        loss = time_loss + spectral_weight * spec_loss
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return cell, history


def trained_wavetable(name: str, *, size: int = 2048, **fit_kw) -> Wavetable:
    """Convenience: fit a named target and return a ready-to-play wavetable."""
    cell, _ = fit_chebyshev(target_array(name), **fit_kw)
    wt = chebyshev_wavetable(cell, 0, size=size)
    wt.name = f"trained[{name}]"
    return wt


def fit_quality(cell: ChebyshevCRActivation, target: np.ndarray) -> dict[str, float]:
    """Correlation + RMSE between the trained curve and the (normalised) target."""
    tgt = _norm(np.asarray(target))
    x = torch.linspace(-1.0, 1.0, tgt.shape[0]).unsqueeze(-1)
    with torch.no_grad():
        out = _norm(cell(x).squeeze(-1).numpy())
    corr = float(np.corrcoef(out, tgt)[0, 1])
    rmse = float(np.sqrt(np.mean((out - tgt) ** 2)))
    return {"corr": corr, "rmse": rmse}
