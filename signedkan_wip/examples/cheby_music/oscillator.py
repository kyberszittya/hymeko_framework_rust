"""Wavetable oscillators: the *waveform* is a Chebyshev-CR curve or a wavelet.

This is the heart of the synth. A ``ChebyshevCRActivation`` channel's curve over
``[-1, 1]`` is read as **one cycle of a periodic waveform** (a wavetable); playing
it back at a note's frequency makes that channel an oscillator *timbre*. Wavelet
tables (Ricker / Morlet) give complementary timbres, and two tables can be
morphed. Playback is shared (linear-interpolated table lookup), so an oscillator
is just "a table + a frequency".

The Chebyshev band-limit (degree ``k``) bounds the waveform's harmonic content,
which keeps aliasing modest at musical pitches without an explicit anti-alias
pass — a natural fit for the CR-Chebyshev cell.
"""
from __future__ import annotations

import numpy as np
import torch

from signed_kan import ChebyshevCRActivation


class Wavetable:
    """One-cycle waveform table with linear-interpolated playback.

    Preconditions: ``table`` has length >= 2.
    Postconditions: the stored table is normalised to ``[-1, 1]``; ``render``
    returns ``n`` samples in roughly ``[-1, 1]``.
    """

    def __init__(self, table: np.ndarray, name: str = "wavetable") -> None:
        t = np.asarray(table, dtype=np.float32).ravel()
        if t.size < 2:
            raise ValueError("wavetable needs >= 2 samples")
        peak = float(np.abs(t).max())
        self.table = (t / peak if peak > 0 else t).astype(np.float32)
        self.size = t.size
        self.name = name

    def render(self, freq: "float | np.ndarray", n: int, sr: int) -> np.ndarray:
        """``n`` samples at ``freq`` Hz (scalar, or a per-sample array for
        vibrato). Phase advances by ``freq/sr`` per sample (cumulative)."""
        inc = np.broadcast_to(np.asarray(freq, dtype=np.float32) / sr, (n,))
        phase = np.cumsum(inc) % 1.0
        pos = phase * self.size
        i0 = np.floor(pos).astype(np.int64) % self.size
        i1 = (i0 + 1) % self.size
        frac = (pos - np.floor(pos)).astype(np.float32)
        return (1.0 - frac) * self.table[i0] + frac * self.table[i1]


def chebyshev_wavetable(cell: ChebyshevCRActivation, channel: int,
                        size: int = 2048) -> Wavetable:
    """Read channel ``channel``'s Chebyshev-CR curve as a single-cycle waveform."""
    n_ch = cell.coef.shape[0]
    if not 0 <= channel < n_ch:
        raise ValueError(f"channel {channel} out of range [0,{n_ch})")
    t = torch.linspace(-1.0, 1.0, size)
    x = t.unsqueeze(-1).expand(size, n_ch)
    with torch.no_grad():
        y = cell(x)[:, channel].numpy()
    return Wavetable(y, name=f"cheby[{channel}]")


def wavelet_wavetable(kind: str = "ricker", *, size: int = 2048,
                      cycles: float = 3.0, span: float = 5.0) -> Wavetable:
    """A wavelet as a single-cycle waveform.

    ``ricker`` (Mexican hat, ``(1-τ²)e^{-τ²/2}``) — soft, percussive, few
    harmonics. ``morlet`` (Gaussian-windowed cosine of ``cycles`` periods) —
    harmonically richer, vocal-ish. ``span`` sets the support in std-devs.
    """
    t = np.linspace(-span, span, size, dtype=np.float32)
    if kind == "ricker":
        w = (1.0 - t ** 2) * np.exp(-t ** 2 / 2.0)
    elif kind == "morlet":
        env = np.exp(-(t ** 2) / (2.0 * (span / 2.0) ** 2))
        w = np.cos(2.0 * np.pi * cycles * t / (2.0 * span)) * env
    else:
        raise ValueError(f"unknown wavelet {kind!r}; choose 'ricker' or 'morlet'")
    return Wavetable(w, name=f"wavelet[{kind}]")


def morph(a: Wavetable, b: Wavetable, frac: float, *, size: int = 2048) -> Wavetable:
    """Blend two tables: ``(1-frac)·a + frac·b`` (resampled to ``size``)."""
    frac = float(np.clip(frac, 0.0, 1.0))
    grid = np.linspace(0.0, 1.0, size, dtype=np.float32)
    ra = np.interp(grid, np.linspace(0, 1, a.size), a.table)
    rb = np.interp(grid, np.linspace(0, 1, b.size), b.table)
    return Wavetable((1.0 - frac) * ra + frac * rb,
                     name=f"morph({a.name},{b.name},{frac:.2f})")


def chebyshev_bank(n_channels: int = 8, *, degree: int = 7, grid: int = 16,
                   seed: int = 0, size: int = 2048) -> list[Wavetable]:
    """A bank of ``n_channels`` Chebyshev-CR wavetables (one per cell channel)."""
    torch.manual_seed(seed)
    cell = ChebyshevCRActivation(n_channels, grid=grid, k=degree)
    return [chebyshev_wavetable(cell, c, size=size) for c in range(n_channels)]


def _atanh(v: float) -> float:
    return float(np.arctanh(np.clip(v, -0.999, 0.999)))


def kochanek_bartels_wavetable(control_points, *, tension: float = 0.0,
                               continuity: float = 0.0, bias: float = 0.0,
                               size: int = 2048) -> Wavetable:
    """Interpolate ``control_points`` (one period) with a Kochanek-Bartels (TCB)
    spline — tension/continuity/bias ∈ (-1, 1) shape the tangents at each knot.

    ``(t, c, b) = (0, 0, 0)`` reduces to Catmull-Rom exactly (the CR baseline).
    Tension sharpens, continuity kinks, bias skews the waveform — three extra
    timbre dimensions. Reuses the repo's ``_kb_eval`` (no reimplemented spline).

    Preconditions: ``len(control_points) >= 4``.
    """
    from signedkan_wip.src.core.splines import _kb_eval
    cp = torch.as_tensor(np.asarray(control_points, dtype=np.float32)).ravel()
    grid = cp.numel()
    if grid < 4:
        raise ValueError(f"need >= 4 control points; got {grid}")
    x = torch.linspace(-1.0, 1.0, size)
    coef = cp.view(1, grid)
    tcb_raw = torch.zeros(1, grid, 3)
    tcb_raw[..., 0] = _atanh(tension)
    tcb_raw[..., 1] = _atanh(continuity)
    tcb_raw[..., 2] = _atanh(bias)
    with torch.no_grad():
        y = _kb_eval(coef, tcb_raw, x, grid).numpy()
    return Wavetable(y, name=f"kb(t={tension:+.2f},c={continuity:+.2f},b={bias:+.2f})")


def kb_bank(n_channels: int = 2, *, tension: float = 0.0, continuity: float = 0.0,
            bias: float = 0.0, degree: int = 7, grid: int = 16, seed: int = 0,
            size: int = 2048) -> list[Wavetable]:
    """A KB-shaped bank: take each Chebyshev-CR channel's control points and
    re-interpolate with the given TCB (tension/continuity/bias as timbre knobs)."""
    torch.manual_seed(seed)
    cell = ChebyshevCRActivation(n_channels, grid=grid, k=degree)
    cps = cell.control_points().detach().numpy()
    return [kochanek_bartels_wavetable(cps[c], tension=tension, continuity=continuity,
                                       bias=bias, size=size)
            for c in range(n_channels)]
