"""Triton fused kernels for HSiKAN — split out of the 1345-LOC
`triton_kernels.py` monolith 2026-05-11 per CLAUDE.md §6.5 #4.

Sub-modules (all ≤ 300 LOC, by concern):

    dispatch.py                  runtime gate for the Triton backward path
    catmull_rom.py               Catmull-Rom spline kernel + autograd wrapper
    inner.py                     plain SignedKAN inner forward kernel
    inner_highway.py             SignedKAN inner forward + Highway gate
    inner_backward.py            plain inner backward (closed-form)
    inner_highway_backward.py    inner-highway backward (closed-form)
    autograd.py                  torch.autograd.Function wrappers + install
    debug.py                     parity-check + benchmark utilities

External imports keep the flat `from .triton_kernels import X` shape via
the re-exports below.
"""
from __future__ import annotations

# ── Forward kernels ────────────────────────────────────────────────
from .catmull_rom import (
    catmull_rom_triton,
    catmull_rom_triton_autograd,
    install_triton_catmull_rom,
    uninstall_triton_catmull_rom,
)
from .inner import signedkan_inner_triton
from .inner_highway import signedkan_inner_highway_triton

# ── Backward kernels ───────────────────────────────────────────────
from .inner_backward import signedkan_inner_backward_triton
from .inner_highway_backward import signedkan_inner_highway_backward_triton

# ── torch.autograd.Function wrappers (user-facing) ─────────────────
from .autograd import (
    signedkan_inner_triton_autograd,
    signedkan_inner_highway_triton_autograd,
)

# ── Runtime gate ───────────────────────────────────────────────────
from .dispatch import _triton_backward_enabled

__all__ = [
    "catmull_rom_triton",
    "catmull_rom_triton_autograd",
    "install_triton_catmull_rom",
    "uninstall_triton_catmull_rom",
    "signedkan_inner_triton",
    "signedkan_inner_highway_triton",
    "signedkan_inner_backward_triton",
    "signedkan_inner_highway_backward_triton",
    "signedkan_inner_triton_autograd",
    "signedkan_inner_highway_triton_autograd",
    "_triton_backward_enabled",
]
