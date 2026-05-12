"""HymeKo-Gömb (sphere) — three-shell concentric cascade for
signed-hypergraph link prediction.

Architecture (plan: docs/plans/2026-05-11-hymeko-gomb-sphere/):

    Outer shell  →  Volume of M parallel Clifford-FIR filter banks.
                     Cheap, redundant, sign-aware. Bio-analogue: V1.
    Middle shell →  Single HSiKAN spline layer with Catmull-Rom gates.
                     Bio-analogue: V4.
    Inner core   →  CPML tier-stratified topology compression toward
                     the predictor. Bio-analogue: IT.

Package layout:

    shells.py    OuterFIRShell, MiddleHSiKAN, InnerCPMLCore, scatter_mean
                 (the role-distinct primitives)
    cascade.py   GombConfig, HymeKoGomb (full cascade),
                 GombNoOuter / GombNoMiddle / GombNoInner (ablations),
                 MixedArityGomb (k=3+k=4, k=4+k=5, … with learned αₖ)

External imports stay flat: `from signedkan_wip.src.hymeko_gomb import …`
continues to work because this module re-exports the public surface.
"""
from __future__ import annotations

from .cascade import (
    GombConfig,
    GombNoInner,
    GombNoMiddle,
    GombNoOuter,
    HymeKoGomb,
    MixedArityGomb,
)
from .shells import InnerCPMLCore, MiddleHSiKAN, OuterFIRShell, scatter_mean

__all__ = [
    "GombConfig",
    "HymeKoGomb",
    "OuterFIRShell", "MiddleHSiKAN", "InnerCPMLCore", "scatter_mean",
    "GombNoOuter", "GombNoMiddle", "GombNoInner",
    "MixedArityGomb",
]
