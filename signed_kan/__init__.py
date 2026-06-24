"""signed_kan — the shared signed-KAN core.

One ``SignedKANLayer`` parameterised over four orthogonal axes (aggregation backend · edge spline · skip/highway ·
incidence), so the dense/inductive RL line and the sparse/transductive vision line are thin configs over one
implementation rather than two forked copies. Phase 1 of the unification plan
(``docs/plans/2026-06-24-unify-hsikan-signed-kan-core``): the core + the dense backend, isolated and tested.

HSiKAN = Highway Signed KAN (the highway gate is the Schmidhuber-style :class:`~signed_kan.layer.HighwaySkip`).
"""
from __future__ import annotations

from .backbone import INCIDENCE_MODES, POOL_MODES, SignedKANBackbone
from .backends import AggregationBackend, DenseBatchedBackend, SparseSignedBackend
from .graph_model import SignedGraphHSiKAN
from .heads import EdgeSignHead, GraphHead
from .layer import SKIP_MODES, HighwaySkip, SignedKANLayer
from .splines import CatmullRomActivation, EdgeActivation, catmull_rom, make_activation

__all__ = [
    "SignedKANBackbone", "INCIDENCE_MODES", "POOL_MODES",
    "SignedKANLayer", "SKIP_MODES", "HighwaySkip",
    "AggregationBackend", "DenseBatchedBackend", "SparseSignedBackend",
    "EdgeSignHead", "GraphHead", "SignedGraphHSiKAN",
    "CatmullRomActivation", "EdgeActivation", "catmull_rom", "make_activation",
]
