"""GömbSoma — compositional hypergraph hierarchy for the sensorimotor stack.

Companion to the Gömb cognitive-stack architecture (three orthogonal
shells: OuterFIRShell, MiddleHSiKAN, InnerCPMLCore). Where Gömb operates
on already-abstracted representations, GömbSoma builds structure
compositionally from raw input via the ladder:

    walks (open) → polygons (closed k≥4) → triangles (c3 + Cartwright–Harary)
        → derivative-nodelet contraction → recurse on coarsened graph

with Clifford-FIR as the inter-layer grade-lifting transfer.

Plan: docs/plans/2026-05-14-gomb-soma/.

Phase 1 (this commit): HypergraphConv ABC. The shared message-passing
primitive that all GömbSoma layer types implement.
"""
from __future__ import annotations

from signedkan_wip.src.hymeko_gomb.soma.hg_conv import (
    HypergraphConv,
    HypergraphConvConfig,
)
from signedkan_wip.src.hymeko_gomb.soma.hg_conv_bochner import (
    BochnerHypergraphConv,
)
from signedkan_wip.src.hymeko_gomb.soma.polygon_layer import PolygonConvLayer
from signedkan_wip.src.hymeko_gomb.soma.walk_layer import WalkConvLayer

__all__ = [
    "BochnerHypergraphConv",
    "HypergraphConv",
    "HypergraphConvConfig",
    "PolygonConvLayer",
    "WalkConvLayer",
]
