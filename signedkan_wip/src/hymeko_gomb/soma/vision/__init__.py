"""GömbSoma — vision subpackage.

First sensorimotor-stack application: encode an image as a signed
4-connected patch graph, run WalkConvLayer over the patch walks,
classify.

The encoder turns an (C, H, W) image into:
  * vertex features: one per patch (linear-projected flattened patch),
  * 4-connected edges: between spatially-adjacent patches,
  * edge signs: +1 if src patch is brighter than dst, -1 otherwise,
  * walks: all length-2 walks (3 vertices, 2 edges) over the grid,
  * walk signs: σ-product of the two constituent edge signs,
  * M_v: sparse vertex-to-walk incidence.

The grid topology is shared across the batch; only signs and
features differ per image, so the topology is precomputed once.

Phase 3-V — plan: docs/plans/2026-05-14-gomb-soma/.
"""
from __future__ import annotations

from signedkan_wip.src.hymeko_gomb.soma.vision.forman import (
    FormanCurvature,
    FormanCurvatureHead,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.hodge import (
    HodgeLaplacian,
    HodgeOperators,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.patch_graph import (
    PatchGraphBuilder,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.quadtree import (
    AdaptiveQuadtree,
    AnchorTree,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.sdrf import (
    SDRFOutput,
    SDRFRewiring,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_backbone import (
    RicciStimBackbone,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_classifier import (
    RicciStimClassifier,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.ricci_stim_detector import (
    DetectionOutput,
    RicciStimDetector,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.stim_graph import (
    StimulusGraph,
    StimulusGraphBuilder,
)
from signedkan_wip.src.hymeko_gomb.soma.vision.walk_conv_classifier import (
    WalkConvImageClassifier,
)

__all__ = [
    "AdaptiveQuadtree",
    "AnchorTree",
    "DetectionOutput",
    "FormanCurvature",
    "FormanCurvatureHead",
    "HodgeLaplacian",
    "HodgeOperators",
    "PatchGraphBuilder",
    "RicciStimBackbone",
    "RicciStimClassifier",
    "RicciStimDetector",
    "SDRFOutput",
    "SDRFRewiring",
    "StimulusGraph",
    "StimulusGraphBuilder",
    "WalkConvImageClassifier",
]
