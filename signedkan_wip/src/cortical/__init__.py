"""Cortical-benchmark package for the GömbSoma vision architecture.

This subpackage implements the cortical-benchmark slice 1 of the
2026-05-16 plan
(``docs/plans/2026-05-16-gomb-soma-cortical-benchmark/``):
score per-image GömbSoma features against per-ROI fMRI-like signals
using a Brain-Score-style PLS+Ridge+noise-ceiling pipeline.

Submodules
----------

* :mod:`.features` — :class:`CorticalFeatureExtractor`,
  :class:`BinningConfig`, :class:`PerDepthFeatures`. Wraps any
  GömbSoma-style backbone (``forward(image) -> (features, tree)``)
  into a per-depth retinotopically-binned feature emitter.
* :mod:`.scoring` — :class:`BrainScorer`, :class:`BrainScore`. The
  Brain-Score-2018 protocol (PLS reduce → Ridge per voxel → K-fold
  CV → split-half noise-ceiling correction), implemented in
  ``signedkan_wip`` rather than via the external ``brainscore``
  package (no network dependency tonight).
* :mod:`.baselines` — :class:`ResNetTinyCortical`,
  :func:`count_parameters`, :func:`assert_param_match`. A parameter-
  matched ResNet-tiny that exposes the same per-depth feature
  interface, so the scorer compares apples to apples.
* :mod:`.synthetic` — :class:`SyntheticCorticalDataset`,
  :func:`make_synthetic_cichy_like`. A shape-faithful Cichy-92
  generator for end-to-end smoke without a real-data dependency.

Real Cichy 92 + Brain-Score-API integration are queued as future
slices; the API here is intentionally dataset-agnostic so the
swap-in is one new adapter function.

Object-oriented overview
------------------------

The pipeline is three frozen-dataclass payloads
(:class:`PerDepthFeatures`, :class:`BrainScore`,
:class:`SyntheticCorticalDataset`) plus three classes:

  Stimuli  ──→  CorticalFeatureExtractor  ──→  PerDepthFeatures
  (image)        OR ResNetTinyCortical                │
                                                      ▼
                                              BrainScorer
                                                      │
                                                      ▼
                                              BrainScore (per ROI)
"""
from .features import (
    BinningConfig,
    CorticalFeatureExtractor,
    PerDepthFeatures,
)
from .scoring import (
    BrainScore,
    BrainScorer,
    score_all_rois,
)
from .baselines import (
    ResNetTinyCortical,
    assert_param_match,
    count_parameters,
)
from .synthetic import (
    SyntheticCorticalDataset,
    make_synthetic_cichy_like,
)

__all__ = [
    # features
    "BinningConfig",
    "CorticalFeatureExtractor",
    "PerDepthFeatures",
    # scoring
    "BrainScore",
    "BrainScorer",
    "score_all_rois",
    # baselines
    "ResNetTinyCortical",
    "assert_param_match",
    "count_parameters",
    # synthetic
    "SyntheticCorticalDataset",
    "make_synthetic_cichy_like",
]
