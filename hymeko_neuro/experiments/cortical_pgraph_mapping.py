"""Map GömbSoma cortical-benchmark P-graph unit names →
``run_cortical_benchmark`` SimpleExperiment kwargs.

The sweep fixture ``data/hsikan/sweep_msg_cortical.hymeko`` describes
the cortical-benchmark architecture-search as a P-graph with three
orthogonal axes:

  * backbone width   ``d_hidden_{4,8,16}``  → ``d_hidden``
  * binning depths   ``binning_shallow`` / ``binning_deep``
                     → ``binning`` (resolved to BinningConfig below)
  * BrainScorer rank ``pls_25`` / ``pls_50``
                     → ``n_pls_components``

Phase 12 (2026-05-20) shipped this module alongside the parallel
HSIKAN and Gömb mappings (``hsikan_pgraph_mapping.py`` /
``gomb_pgraph_mapping.py``) so the cortical sweep driver consumes
the ABB-selected unit set in the same way.
"""
from __future__ import annotations

from typing import Any


CORTICAL_UNIT_TO_KNOBS: dict[str, dict[str, Any]] = {
    # Backbone-width axis — ResNetTinyCortical.d_hidden.
    "d_hidden_4":  {"d_hidden":  4, "backbone": "resnet"},
    "d_hidden_8":  {"d_hidden":  8, "backbone": "resnet"},
    "d_hidden_16": {"d_hidden": 16, "backbone": "resnet"},
    # Phase 12.5 (2026-05-20): GömbSoma RicciStimBackbone units —
    # the hypergraph-machine CV branch parallel to the ResNet
    # baseline. Same d_hidden semantics; the `backbone` flag tells
    # the driver to instantiate the hypergraph backbone instead.
    "gomb_d4":     {"d_hidden":  4, "backbone": "gomb"},
    "gomb_d8":     {"d_hidden":  8, "backbone": "gomb"},
    "gomb_d16":    {"d_hidden": 16, "backbone": "gomb"},
    # Binning-depth axis — string flag the driver resolves into a
    # concrete BinningConfig.
    "binning_shallow": {"binning": "shallow"},
    "binning_deep":    {"binning": "deep"},
    # BrainScorer PLS-rank axis.
    "pls_25": {"n_pls_components": 25},
    "pls_50": {"n_pls_components": 50},
}


def merge_structure_knobs(unit_names: list[str]) -> dict[str, Any]:
    """Merge knob dicts for all units in one ABB / SSG solution.

    Later keys win on collision (prefer the last unit in
    ``unit_names``). Raises ``KeyError`` on any unrecognised name —
    same contract as the HSIKAN and Gömb mappings.
    """
    merged: dict[str, Any] = {}
    for raw in unit_names:
        u = raw.strip()
        patch = CORTICAL_UNIT_TO_KNOBS.get(u)
        if patch is None:
            raise KeyError(
                f"unknown cortical P-graph unit {u!r}; add it to "
                f"CORTICAL_UNIT_TO_KNOBS in cortical_pgraph_mapping.py"
            )
        merged.update(patch)
    return merged


def benchmark_kwargs(
    *,
    seed: int,
    structure: dict[str, Any],
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build kwargs for the cortical-benchmark
    :class:`CorticalBenchmarkExperiment.run_seed` (Slice 1's
    SimpleExperiment subclass).

    ``base`` may set defaults the structure didn't override
    (e.g.\\ ``n_images``, ``n_subjects``, ``image_size``).
    """
    base = dict(base or {})
    d_hidden = int(structure.get("d_hidden", base.get("d_hidden", 8)))
    binning = str(structure.get("binning", base.get("binning", "deep")))
    n_pls = int(structure.get("n_pls_components",
                              base.get("n_pls_components", 25)))
    backbone = str(structure.get("backbone",
                                 base.get("backbone", "resnet")))
    return {
        "seed": seed,
        "n_images":   int(base.get("n_images",   20)),
        "n_subjects": int(base.get("n_subjects",  4)),
        "image_size": int(base.get("image_size", 32)),
        "in_channels": int(base.get("in_channels", 1)),
        "d_hidden":   d_hidden,
        "snr":        float(base.get("snr", 0.3)),
        "n_pls_components": n_pls,
        "n_cv_folds": int(base.get("n_cv_folds", 4)),
        "binning":    binning,
        "backbone":   backbone,
    }


__all__ = [
    "CORTICAL_UNIT_TO_KNOBS",
    "merge_structure_knobs",
    "benchmark_kwargs",
]
