"""Unit tests for the GömbSoma cortical-benchmark P-graph
unit→config mapping (Phase 12, 2026-05-20).
"""
from __future__ import annotations

import pytest

from signedkan_wip.src.cortical_pgraph_mapping import (
    CORTICAL_UNIT_TO_KNOBS,
    benchmark_kwargs,
    merge_structure_knobs,
)


def test_cost_minimum_path_resolves_to_d4_shallow_pls25():
    merged = merge_structure_knobs(
        ["d_hidden_4", "binning_shallow", "pls_25"]
    )
    assert merged["d_hidden"] == 4
    assert merged["binning"] == "shallow"
    assert merged["n_pls_components"] == 25


def test_quality_path_resolves_to_d16_deep_pls50():
    merged = merge_structure_knobs(
        ["d_hidden_16", "binning_deep", "pls_50"]
    )
    assert merged["d_hidden"] == 16
    assert merged["binning"] == "deep"
    assert merged["n_pls_components"] == 50


def test_unknown_unit_raises():
    with pytest.raises(KeyError, match="unknown cortical P-graph unit"):
        merge_structure_knobs(["NotARealUnit"])


def test_benchmark_kwargs_overrides_base_with_structure():
    base = {"d_hidden": 999, "n_pls_components": 999, "n_images": 30}
    merged = merge_structure_knobs(
        ["d_hidden_8", "binning_deep", "pls_25"]
    )
    kw = benchmark_kwargs(seed=7, structure=merged, base=base)
    assert kw["seed"] == 7
    # Structure-derived values override base.
    assert kw["d_hidden"] == 8
    assert kw["n_pls_components"] == 25
    assert kw["binning"] == "deep"
    # Base values come through where structure is silent.
    assert kw["n_images"] == 30
    # Defaults for unmentioned axes.
    assert kw["n_subjects"] == 4
    assert kw["image_size"] == 32


def test_all_seven_units_are_registered():
    expected = {
        "d_hidden_4", "d_hidden_8", "d_hidden_16",
        "binning_shallow", "binning_deep",
        "pls_25", "pls_50",
    }
    assert expected.issubset(CORTICAL_UNIT_TO_KNOBS.keys())
