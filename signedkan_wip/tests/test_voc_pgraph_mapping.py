"""Unit tests for the HyMeYOLO VOC Stage-D P-graph mapping
(Phase 15, 2026-05-20 overnight).
"""
from __future__ import annotations

import pytest

from signedkan_wip.src.voc_pgraph_mapping import (
    VOC_UNIT_TO_KNOBS,
    merge_structure_knobs,
    train_voc_kwargs,
)


def test_cost_minimum_path_resolves_to_resnet_q4_lam_low():
    merged = merge_structure_knobs(
        ["backbone_resnet", "q4", "lam_low"]
    )
    assert merged["backbone"] == "resnet"
    assert merged["n_box_queries"] == 4
    assert merged["lam_no_obj"] == 0.5


def test_quality_path_resolves_to_imagenet_q12_lam_high():
    merged = merge_structure_knobs(
        ["backbone_resnet18_imagenet", "q12", "lam_high"]
    )
    assert merged["backbone"] == "resnet18_imagenet"
    assert merged["n_box_queries"] == 12
    assert merged["lam_no_obj"] == 2.0


def test_hsikan_backbone_resolves():
    """Phase 9 (wheel rebuild) made hsikan an actual valid backbone
    in train_voc_stagec.py. Confirm the mapping wires it."""
    merged = merge_structure_knobs(
        ["backbone_hsikan", "q8", "lam_low"]
    )
    assert merged["backbone"] == "hsikan"


def test_unknown_unit_raises():
    with pytest.raises(KeyError, match="unknown VOC P-graph unit"):
        merge_structure_knobs(["NotARealUnit"])


def test_train_voc_kwargs_overrides_base_with_structure():
    base = {"backbone": "fake", "lam_no_obj": 99.0, "epochs": 5}
    merged = merge_structure_knobs(
        ["backbone_hsikan", "q8", "lam_low"]
    )
    kw = train_voc_kwargs(seed=7, structure=merged, base=base)
    assert kw["seed"] == 7
    # Structure-derived values override base.
    assert kw["backbone"] == "hsikan"
    assert kw["lam_no_obj"] == 0.5
    assert kw["n_box_queries"] == 8
    # Base values come through where structure is silent.
    assert kw["epochs"] == 5
    # Defaults.
    assert kw["year"] == "2007"
    assert kw["query_head_kind"] == "hungarian"


def test_all_eight_units_are_registered():
    expected = {
        "backbone_resnet", "backbone_resnet18_imagenet", "backbone_hsikan",
        "q4", "q8", "q12",
        "lam_low", "lam_high",
    }
    assert expected.issubset(VOC_UNIT_TO_KNOBS.keys())
