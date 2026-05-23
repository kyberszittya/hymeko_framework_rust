"""Unit tests for the HSIKAN P-graph unit→config mapping (Phase 7).

Pins the translation table that converts ABB-selected operating-unit
names from ``sweep_msg.hymeko`` (or the by-product variant) onto
``run_compare.run_one`` kwargs.
"""
from __future__ import annotations

import pytest

from signedkan_wip.src.hsikan_pgraph_mapping import (
    HSIKAN_UNIT_TO_KNOBS,
    merge_structure_knobs,
    run_one_kwargs,
)


def test_cheap_path_merges_cycle_topk_m4_model_h8_train_short():
    """The cost-minimum ABB selection on `sweep_msg.hymeko` should
    map to (m_cycles=4, hidden=8, n_epochs=10)."""
    merged = merge_structure_knobs(["cycle_topk_m4", "model_h8", "train_short"])
    assert merged["m_cycles"] == 4
    assert merged["hidden"] == 8
    assert merged["n_epochs"] == 10


def test_strict_mode_byproduct_path_picks_m16():
    """Under Phase 6's strict-no-excess + by-product variant, ABB
    drops cycle_topk_m4 and picks cycle_topk_m16. The mapping must
    translate this to (m_cycles=16, hidden=8, n_epochs=10)."""
    merged = merge_structure_knobs(["cycle_topk_m16", "model_h8", "train_short"])
    assert merged["m_cycles"] == 16
    assert merged["hidden"] == 8
    assert merged["n_epochs"] == 10


def test_long_training_mode():
    merged = merge_structure_knobs(["cycle_topk_m64", "model_h32", "train_long"])
    assert merged["m_cycles"] == 64
    assert merged["hidden"] == 32
    assert merged["n_epochs"] == 60


def test_unknown_unit_raises():
    with pytest.raises(KeyError, match="unknown HSIKAN P-graph unit"):
        merge_structure_knobs(["NotAUnit"])


def test_run_one_kwargs_uses_structure_over_base():
    base = {"hidden": 999, "n_epochs": 999, "lr": 1e-2}
    merged = merge_structure_knobs(["cycle_topk_m4", "model_h8", "train_short"])
    kw = run_one_kwargs(dataset="bitcoin_alpha", seed=7, structure=merged, base=base)
    assert kw["dataset"] == "bitcoin_alpha"
    assert kw["seed"] == 7
    # Structure-derived values must override base.
    assert kw["hidden"] == 8
    assert kw["n_epochs"] == 10
    assert kw["m_cycles"] == 4
    # Base-only values come through.
    assert kw["lr"] == pytest.approx(1e-2)
    assert kw["model_name"] == "signedkan"


def test_run_one_kwargs_defaults_when_structure_silent():
    """A degenerate selection with no structure-derived knobs must
    fall back to base defaults."""
    kw = run_one_kwargs(dataset="bitcoin_alpha", seed=0, structure={}, base={})
    assert kw["hidden"] == 16
    assert kw["n_epochs"] == 60
    assert kw["m_cycles"] == 16


def test_all_eight_units_are_registered():
    """Every unit in the canonical HSIKAN sweep file must be in the
    mapping table; if a future sweep variant adds units the mapping
    must be extended alongside."""
    expected = {
        "cycle_topk_m4", "cycle_topk_m16", "cycle_topk_m64",
        "model_h8", "model_h16", "model_h32",
        "train_short", "train_long",
    }
    assert expected.issubset(HSIKAN_UNIT_TO_KNOBS.keys())
