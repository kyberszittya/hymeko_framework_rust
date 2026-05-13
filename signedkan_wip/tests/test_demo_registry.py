"""Tests for the demo model registry loader.

Confirms the catalogue YAML parses, every entry has the schema fields the
GUI relies on, and the helpers (`dropdown_choices`, `find_by_id`) behave.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from signedkan_wip.src.demo.registry import (
    DEFAULT_REGISTRY, ModelEntry, dropdown_choices, find_by_id,
    load_registry,
)


def test_default_registry_loads():
    entries = load_registry()
    assert len(entries) >= 1, "models.yaml is empty or unparseable"
    # ids must be unique.
    ids = [e.id for e in entries]
    assert len(set(ids)) == len(ids), f"duplicate ids: {ids}"


def test_every_entry_has_required_fields():
    for e in load_registry():
        assert e.id and isinstance(e.id, str)
        assert e.framework and e.dataset and e.label
        assert isinstance(e.path, Path)
        # train_cmd should be non-empty so a user can reproduce the model.
        assert e.train_cmd, f"entry {e.id} has empty train_cmd"


def test_known_entries_present():
    """Pin the catalogue's expected coverage — fail loudly if entries get
    dropped silently in an edit."""
    ids = {e.id for e in load_registry()}
    for required in (
        "hsikan_bitcoin_alpha_optuna",
        "hsikan_bitcoin_otc_optuna",
        "hsikan_slashdot_edge_cr",
        "gomb_bitcoin_otc_full",
        "gomb_slashdot_full",
    ):
        assert required in ids, f"missing required catalogue entry: {required}"


def test_metrics_match_reports():
    """Numbers in the catalogue should match the memory log."""
    by_id = {e.id: e for e in load_registry()}
    # 10-seed Bitcoin Optuna result (memory: project_bitcoin_optuna_best_10seed_2026_05_13).
    e = by_id["hsikan_bitcoin_alpha_optuna"]
    assert e.auc == pytest.approx(0.9959, abs=1e-4)
    assert e.n_params == 30487
    assert e.n_seeds == 10
    e = by_id["hsikan_bitcoin_otc_optuna"]
    assert e.auc == pytest.approx(0.9933, abs=1e-4)
    assert e.n_params == 23815


def test_dropdown_choices_partition_by_availability(tmp_path):
    """Available entries sort first; missing ones get the [NOT TRAINED] tag."""
    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"")
    e_avail = ModelEntry(
        id="a", framework="HSiKAN", dataset="bitcoin_alpha",
        label="Z available", path=fake, train_cmd="x",
    )
    e_miss = ModelEntry(
        id="b", framework="HSiKAN", dataset="bitcoin_alpha",
        label="A missing", path=tmp_path / "nope.pt", train_cmd="y",
    )
    pairs = dropdown_choices([e_miss, e_avail])
    labels = [p[0] for p in pairs]
    # Available first, then NOT TRAINED.
    assert labels == ["Z available", "[NOT TRAINED] A missing"]


def test_find_by_id_returns_entry_or_none():
    entries = load_registry()
    e = find_by_id(entries, "hsikan_bitcoin_alpha_optuna")
    assert e is not None and e.id == "hsikan_bitcoin_alpha_optuna"
    assert find_by_id(entries, "does_not_exist") is None


def test_missing_registry_returns_empty(tmp_path):
    """Loader degrades gracefully rather than raising."""
    entries = load_registry(tmp_path / "nope.yaml")
    assert entries == []


def test_default_registry_path_is_packaged():
    assert DEFAULT_REGISTRY.is_file(), (
        f"default registry not packaged: {DEFAULT_REGISTRY}"
    )
