"""Unit tests for ``run_hsikan_optuna_chase`` helpers (no Optuna runs)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from signedkan_wip.src.benchmarks.run_hsikan_optuna_chase import (
    competitor_target_auc,
    load_sota_reference,
)


def _reference_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "src"
        / "benchmarks"
        / "sota_reference.json"
    )


def test_load_sota_reference_round_trip():
    ref = load_sota_reference()
    assert ref["metric"] == "roc_auc_test_edges"
    assert "bitcoin_otc" in ref["targets"]


def test_competitor_target_auc_matches_json_file():
    raw = json.loads(_reference_path().read_text(encoding="utf-8"))
    t_otc, n_otc = competitor_target_auc(raw, "bitcoin_otc")
    assert n_otc == "SiGAT"
    assert t_otc == pytest.approx(0.934)
    t_ba, n_ba = competitor_target_auc(raw, "bitcoin_alpha")
    assert n_ba == "SiGAT"
    assert t_ba == pytest.approx(0.899)


def test_competitor_target_unknown_dataset_raises():
    ref = load_sota_reference()
    with pytest.raises(KeyError):
        competitor_target_auc(ref, "not_a_dataset")
