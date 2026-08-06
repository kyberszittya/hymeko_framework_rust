"""Tests for the hypergraph-vision vs CNN re-benchmark harness.

Unit  — subset selection, matrix enumeration, aggregation + CNN-gap maths,
        cell-key/dry-run. (Heavy model construction + training is validated
        by the standalone CPU crash-check / GPU smoke, not pytest.)
"""
from __future__ import annotations

import json

from hymeko_neuro.experiments.runs.run_vision_hypergraph_vs_cnn import (
    aggregate,
    cell_key,
    enumerate_cells,
)
from hymeko_neuro.experiments.runs.run_vision_hypergraph_vs_cnn import (
    main as orch_main,
)
from hymeko_neuro.experiments.vision.vision_bench_cell import (
    DATASETS,
    MODEL_NAMES,
    subset_indices,
)


# ----------------------------------------------------------- subset_indices
def test_subset_full_when_zero_or_oversized():
    assert subset_indices(100, 0, 0) == list(range(100))
    assert subset_indices(100, 200, 0) == list(range(100))


def test_subset_size_and_determinism():
    a = subset_indices(1000, 50, seed=7)
    b = subset_indices(1000, 50, seed=7)
    c = subset_indices(1000, 50, seed=8)
    assert len(a) == 50 and a == b      # deterministic per seed
    assert a != c                        # seed changes the draw
    assert all(0 <= i < 1000 for i in a)
    assert len(set(a)) == 50             # no duplicates


# --------------------------------------------------------- matrix / cell key
def test_enumerate_cells_count_and_order():
    cells = enumerate_cells(["cnn", "hgnn"], ["mnist", "fashion"], [0, 1, 2])
    assert len(cells) == 2 * 2 * 3
    assert cells[0] == ("cnn", "mnist", 0)
    assert cell_key("hgnn", "fashion", 2) == "hgnn|fashion|seed2"


def test_model_and_dataset_name_constants():
    assert "ricci_stim" in MODEL_NAMES and "cnn" in MODEL_NAMES
    assert set(DATASETS) == {"mnist", "fashion"}


# --------------------------------------------------------------- aggregate
def test_aggregate_mean_sd_and_cnn_gap():
    rows = []
    accs = {"cnn": 0.99, "hgnn": 0.80, "ricci_stim": 0.70}
    for model, a in accs.items():
        for seed in range(3):
            rows.append({"model": model, "dataset": "mnist", "seed": seed,
                         "test_accuracy": a})
    agg = aggregate(rows)
    assert abs(agg["per_model_dataset"]["cnn|mnist"]["mean"] - 0.99) < 1e-9
    assert agg["per_model_dataset"]["hgnn|mnist"]["n"] == 3
    # CNN gap = cnn_mean - model_mean (positive = below CNN).
    assert abs(agg["cnn_gap"]["mnist"]["hgnn"] - 0.19) < 1e-9
    assert abs(agg["cnn_gap"]["mnist"]["ricci_stim"] - 0.29) < 1e-9
    assert abs(agg["cnn_gap"]["mnist"]["cnn"] - 0.0) < 1e-9


def test_aggregate_skips_failed_cells():
    rows = [
        {"model": "cnn", "dataset": "mnist", "seed": 0, "test_accuracy": 0.99},
        {"model": "hsikan", "dataset": "mnist", "seed": 0, "test_accuracy": None,
         "error": "boom"},
    ]
    agg = aggregate(rows)
    assert "hsikan|mnist" not in agg["per_model_dataset"]
    assert "cnn|mnist" in agg["per_model_dataset"]


# ----------------------------------------------------------------- dry-run
def test_full_cell_key_includes_config_axes():
    """Regression for the 2026-05-29 depth+narrow bug: full_cell_key
    MUST include the config axes that distinguish configs in a
    multi-config sweep (hidden, n_layers, etc.), so subsequent configs
    don't false-positive-skip when sharing a results file."""
    from hymeko_neuro.experiments.runs.run_vision_hypergraph_vs_cnn import (
        full_cell_key,
    )
    k_h8_L2 = full_cell_key(
        {"hidden": 8, "n_layers": 2, "spatial_filter": "none"},
        "hsikan", "mnist", 0,
    )
    k_h8_L4 = full_cell_key(
        {"hidden": 8, "n_layers": 4, "spatial_filter": "none"},
        "hsikan", "mnist", 0,
    )
    k_h16_L2 = full_cell_key(
        {"hidden": 16, "n_layers": 2, "spatial_filter": "none"},
        "hsikan", "mnist", 0,
    )
    assert k_h8_L2 != k_h8_L4, "n_layers axis must distinguish"
    assert k_h8_L2 != k_h16_L2, "hidden axis must distinguish"
    assert "n_layers=2" in k_h8_L2
    assert "hidden=8" in k_h8_L2


def test_orchestrator_dry_run(capsys):
    rc = orch_main(["--dry-run", "--models", "cnn,ricci_stim",
                    "--datasets", "mnist", "--seeds", "0,1"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_cells"] == 4
    assert "ricci_stim|mnist|seed1" in out["cells"]
