from __future__ import annotations

import torch

from scripts.dev.hymeko_structure_toy_models import (
    benchmark_prepared_suite,
    load_toy_graph,
    prepare_toy_suite,
    run_general_problem_suite,
    run_prepared_toy_suite,
    run_supervised_toy_suite,
    run_toy_suite,
)


def test_hymeko_structure_toy_projection_counts():
    graph = load_toy_graph()
    assert graph.names == ["v0", "v1", "v2", "v3", "v4", "v5"]
    assert graph.features.shape == (6, 4)
    assert graph.tier_of.tolist() == [0, 0, 1, 1, 2, 2]
    assert graph.cycles.shape == (4, 3)
    assert graph.walks.shape == (4, 3)
    assert graph.edges.shape == (3, 2)
    assert graph.edge_signs.tolist() == [-1, 1, -1]
    assert graph.sequence_order == [0, 1, 2, 3, 4, 5]


def test_hymeko_structure_toy_models_all_run_cpu():
    result = run_toy_suite()
    assert result["n_nodes"] == 6
    assert result["n_cycles"] == 4
    assert result["n_walks"] == 4
    assert result["n_edges"] == 3
    assert result["sequence_len"] == 6
    assert result["hsikan"]["shape"] == [6, 8]
    assert result["fixed_hsikan"]["shape"] == [6, 8]
    assert result["fixed_hsikan"]["max_abs_delta_vs_hsikan"] < 1e-6
    assert result["sa_hsikan"]["shape"] == [6, 8]
    assert result["gomb"]["shape"] == [3]
    assert result["fixed_gomb"]["shape"] == [3]
    assert result["fixed_gomb"]["max_abs_delta_vs_gomb"] < 1e-6
    assert result["gomb_soma"]["shape"] == [6, 8]
    assert result["fsr"]["shape"] == [1, 6, 6]
    for key in ("hsikan", "fixed_hsikan", "sa_hsikan", "gomb", "gomb_soma", "fsr"):
        assert result[key]["finite"] is True
        assert result[key]["n_params"] > 0
    assert torch.isfinite(torch.tensor(result["gomb"]["scores"])).all()


def test_hymeko_structure_toy_prepared_suite_hot_path_benchmarks():
    suite = prepare_toy_suite()
    result = run_prepared_toy_suite(suite)
    assert result["hsikan"]["shape"] == [6, 8]
    assert result["fixed_hsikan"]["shape"] == [6, 8]
    assert result["gomb"]["shape"] == [3]
    assert result["gomb_soma"]["shape"] == [6, 8]
    assert result["fsr"]["shape"] == [1, 6, 6]

    perf = benchmark_prepared_suite(repeats=2, warmup=1)
    assert perf["hot_all_forwards"]["mean_us"] > 0
    assert set(perf["per_adapter"]) == {
        "hsikan",
        "fixed_hsikan",
        "sa_hsikan",
        "gomb",
        "fixed_gomb",
        "gomb_soma",
        "fsr",
    }


def test_hymeko_structure_toy_supervised_probes_run():
    result = run_supervised_toy_suite()
    assert result["n_nodes"] == 6
    expected = {"raw_features", "hsikan", "fixed_hsikan", "sa_hsikan", "gomb_soma", "fsr"}
    assert set(result["classification"]["models"]) == expected
    assert set(result["regression"]["models"]) == expected
    for row in result["classification"]["models"].values():
        assert 0.0 <= row["train_acc"] <= 1.0
        assert 0.0 <= row["loo_acc"] <= 1.0
        assert len(row["pred"]) == 6
        assert len(row["loo_pred"]) == 6
    for row in result["regression"]["models"].values():
        assert row["train_rmse"] >= 0.0
        assert row["loo_rmse"] >= 0.0
        assert len(row["pred"]) == 6
        assert len(row["loo_pred"]) == 6


def test_hymeko_structure_general_problem_runs():
    result = run_general_problem_suite(n_samples=18, train_samples=12, timing_repeats=2)
    assert result["n_samples"] == 18
    assert result["train_samples"] == 12
    assert result["test_samples"] == 6
    assert "mean, std, max" in result["global_pool"]
    assert "entropy" in result["entropy_feedback"]
    assert set(result["structure_search"]) >= {"msg", "ssg", "abb", "targets"}
    assert "alpha-mixing branch pruning" in result["structure_search"]["targets"]
    assert result["closed_form_entropy_feedback"]["training_loop"] == "none"
    assert result["fsr_clifford"]["algebra"].endswith("Cl(0,2)+")
    assert result["fsr_clifford"]["rotor"]["unit_norm_max_abs_error"] < 1e-6
    assert result["fsr_clifford"]["transport_slots"]["sparse_vs_dense_ratio"] < 1.0
    for row in result["models"].values():
        assert row["pooled_dim"] > 0
        assert row["forward_pool_us"]["mean_us"] > 0
        assert 0.0 <= row["classification"]["plain"]["acc"] <= 1.0
        assert 0.0 <= row["classification"]["entropy_feedback"]["acc"] <= 1.0
        assert row["regression"]["plain"]["rmse"] >= 0.0
        assert row["regression"]["entropy_feedback"]["rmse"] >= 0.0
