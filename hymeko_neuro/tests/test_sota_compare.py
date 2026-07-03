"""Tests for the slide-18 accuracy-vs-cost loaders, Pareto, and table (pure).

No torch: these exercise only the committed-json loaders and the shaping/render
helpers. Run with::

    python -m pytest -p no:randomly hymeko_neuro/tests/test_sota_compare.py
"""
from __future__ import annotations

import json

import pytest

from hymeko_neuro.experiments.runs.bench_to_png import (
    pareto_points, render_pareto, table_rows, write_table,
)
from hymeko_neuro.experiments.runs.sota_compare import (
    AccuracyRecord, load_optuna_best, load_phase8_accuracy, merge_records,
)


# --------------------------------------------------------------------------- #
# loaders                                                                      #
# --------------------------------------------------------------------------- #
def _phase8_fixture(tmp_path):
    runs = []
    for seed, auc in enumerate([0.90, 0.92, 0.94]):  # sgcn, 3 seeds
        runs.append(dict(dataset="bitcoin_otc", arch="sgcn_balance", seed=seed,
                         test_auc=auc, n_params=200_000))
    runs.append(dict(dataset="bitcoin_otc", arch="mystery_net", seed=0,
                     test_auc=0.99, n_params=10))  # not in map → dropped
    runs.append(dict(dataset="bitcoin_otc", arch="gcn_blind", seed=0,
                     test_auc=None, n_params=5))    # null auc → skipped
    p = tmp_path / "phase8.json"
    p.write_text(json.dumps({"runs": runs}))
    return p


def test_phase8_mean_std_and_seedcount(tmp_path):
    recs = load_phase8_accuracy(_phase8_fixture(tmp_path))
    r = recs[("bitcoin_otc", "SGCN")]
    assert r.mean_auc == pytest.approx(0.92)
    assert r.n_seeds == 3
    assert r.n_params == 200_000
    assert r.tuned is False


def test_phase8_drops_unmapped_and_null(tmp_path):
    recs = load_phase8_accuracy(_phase8_fixture(tmp_path))
    methods = {m for (_ds, m) in recs}
    assert "SGCN" in methods
    # mystery_net unmapped, gcn_blind had only a null-auc run → neither present.
    assert not any(m not in {"SGCN"} for m in methods)


def test_optuna_loader_tuned_field(tmp_path):
    lines = [json.dumps(dict(dataset="bitcoin_otc", seed=s, auc=0.99, n_params=23_815))
             for s in range(5)]
    p = tmp_path / "optuna.jsonl"
    p.write_text("\n".join(lines))
    recs = load_optuna_best(p)
    r = recs[("bitcoin_otc", "HSiKAN-optuna")]
    assert r.tuned is True and r.n_seeds == 5
    assert r.mean_auc == pytest.approx(0.99)


def test_merge_later_wins():
    a = {("d", "X"): AccuracyRecord("d", "X", 0.1, 0, 1, 1, False, "a")}
    b = {("d", "X"): AccuracyRecord("d", "X", 0.9, 0, 1, 1, True, "b")}
    assert merge_records(a, b)[("d", "X")].mean_auc == 0.9


# --------------------------------------------------------------------------- #
# pareto + table shaping                                                       #
# --------------------------------------------------------------------------- #
def _records():
    mk = lambda m, auc, p, tuned: AccuracyRecord(  # noqa: E731
        "bitcoin_otc", m, auc, 0.005, p, 5, tuned, "src")
    return {
        ("bitcoin_otc", "SGCN"): mk("SGCN", 0.942, 202_721, False),
        ("bitcoin_otc", "HSiKAN-leanest"): mk("HSiKAN-leanest", 0.851, 94_627, False),
        ("bitcoin_otc", "HSiKAN-optuna"): mk("HSiKAN-optuna", 0.993, 23_815, True),
        ("bitcoin_alpha", "SGCN"): mk("SGCN", 0.929, 135_585, False),
    }


def test_pareto_points_filtered_and_sorted():
    pts = pareto_points(_records(), "bitcoin_otc")
    assert [p.method for p in pts] == ["HSiKAN-optuna", "HSiKAN-leanest", "SGCN"]
    # no bitcoin_alpha leakage
    assert all(p.dataset == "bitcoin_otc" for p in pts)


def test_render_pareto_writes_png(tmp_path):
    out = render_pareto(pareto_points(_records(), "bitcoin_otc"),
                        "bitcoin_otc", tmp_path / "pareto.png")
    assert out.exists() and out.stat().st_size > 0


def test_render_pareto_rejects_empty():
    with pytest.raises(AssertionError):
        render_pareto([], "bitcoin_otc", "unused.png")


def _latency_rows():
    return [
        dict(dataset="bitcoin_otc", device="cpu", model="SGCN",
             fwd_med_ms=20.5, n_params=202_721),
        dict(dataset="bitcoin_otc", device="cuda", model="SGCN",
             fwd_med_ms=4.9, n_params=202_721),
        dict(dataset="bitcoin_otc", device="cpu", model="SGT",
             fwd_med_ms=33.0, n_params=180_000),  # latency only, no accuracy
    ]


def test_table_rows_union_and_missing_axes():
    rows = table_rows(_latency_rows(), _records(), "bitcoin_otc")
    by = {r["method"]: r for r in rows}
    # SGT has latency but no committed accuracy → AUC None.
    assert by["SGT"]["mean_auc"] is None
    assert by["SGT"]["fwd_cpu_ms"] == 33.0
    # HSiKAN-optuna has accuracy but no latency here → fwd None.
    assert by["HSiKAN-optuna"]["fwd_cpu_ms"] is None
    assert by["HSiKAN-optuna"]["tuned"] is True
    # sorted ascending by params: optuna (23.8k) before SGCN (202k).
    params = [r["n_params"] for r in rows]
    assert params == sorted(params)


def test_write_table_smoke(tmp_path):
    out = write_table(_latency_rows(), _records(), "bitcoin_otc",
                      tmp_path / "t.md")
    txt = out.read_text(encoding="utf-8")
    assert "| method |" in txt and "HSiKAN-optuna" in txt and "—" in txt
