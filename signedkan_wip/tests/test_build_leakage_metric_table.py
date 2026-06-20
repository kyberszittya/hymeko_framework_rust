"""Tests for the leakage-metric table builder (signedkan_wip/src/paperkit/build_leakage_metric_table)."""
from __future__ import annotations

import importlib

m = importlib.import_module("signedkan_wip.src.paperkit.build_leakage_metric_table")


def test_agg_mean_and_ci() -> None:
    a = m._agg([0.5, 0.5, 0.5])
    assert abs(a.mean - 0.5) < 1e-12 and a.ci == 0.0 and a.n == 3
    b = m._agg([0.4, 0.6])
    assert abs(b.mean - 0.5) < 1e-12 and b.ci > 0.0


def test_leak_and_drop_definitions() -> None:
    leaking = m.Cell(m._agg([0.9, 0.9, 0.9]), m._agg([0.6, 0.6, 0.6]))
    assert abs(leaking.leak - 0.1) < 1e-9      # L = max(0, shuffle-0.5)
    assert abs(leaking.drop - 0.3) < 1e-9      # Δ = real-shuffle
    assert leaking.leaks                        # shuffle 0.6 > 0.55
    clean = m.Cell(m._agg([0.85, 0.85]), m._agg([0.49, 0.49]))
    assert clean.leak == 0.0 and not clean.leaks   # below chance clamps to 0


def test_seed_means_dedups_last_wins() -> None:
    rows = [
        {"model": "x", "dataset": "d", "seed": 0, "shuffle": False, "auc": 0.1},
        {"model": "x", "dataset": "d", "seed": 0, "shuffle": False, "auc": 0.9},  # re-run, wins
        {"model": "x", "dataset": "d", "seed": 1, "shuffle": False, "auc": 0.8},
    ]
    sm = m._seed_means(rows, "model")
    assert sorted(sm[("x", "d", False)]) == [0.8, 0.9]   # seed 0 deduped to the last (0.9)


def test_real_baseline_table_has_methods_and_reddit_leaks() -> None:
    """Smoke on the committed JSONL: the 7 methods appear and the reddit_body residual shows up."""
    md, tex = m.build_baseline_tables()
    for method in ("sgcn", "sigat", "dadsgnn", "sesgformer"):
        assert method in md and method in tex
    # the stable residual leak is on reddit_body (L > 0, flagged yes) for at least one method.
    assert "reddit_body" in md and "**yes**" in md


def test_cycle_table_leaks_only_at_full() -> None:
    md = m.build_cycle_table()
    # full is flagged leaking; strict/topo lines exist and are not.
    assert "| full |" in md and "| topo |" in md and "**yes**" in md
