"""Tests for the seminar ``balance`` demo (Demo 1 / plan-item 5).

The pure helpers (``cycle_sign_products``, ``build_affective_graph``) and the
figure need only numpy/networkx; ``balance_statistics`` and the end-to-end run
need the built ``hymeko`` engine and skip without it. Run via the venv::

    PYTHONPATH=. .venv/Scripts/python.exe -m pytest -p no:randomly \
        hymeko_neuro/tests/test_seminar_balance.py
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_neuro.demos.seminar.base import DemoRunner
from hymeko_neuro.demos.seminar.cli import build_parser
from hymeko_neuro.demos.seminar.demos import DEMOS
from hymeko_neuro.demos.seminar.demos.balance import (
    build_affective_graph, cycle_sign_products,
)

try:
    import hymeko  # noqa: F401

    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False

needs_engine = pytest.mark.skipif(not _HAS_ENGINE, reason="needs built hymeko")


# ─── Unit: cycle_sign_products (pure) ───────────────────────────────────


def test_sign_product_balanced_triangle_is_positive() -> None:
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    signs = np.array([1, 1, 1], dtype=np.int8)
    out = cycle_sign_products(edges, signs, np.array([[0, 1, 2]]))
    assert out.tolist() == [1]


def test_sign_product_one_negative_is_frustrated() -> None:
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    signs = np.array([-1, 1, 1], dtype=np.int8)
    out = cycle_sign_products(edges, signs, np.array([[0, 1, 2]]))
    assert out.tolist() == [-1]


def test_sign_product_two_negatives_is_balanced() -> None:
    # A high fraction_negative cycle that is nonetheless BALANCED — proves
    # balance must come from the product, not the score.
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    signs = np.array([-1, -1, 1], dtype=np.int8)
    out = cycle_sign_products(edges, signs, np.array([[0, 1, 2]]))
    assert out.tolist() == [1]


def test_sign_product_rejects_non_edge() -> None:
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    signs = np.array([1, 1, 1], dtype=np.int8)
    with pytest.raises(KeyError):
        cycle_sign_products(edges, signs, np.array([[0, 1, 3]]))


# ─── Unit: fixtures ─────────────────────────────────────────────────────


def test_camps_fixture_is_balanced_by_construction() -> None:
    g, planted = build_affective_graph("camps")
    assert planted is None and g.n_nodes == 6
    # Every triangle present is all-positive within a camp.
    products = cycle_sign_products(
        np.asarray(g.edges), np.asarray(g.signs),
        np.array([[0, 1, 2], [3, 4, 5]]))
    assert products.tolist() == [1, 1]


def test_planted_fixture_flips_one_edge_and_names_the_triad() -> None:
    g, planted = build_affective_graph("planted")
    assert planted == frozenset({0, 1, 2})
    products = cycle_sign_products(
        np.asarray(g.edges), np.asarray(g.signs),
        np.array([[0, 1, 2], [3, 4, 5]]))
    assert products.tolist() == [-1, 1]  # planted triad frustrated, other intact


# ─── Figure (no engine) ─────────────────────────────────────────────────


def test_frustration_figure_renders() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from hymeko_neuro.experiments.demo.plotting import frustration_figure
    g, _ = build_affective_graph("planted")
    fig = frustration_figure(np.asarray(g.edges), np.asarray(g.signs),
                             [[0, 1, 2]], 6, balance=0.5)
    assert fig is not None
    plt.close(fig)


# ─── Engine-backed: balance_statistics + integration ────────────────────


def test_balance_is_registered() -> None:
    assert "balance" in DEMOS and DEMOS["balance"].name == "balance"


@needs_engine
def test_balance_statistics_camps_is_one() -> None:
    from hymeko_neuro.demos.seminar.demos.balance import balance_statistics
    g, _ = build_affective_graph("camps")
    stats = balance_statistics(g, k=3, k_keep=100_000)
    assert stats.balance == 1.0
    assert stats.n_frustrated == 0 and stats.frustrated == []
    assert not stats.cap_hit


@needs_engine
def test_balance_statistics_planted_surfaces_triad() -> None:
    from hymeko_neuro.demos.seminar.demos.balance import balance_statistics
    g, planted = build_affective_graph("planted")
    stats = balance_statistics(g, k=3, k_keep=100_000)
    assert stats.balance == pytest.approx(0.5)
    assert stats.n_frustrated == 1
    assert any(frozenset(c) == planted for c in stats.frustrated)


def _run(tmp_path, extra: list[str]):
    args = build_parser().parse_args(["balance", "--device", "cpu", *extra])
    return DemoRunner(DEMOS["balance"], out_root=tmp_path).run(args)


@needs_engine
def test_balance_run_planted_writes_figure_and_metrics(tmp_path) -> None:
    result = _run(tmp_path, ["--graph", "planted"])
    assert result.metrics["balance"] == 0.5
    assert result.metrics["planted_cycle_surfaced"] is True
    assert result.metrics["frustration"] == 0.5
    assert result.artifacts and result.artifacts[0].is_file()
    assert result.provenance["balance"].startswith("planted fixture")


@needs_engine
def test_balance_run_no_figures(tmp_path) -> None:
    result = _run(tmp_path, ["--graph", "camps", "--no-figures"])
    assert result.metrics["balance"] == 1.0
    assert result.artifacts == []


@needs_engine
def test_balance_rejects_k_below_three(tmp_path) -> None:
    with pytest.raises(ValueError):
        _run(tmp_path, ["--graph", "camps", "--k", "2"])
