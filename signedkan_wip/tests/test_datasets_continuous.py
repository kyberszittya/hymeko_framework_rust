"""Tests for the continuous-weight dataset loader.

Validates that Bitcoin Alpha / OTC continuous loaders preserve
the original rating magnitude (vs the binary loader which would
collapse to ±1), and that the WeightedSignedGraph dataclass
exposes both .weights (continuous) and .signs (derived binary).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_weighted_signed_graph_dataclass_signs_property():
    from signedkan_wip.src.datasets import WeightedSignedGraph
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    weights = np.array([+0.7, -0.3, +0.0001], dtype=np.float32)
    g = WeightedSignedGraph(edges=edges, weights=weights, n_nodes=3)
    # signs property is sign() of weights (binary view)
    assert tuple(g.signs.tolist()) == (1, -1, 1)
    # stats include continuous-weight statistics
    s = g.stats()
    assert s["n_edges"] == 3
    assert s["weight_min"] < 0
    assert s["weight_max"] > 0
    assert "weight_mean" in s
    assert "weight_std" in s


def test_continuous_bitcoin_alpha_preserves_magnitude(tmp_path, monkeypatch):
    """Test the loader against a synthetic Bitcoin-format CSV.
    Don't trigger an actual network download."""
    fake_csv = tmp_path / "bitcoin_alpha.csv"
    fake_csv.write_text(
        # src, dst, rating, timestamp
        "0,1,10,123\n"
        "1,2,-5,456\n"
        "2,3,7,789\n"
        "3,0,-10,999\n"
        # Rating 0 should be dropped:
        "0,2,0,888\n"
    )
    # Patch DATA_DIR so the loader looks in tmp_path.
    monkeypatch.setattr("signedkan_wip.src.datasets.DATA_DIR", tmp_path)
    monkeypatch.setattr("signedkan_wip.src.datasets.legacy.DATA_DIR", tmp_path)
    monkeypatch.setattr("signedkan_wip.src.datasets.DATA_DIR", tmp_path)
    monkeypatch.setattr("signedkan_wip.src.datasets.legacy.DATA_DIR", tmp_path)
    from signedkan_wip.src.datasets import load_continuous
    g = load_continuous("bitcoin_alpha")
    assert g.n_nodes == 4
    assert g.edges.shape == (4, 2)  # 5 rows minus the rating-0 line
    # Weights are r/10 in {+1, -0.5, +0.7, -1}
    s = sorted(g.weights.tolist())
    assert abs(s[0] - (-1.0)) < 1e-6
    assert abs(s[1] - (-0.5)) < 1e-6
    assert abs(s[2] - (+0.7)) < 1e-6
    assert abs(s[3] - (+1.0)) < 1e-6


def test_continuous_loader_drops_zero_ratings(tmp_path, monkeypatch):
    fake_csv = tmp_path / "bitcoin_alpha.csv"
    fake_csv.write_text(
        "0,1,5,111\n"
        "1,2,0,222\n"  # zero — should be dropped
        "2,3,-8,333\n"
    )
    monkeypatch.setattr("signedkan_wip.src.datasets.DATA_DIR", tmp_path)
    monkeypatch.setattr("signedkan_wip.src.datasets.legacy.DATA_DIR", tmp_path)
    monkeypatch.setattr("signedkan_wip.src.datasets.DATA_DIR", tmp_path)
    monkeypatch.setattr("signedkan_wip.src.datasets.legacy.DATA_DIR", tmp_path)
    from signedkan_wip.src.datasets import load_continuous
    g = load_continuous("bitcoin_alpha")
    assert g.edges.shape == (2, 2)


def test_continuous_loader_rejects_unsupported_dataset_format(monkeypatch):
    """If a future dataset has an unsupported format, loader raises
    cleanly (not a cryptic KeyError or empty result)."""
    # Force a registry-known dataset to use an unsupported format key.
    monkeypatch.setitem(
        __import__("signedkan_wip.src.datasets", fromlist=["FORMATS"]).FORMATS,
        "bitcoin_alpha", "fake_format_nonexistent",
    )
    from signedkan_wip.src.datasets import load_continuous
    with pytest.raises((NotImplementedError, KeyError)):
        load_continuous("bitcoin_alpha")
