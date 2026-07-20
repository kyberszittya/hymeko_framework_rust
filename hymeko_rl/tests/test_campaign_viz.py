"""Tests for the campaign viz helpers (hymeko_rl/viz/campaign_viz.py) — the plotted output of §9."""
from __future__ import annotations

from pathlib import Path

import pytest

from hymeko_rl.viz.campaign_viz import plot_curve_overlay, plot_paired_deltas

_BASELINE = {"seeds": [{"curve": [{"step": 0.0, "delivery": 0.52}, {"step": 25000.0, "delivery": 0.0},
                                  {"step": 50000.0, "delivery": 0.02}]}]}
_FIX = {"seeds": [{"curve": [{"step": 0.0, "delivery": 0.52}, {"step": 25000.0, "delivery": 0.55},
                             {"step": 50000.0, "delivery": 0.6}]}]}


def test_plot_curve_overlay_writes_png(tmp_path: Path) -> None:
    out = plot_curve_overlay([("baseline (MSE)", _BASELINE), ("fix (Huber)", _FIX)],
                             tmp_path / "cmp.png", metric="delivery")
    assert out.exists() and out.stat().st_size > 0


def test_plot_curve_overlay_median_over_seeds(tmp_path: Path) -> None:
    two_seed = {"seeds": [{"curve": [{"step": 0.0, "delivery": 0.4}]},
                          {"curve": [{"step": 0.0, "delivery": 0.6}]}]}      # median @0 = 0.5
    out = plot_curve_overlay([("r", two_seed)], tmp_path / "m.png", seed_reduce="median")
    assert out.exists()


def test_plot_paired_deltas_writes_png(tmp_path: Path) -> None:
    agg = {"s2_cov": {"median": 0.0, "boot95": [0.0, 0.0]},
           "s1_clean": {"median": 0.2, "boot95": [0.05, 0.35]},          # clears zero (positive)
           "s1_ret": {"median": -1.0, "boot95": [-3.0, -0.4]}}           # clears zero (negative)
    out = plot_paired_deltas(agg, tmp_path / "deltas.png", order=["s2_cov", "s1_clean", "s1_ret"],
                             title="F12 − F11")
    assert out.exists() and out.stat().st_size > 0


def test_plot_paired_deltas_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one endpoint"):
        plot_paired_deltas({}, tmp_path / "x.png")


def test_plot_curve_overlay_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one run"):
        plot_curve_overlay([], tmp_path / "x.png")
