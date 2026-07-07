"""Tests for the campaign viz helpers (hymeko_rl/viz/campaign_viz.py) — the plotted output of §9."""
from __future__ import annotations

from pathlib import Path

import pytest

from hymeko_rl.viz.campaign_viz import plot_curve_overlay

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


def test_plot_curve_overlay_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one run"):
        plot_curve_overlay([], tmp_path / "x.png")
