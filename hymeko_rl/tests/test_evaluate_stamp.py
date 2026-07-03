"""Tests for the artifact-provenance stamping (timestamps on GIFs + result tables).

All pure-PIL/numpy — no GL context needed (frames are synthesized, not rendered).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

import csv as _csv

from hymeko_rl.eval.evaluate import (
    EvalStats, _stamp_frames, compare_gif, experiment_dir, now_stamp, plot_scoreboard, results_to_csv,
)


def _frames(n: int = 3, h: int = 40, w: int = 60) -> list[np.ndarray]:
    # distinct frames (top-left marker, outside the bottom-right stamp) so the GIF keeps every frame
    out = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:4, :4] = (i + 1) * 40
        out.append(f)
    return out


def test_experiment_dir_is_timestamped_and_created(tmp_path: Path) -> None:
    """A run dir ``<base>/YYYY_MM_DD_HH_MM_<name>`` is created; the leading stamp sorts runs chronologically."""
    out = experiment_dir(tmp_path, "pernode_galambos")
    assert out.exists() and out.is_dir()
    assert out.name.endswith("_pernode_galambos")
    assert re.fullmatch(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_pernode_galambos", out.name)


def test_results_to_csv_one_row_per_config(tmp_path: Path) -> None:
    """The results table writes one CSV row per config with the union of metric columns."""
    results = {"hsikan_pernode": {"delivery": 0.4, "mean_return": -12.3},
               "mlp": {"delivery": 0.25, "mean_return": -20.1}}
    out = results_to_csv(tmp_path / "results", results)
    assert out.suffix == ".csv"
    rows = list(_csv.DictReader(out.open(encoding="utf-8")))
    assert [r["config"] for r in rows] == ["hsikan_pernode", "mlp"]
    assert rows[0]["delivery"] == "0.4" and rows[1]["mean_return"] == "-20.1"


def test_plot_scoreboard_handles_three_sources(tmp_path: Path) -> None:
    """A 3-way scoreboard plots without an alpha>1 crash (regression: the per-series alpha was 0.6+0.4·i,
    which exceeds matplotlib's [0,1] range at the third source — pernode-actor A/B has three configs)."""
    stats = [EvalStats(s, g, 0, 10 - g, tuple(float(k) for k in range(10)))
             for s, g in (("pooled", 1), ("pernode", 3), ("mlp", 0))]
    out = plot_scoreboard(stats, tmp_path / "sb", title="three")
    assert out.exists()


def test_now_stamp_format() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", now_stamp())


def test_stamp_frames_preserves_shape_and_marks_corner() -> None:
    frames = [np.zeros((40, 60, 3), dtype=np.uint8)]
    out = _stamp_frames(frames, "2026-06-22 13:00:00")
    assert len(out) == 1
    assert out[0].shape == frames[0].shape and out[0].dtype == np.uint8
    # the bottom-right region must now carry ink (the label box), the top-left must not.
    assert out[0][-8:, -8:].sum() > 0
    assert out[0][:8, :8].sum() == 0


def test_stamp_frames_empty_label_is_noop() -> None:
    frames = _frames()
    assert _stamp_frames(frames, "") is frames


def test_compare_gif_writes_stamped_gif(tmp_path: Path) -> None:
    out = compare_gif([_frames(), _frames()], tmp_path / "cmp", fps=10, stamp="2026-06-22 13:00:00")
    assert out.exists() and out.suffix == ".gif"
    im = Image.open(out)
    assert im.format == "GIF" and getattr(im, "n_frames", 1) >= 2
    assert im.size == (60 * 2 + 4, 40)               # two 60-wide panels + a 4-px separator
