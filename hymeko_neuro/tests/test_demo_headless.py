"""Tests for the HyMeYOLO headless slide-capture path.

Covers `hymeko_neuro.experiments.vision.demo_hymeyolo_core`:

 * unit    — `_peak_rss_mb`, `_measure_fwd_ms` (incl. the <5-rep
             failure case), the checkpoint-required / n_panels guards.
 * integ   — `render_headless` with a synthetic checkpoint writes the
             requested panels; with the committed trained checkpoint
             the mAP_50 lands in the published 0.90 band.
 * perf    — forward median < 250 ms CPU and peak RSS < 16 GB.
 * regress — the Tk GUI module still re-exports the functions moved to
             the core module (decompose on 2026-06-10).

Plan: docs/plans/2026-06-10-hymeyolo-headless-capture/.
"""
from __future__ import annotations

import pathlib

import pytest
import torch

pytest.importorskip("matplotlib")

from hymeko_neuro.experiments.vision import demo_hymeyolo_core as core

_COMMITTED_CKPT = pathlib.Path(
    "checkpoints/hymeyolo_demo/b_hsikan/ricci-mod_seed0.pt"
)


def _synthetic_ckpt(path: pathlib.Path, label: str = "+ricci-mod (synthetic)") -> pathlib.Path:
    """A loadable RicciHyMeYOLOMulti checkpoint with random weights.

    Used by structural tests that exercise the capture plumbing without
    needing trained-quality detections.
    """
    from hymeko_neuro.experiments.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
    torch.manual_seed(0)
    m = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, circle_k=8,
        n_classes=10, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0, use_layernorm=False,
    )
    torch.save({
        "label": label, "epochs": 0, "ricci_scale": 1.0,
        "warm_start": False, "schedule": "none",
        "backbone": "tiny", "fpn": "none",
        "state_dict": m.state_dict(), "model_class": "RicciHyMeYOLOMulti",
    }, path)
    return path


# ─── unit ─────────────────────────────────────────────────────────────


def test_peak_rss_mb_is_positive():
    rss = core._peak_rss_mb()
    # win32 + linux paths return a real number; only exotic platforms -1.
    assert rss > 0.0, f"expected positive peak RSS, got {rss}"
    assert rss < 16_000.0, "test harness itself already over the cap?"


def test_measure_fwd_ms_stats(tmp_path):
    ckpt = _synthetic_ckpt(tmp_path / "syn.pt")
    model, _ = core.load_or_train(str(ckpt), 0, "cpu")
    img = torch.randn(3, 64, 64)
    lat = core._measure_fwd_ms(model, img, "cpu", reps=7, warmup=2)
    assert lat["median_ms"] > 0.0
    assert lat["worst_ms"] >= lat["median_ms"]
    assert lat["iqr_ms"] >= 0.0


def test_measure_fwd_ms_rejects_too_few_reps(tmp_path):
    ckpt = _synthetic_ckpt(tmp_path / "syn.pt")
    model, _ = core.load_or_train(str(ckpt), 0, "cpu")
    img = torch.randn(3, 64, 64)
    with pytest.raises(AssertionError):
        core._measure_fwd_ms(model, img, "cpu", reps=3)


def test_render_headless_requires_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.render_headless(None, 2, 8, tmp_path, "cpu", 0)
    with pytest.raises(FileNotFoundError):
        core.render_headless(str(tmp_path / "nope.pt"), 2, 8, tmp_path, "cpu", 0)


def test_render_headless_rejects_zero_panels(tmp_path):
    ckpt = _synthetic_ckpt(tmp_path / "syn.pt")
    with pytest.raises(ValueError):
        core.render_headless(str(ckpt), 0, 8, tmp_path, "cpu", 0)


# ─── integration ──────────────────────────────────────────────────────


def test_render_headless_writes_panels_synthetic(tmp_path):
    ckpt = _synthetic_ckpt(tmp_path / "syn.pt")
    out = tmp_path / "out"
    res = core.render_headless(
        str(ckpt), n_panels=2, n_eval=8, out_dir=out, device="cpu", seed=0,
    )
    assert len(res.panel_paths) == 2
    for p in res.panel_paths:
        assert p.exists() and p.stat().st_size > 0, f"empty panel {p}"
    # n_eval is floored to >= n_panels.
    assert res.n_eval >= res.n_panels
    assert 0.0 <= res.mAP_50 <= 1.0
    assert res.peak_rss_mb < 16_000.0


@pytest.mark.skipif(
    not _COMMITTED_CKPT.is_file(),
    reason=f"committed checkpoint absent: {_COMMITTED_CKPT}",
)
def test_render_headless_real_checkpoint_map_band(tmp_path):
    out = tmp_path / "out"
    res = core.render_headless(
        str(_COMMITTED_CKPT), n_panels=3, n_eval=64,
        out_dir=out, device="cpu", seed=0,
    )
    assert len(res.panel_paths) == 3
    # Published corrected metric is 0.903 +/- 0.009; the demo split is
    # small (n_eval=64), so guard the band loosely at >= 0.80.
    assert res.mAP_50 >= 0.80, f"mAP_50={res.mAP_50:.3f} below demo band"
    assert 0.0 <= res.mAP_50_95 <= 1.0


# ─── performance ──────────────────────────────────────────────────────


@pytest.mark.skipif(
    not _COMMITTED_CKPT.is_file(),
    reason=f"committed checkpoint absent: {_COMMITTED_CKPT}",
)
def test_render_headless_perf_budget(tmp_path):
    out = tmp_path / "out"
    res = core.render_headless(
        str(_COMMITTED_CKPT), n_panels=1, n_eval=16,
        out_dir=out, device="cpu", seed=0,
    )
    # Budget from plan: median forward < 250 ms CPU; peak RSS < 16 GB.
    assert res.fwd_ms_median < 250.0, f"fwd median {res.fwd_ms_median:.1f} ms over budget"
    assert res.fwd_ms_worst >= res.fwd_ms_median
    assert 0.0 < res.peak_rss_mb < 16_000.0


# ─── regression: decompose did not break the GUI re-exports ───────────


def test_tk_module_reexports_core_functions():
    pytest.importorskip("tkinter", reason="GUI module imports tkinter")
    from hymeko_neuro.experiments.vision import demo_hymeyolo_tk as gui
    # The functions moved to core on 2026-06-10 must remain importable
    # via the GUI module (two existing tests call them through it).
    assert gui.load_or_train is core.load_or_train
    assert gui.predict is core.predict
    assert gui.render_axes is core.render_axes
