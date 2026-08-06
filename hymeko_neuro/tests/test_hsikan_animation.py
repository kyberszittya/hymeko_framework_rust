"""Tests for the HSiKAN explainer-animation generator.

Runs offline (illustrative αₖ, no checkpoint/torch) at tiny frame counts.
Needs matplotlib + Pillow (the `demo` group); skips otherwise.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = REPO_ROOT / "hymeko_neuro" / "demos" / "make_hsikan_animation.py"
    spec = importlib.util.spec_from_file_location("make_hsikan_animation", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


anim = _load_module()


def test_load_alpha_illustrative_fallback() -> None:
    alpha, labels, src = anim.load_alpha(None)
    assert len(alpha) == len(labels) > 0
    assert abs(float(alpha.sum()) - 1.0) < 0.05  # a regime (softmax-like)
    assert "illustrative" in src


def test_both_gifs_render_offline(tmp_path) -> None:
    rc = anim.main([
        "--mode", "both", "--checkpoint", "", "--out", str(tmp_path),
        "--frames-per-stage", "2", "--hold", "1",
    ])
    assert rc == 0
    fwd = tmp_path / "hsikan_forward.gif"
    neu = tmp_path / "hsikan_neurons.gif"
    assert fwd.is_file() and fwd.stat().st_size > 0
    assert neu.is_file() and neu.stat().st_size > 0
    # default cleans up intermediate frames — only the GIFs remain.
    assert list(tmp_path.glob("*.png")) == []


def test_keep_frames_flag(tmp_path) -> None:
    anim.main([
        "--mode", "neurons", "--out", str(tmp_path),
        "--frames-per-stage", "2", "--hold", "1", "--keep-frames",
    ])
    assert (tmp_path / "hsikan_neurons.gif").is_file()
    assert len(list(tmp_path.glob("neuron_frame_*.png"))) > 0
