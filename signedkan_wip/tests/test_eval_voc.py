"""Tests for held-out VOC evaluation + panel capture (`eval_voc`).

 * unit    — checkpoint-required / n_panels guards; `_draw_voc_panel`
             renders a figure from a Detection list without error.
 * integ   — synthetic VOC checkpoint round-trips through
             `eval_checkpoint_on_split` and `render_voc_panels` on a
             small test-split slice (skips if VOC2007 not on disk).

The synthetic checkpoint uses `backbone='resnet'` (from-scratch tiny)
so the test never downloads ImageNet weights; it carries the nodelet
head so the `voc_detector` reconstruction path (gate decode) is
exercised.

Plan: docs/plans/2026-06-10-voc-test-baseline/.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib")

from signedkan_wip.src.vision import eval_voc

_VOC_ROOT = pathlib.Path("data/torchvision/VOCdevkit/VOC2007")


def _voc_available() -> bool:
    return _VOC_ROOT.is_dir()


def _synthetic_voc_ckpt(path: pathlib.Path, input_size: int = 64) -> pathlib.Path:
    """A loadable nodelet-head VOC checkpoint with random weights."""
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
    torch.manual_seed(0)
    model = RicciHyMeYOLOMulti(
        n_box_queries=6, n_circle_queries=0, n_classes=20, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0, use_layernorm=False,
        backbone="resnet", fpn="2level", query_head_kind="nodelet",
    )
    torch.save({
        "label": "stage_c_voc", "seed": 0, "epochs": 0,
        "ricci_scale": 1.0, "use_layernorm": False,
        "backbone": "resnet", "fpn": "2level",
        "n_box_queries": 6, "n_classes": 20, "input_size": input_size,
        "query_head_kind": "nodelet",
        "state_dict": model.state_dict(),
        "model_class": "RicciHyMeYOLOMulti",
        "dataset": "voc2007_trainval",
    }, path)
    return path


# ─── unit ─────────────────────────────────────────────────────────────


def test_eval_requires_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        eval_voc.eval_checkpoint_on_split(
            str(tmp_path / "nope.pt"), image_set="test", device="cpu",
        )


def test_render_rejects_zero_panels(tmp_path):
    ckpt = _synthetic_voc_ckpt(tmp_path / "syn.pt")
    with pytest.raises(ValueError):
        eval_voc.render_voc_panels(
            str(ckpt), n_panels=0, out_dir=tmp_path, device="cpu",
        )


def test_draw_voc_panel_renders(tmp_path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from signedkan_wip.src.rapport_ros2.voc_detector import Detection

    img = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)
    gt_boxes = np.array([[0.1, 0.1, 0.4, 0.4]], dtype=np.float32)
    gt_classes = np.array([14], dtype=np.int64)  # "person"
    dets = [Detection(x0=5, y0=5, x1=30, y1=30, score=0.8, agent_kind="person")]
    fig = Figure(figsize=(9, 4.5))
    FigureCanvasAgg(fig)
    eval_voc._draw_voc_panel(
        fig, img, gt_boxes, gt_classes, 1, dets,
        eval_voc.VOC_CLASSES, canvas_px=64,
    )
    out = tmp_path / "p.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0


# ─── integration (needs VOC2007 on disk) ──────────────────────────────


@pytest.mark.skipif(not _voc_available(), reason="VOC2007 not downloaded")
def test_eval_checkpoint_on_split_small(tmp_path):
    ckpt = _synthetic_voc_ckpt(tmp_path / "syn.pt", input_size=64)
    m = eval_voc.eval_checkpoint_on_split(
        str(ckpt), image_set="test", input_size=64, n_images=8,
        batch_size=4, device="cpu",
    )
    assert 0.0 <= m["mAP_50"] <= 1.0
    assert 0.0 <= m["mAP_50_95"] <= 1.0
    assert m["n_images"] == 8
    assert m["peak_rss_mb"] < 16_000.0


@pytest.mark.skipif(
    not (_voc_available() and torch.cuda.is_available()),
    reason="needs VOC2007 + cuda (guards the GT-on-device regression)",
)
def test_eval_checkpoint_on_split_cuda(tmp_path):
    # Regression: GT tensors must follow the model to cuda, else the IoU
    # step mixes cuda/cpu (fixed 2026-06-10). The cpu test above cannot
    # catch this because everything is already on one device.
    ckpt = _synthetic_voc_ckpt(tmp_path / "syn.pt", input_size=64)
    m = eval_voc.eval_checkpoint_on_split(
        str(ckpt), image_set="test", input_size=64, n_images=8,
        batch_size=4, device="cuda",
    )
    assert 0.0 <= m["mAP_50"] <= 1.0


@pytest.mark.skipif(not _voc_available(), reason="VOC2007 not downloaded")
def test_render_voc_panels_small(tmp_path):
    ckpt = _synthetic_voc_ckpt(tmp_path / "syn.pt", input_size=64)
    out = tmp_path / "voc_out"
    paths = eval_voc.render_voc_panels(
        str(ckpt), n_panels=2, out_dir=out, image_set="test",
        input_size=64, device="cpu",
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
