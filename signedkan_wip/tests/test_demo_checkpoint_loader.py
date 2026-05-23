"""Regression tests for demo_hymeyolo_tk's checkpoint loader.

The demo learned to read Stage B (`backbone`) and Stage C (`fpn`)
keys from the checkpoint dict on 2026-05-17. Before that change the
loader hard-coded `backbone='tiny'`/`fpn='none'` and would silently
construct the wrong architecture when a Stage B/C `.pt` was passed.
These tests guard against regression in three directions:

 * Stage B (`backbone='resnet'`) ckpt round-trips through the loader.
 * Stage C (`backbone='resnet'`, `fpn='2level'`) ckpt round-trips.
 * Stage A back-compat: a ckpt without `backbone`/`fpn` keys defaults
   to `('tiny', 'none')` and still loads.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import pytest
import torch


pytest.importorskip("matplotlib")
pytest.importorskip("tkinter", reason="demo module touches tkinter at import")


def _build_ckpt(backbone: str, fpn: str, label: str) -> dict:
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
    m = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, circle_k=8,
        n_classes=10, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0,
        use_layernorm=False,
        backbone=backbone, fpn=fpn,
    )
    return {
        "label": label,
        "seed": 0,
        "epochs": 100,
        "lr": 3e-3,
        "ricci_scale": 1.0,
        "warm_start": True,
        "schedule": "cosine",
        "warmup_epochs": 10,
        "use_layernorm": False,
        "weight_decay": 0.0,
        "cls_loss": "ce",
        "box_loss": "giou",
        "backbone": backbone,
        "fpn": fpn,
        "state_dict": m.state_dict(),
        "model_class": "RicciHyMeYOLOMulti",
    }


def test_stage_b_resnet_checkpoint_roundtrip(tmp_path):
    os.environ.pop("DISPLAY", None)
    from signedkan_wip.src.vision import demo_hymeyolo_tk
    ckpt_path = tmp_path / "b_resnet_seed0.pt"
    torch.save(_build_ckpt("resnet", "none", "b_resnet+ricci-mod"), ckpt_path)
    model, info = demo_hymeyolo_tk.load_or_train(str(ckpt_path), 100, "cpu")
    assert info["backbone"] == "resnet"
    assert info["fpn"] == "none"
    assert info["label"] == "b_resnet+ricci-mod"
    # Forward pass smoke
    img = torch.randn(3, 64, 64)
    pred = demo_hymeyolo_tk.predict(model, img, device="cpu")
    assert "box" in pred


def test_stage_c_fpn_checkpoint_roundtrip(tmp_path):
    os.environ.pop("DISPLAY", None)
    from signedkan_wip.src.vision import demo_hymeyolo_tk
    ckpt_path = tmp_path / "c_fpn_seed0.pt"
    torch.save(_build_ckpt("resnet", "2level", "c_fpn+ricci-mod"), ckpt_path)
    model, info = demo_hymeyolo_tk.load_or_train(str(ckpt_path), 100, "cpu")
    assert info["backbone"] == "resnet"
    assert info["fpn"] == "2level"
    img = torch.randn(3, 64, 64)
    pred = demo_hymeyolo_tk.predict(model, img, device="cpu")
    assert "box" in pred


def test_stage_a_legacy_checkpoint_defaults(tmp_path):
    """Old ckpts saved before 2026-05-17 lacked backbone/fpn keys.
    They must still load by defaulting to tiny/none."""
    os.environ.pop("DISPLAY", None)
    from signedkan_wip.src.vision import demo_hymeyolo_tk
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
    m = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, circle_k=8,
        n_classes=10, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0,
        use_layernorm=False,
        backbone="tiny", fpn="none",
    )
    ckpt_path = tmp_path / "old.pt"
    torch.save({
        "label": "baseline+ricci-mod",
        "epochs": 100,
        "state_dict": m.state_dict(),
        "model_class": "RicciHyMeYOLOMulti",
    }, ckpt_path)
    model, info = demo_hymeyolo_tk.load_or_train(str(ckpt_path), 100, "cpu")
    assert info["backbone"] == "tiny"
    assert info["fpn"] == "none"
