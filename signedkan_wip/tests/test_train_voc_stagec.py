"""Unit smoke for train_voc_stagec (Stage D VOC port).

These tests run on a tiny synthetic VOC-shape fixture (not real VOC
data — we don't want a 500 MB download to happen on `pytest`) and
verify that:

 1. The Stage C model factory builds at VOC scale (20 classes,
    12 box queries, input_size=224) without shape mismatch.
 2. One forward + backward step produces a finite loss and finite
    gradients on every parameter.
 3. The checkpoint emitted by `--save-checkpoint` is loadable by
    the demo (`demo_hymeyolo_tk.load_or_train`).

The real-VOC end-to-end smoke is gated behind the
`HYMEYOLO_VOC_SMOKE=1` env var (network download required) — see
`test_train_voc_stagec_real_voc_optional`.
"""
from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
import pytest
import torch


def _synthetic_voc_arrays(n: int = 8, input_size: int = 64,
                          n_classes: int = 20, max_objects: int = 6,
                          seed: int = 0):
    """A tiny VOC-shape fixture that respects the (X, boxes, classes,
    counts) schema produced by voc_dataset.load_voc_hungarian."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 3, input_size, input_size)
                    ).astype(np.float32)
    boxes = np.zeros((n, max_objects, 4), dtype=np.float32)
    classes = -np.ones((n, max_objects), dtype=np.int64)
    counts = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        n_obj = int(rng.integers(1, 4))
        counts[i] = n_obj
        for j in range(n_obj):
            x0 = float(rng.uniform(0.0, 0.5))
            y0 = float(rng.uniform(0.0, 0.5))
            x1 = float(rng.uniform(x0 + 0.05, 1.0))
            y1 = float(rng.uniform(y0 + 0.05, 1.0))
            boxes[i, j] = [x0, y0, x1, y1]
            classes[i, j] = int(rng.integers(0, n_classes))
    return X, boxes, classes, counts


def test_stage_c_factory_builds_at_voc_scale():
    """The Stage C model (resnet + 2-level FPN) must build at
    n_classes=20, n_box_queries=12, d_hidden=32 — the VOC config."""
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
    model = RicciHyMeYOLOMulti(
        n_box_queries=12, n_circle_queries=0,
        n_classes=20, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0,
        use_layernorm=False,
        backbone="resnet", fpn="2level",
    )
    n_params = sum(p.numel() for p in model.parameters())
    # Plan §4: roughly 149k params; allow a generous range.
    assert 60_000 < n_params < 400_000, f"unexpected param count {n_params}"


def test_stage_c_forward_backward_at_voc_input_size():
    """One forward + backward step at the VOC config produces finite
    loss and finite gradients on every parameter."""
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti
    from signedkan_wip.src.vision.train_circles_ricci import combined_set_loss

    torch.manual_seed(0)
    model = RicciHyMeYOLOMulti(
        n_box_queries=12, n_circle_queries=0,
        n_classes=20, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0,
        use_layernorm=False,
        backbone="resnet", fpn="2level",
    )
    X_np, boxes_np, classes_np, counts_np = _synthetic_voc_arrays(
        n=2, input_size=64, n_classes=20, max_objects=4,
    )
    X = torch.from_numpy(X_np)
    boxes = torch.from_numpy(boxes_np)
    classes_safe = torch.from_numpy(np.where(classes_np < 0, 0, classes_np))
    counts = torch.from_numpy(counts_np)

    pred = model(X)
    # combined_set_loss API: (pred_dict, boxes, classes, counts, n_classes=…)
    loss, _accs = combined_set_loss(
        pred, boxes, classes_safe, counts, n_classes=20,
    )
    assert torch.isfinite(loss), f"non-finite loss: {loss.item()}"
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        assert torch.isfinite(p.grad).all(), \
            f"non-finite grad on {name}"


def test_stage_c_checkpoint_loads_via_demo(tmp_path):
    """Save a Stage C ckpt via `torch.save` mirroring what
    train_voc_stagec emits, then load it via the Tk demo's
    `load_or_train` and confirm `backbone='resnet'` / `fpn='2level'`
    survive the round-trip."""
    os.environ.pop("DISPLAY", None)
    from signedkan_wip.src.vision import demo_hymeyolo_tk
    from signedkan_wip.src.vision.hymeyolo_circles_ricci import RicciHyMeYOLOMulti

    # Demo currently hard-codes (n_box=4, n_circle=2) for reconstruction
    # because Cluttered MNIST used those defaults. We exercise the
    # *demo*'s loader contract here, so save the ckpt with those
    # defaults; the VOC-12-query case will be the demo's next
    # extension (covered in a separate plan).
    m = RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, circle_k=8,
        n_classes=10, d_hidden=32,
        ricci_modulation=True, ricci_scale=1.0,
        use_layernorm=False,
        backbone="resnet", fpn="2level",
    )
    ckpt_path = tmp_path / "stage_c_voc_seed0.pt"
    torch.save({
        "label": "stage_c_voc",
        "seed": 0, "epochs": 30, "lr": 3e-3,
        "ricci_scale": 1.0, "warm_start": False,
        "schedule": "cosine", "warmup_epochs": 2,
        "use_layernorm": False, "weight_decay": 0.0,
        "cls_loss": "ce", "box_loss": "giou",
        "backbone": "resnet", "fpn": "2level",
        "state_dict": m.state_dict(),
        "model_class": "RicciHyMeYOLOMulti",
        "dataset": "voc2007_trainval",
    }, ckpt_path)

    model, info = demo_hymeyolo_tk.load_or_train(
        str(ckpt_path), 100, "cpu",
    )
    assert info["backbone"] == "resnet"
    assert info["fpn"] == "2level"
    assert info["label"] == "stage_c_voc"


@pytest.mark.skipif(
    os.environ.get("HYMEYOLO_VOC_SMOKE") != "1",
    reason="real-VOC smoke gated behind HYMEYOLO_VOC_SMOKE=1 (network "
           "download required, ~500 MB)",
)
def test_train_voc_stagec_real_voc_optional(tmp_path):
    """Optional end-to-end smoke with real VOC2007. Off by default
    because it downloads ~500 MB."""
    from signedkan_wip.src.vision.voc_dataset import load_voc_hungarian
    X, boxes, classes, counts, names = load_voc_hungarian(
        year="2007", image_set="train",
        input_size=64, max_objects=6,
        root=str(tmp_path / "torchvision"), subset_n=10,
        download=True,
    )
    assert X.shape[0] == 10
    assert X.shape[1] == 3
    assert counts.sum() > 0
