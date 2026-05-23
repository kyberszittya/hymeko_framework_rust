"""Pascal VOC 2007 / 2012 loader for the HyMeYOLO Hungarian pipeline.

Converts torchvision's VOCDetection annotation dict into the
``(X, boxes, classes, counts)`` ndarray quadruple consumed by
``train_circles_ricci.train_one_config``.

Schema:
    X       : (N, 3, H, W)  float32, [0, 1]
    boxes   : (N, M_max, 4) float32, [x0, y0, x1, y1] in image coords (0..1)
    classes : (N, M_max)    int64,   class id (0..19); -1 = padding
    counts  : (N,)          int64,   number of real objects per image

VOC class ordering matches the standard 20-class list (alphabetical
in VOCDetection).  ``ignore_difficult=True`` mirrors COCO-style
evaluation: hard/difficult-flagged GT boxes are dropped.

Memory budget rough: 128 × 128 × 3 × 4 B × 5011 train images ≈ 940 MB.
Use ``subset_n`` to clip.

Usage:
    from signedkan_wip.src.vision.voc_dataset import load_voc_hungarian
    X, boxes, classes, counts, class_names = load_voc_hungarian(
        year='2007', image_set='train', input_size=128, max_objects=8,
        root='data/torchvision', subset_n=None,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms.functional import resize as tv_resize
from torchvision.transforms.functional import to_tensor

VOC_CLASSES: tuple[str, ...] = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
_CLS2IDX: dict[str, int] = {name: i for i, name in enumerate(VOC_CLASSES)}


def _parse_objects(
    ann: dict, ignore_difficult: bool,
) -> list[tuple[int, list[float]]]:
    """Extract (class_idx, [xmin, ymin, xmax, ymax]) pairs from
    a VOCDetection annotation dict.  Box coords are in pixel space
    of the *original* image; caller is responsible for normalising
    after the resize."""
    objs_field = ann["annotation"].get("object", [])
    if isinstance(objs_field, dict):
        objs_field = [objs_field]
    out: list[tuple[int, list[float]]] = []
    for o in objs_field:
        if ignore_difficult and int(o.get("difficult", "0")) == 1:
            continue
        name = o["name"]
        if name not in _CLS2IDX:
            continue
        bb = o["bndbox"]
        xmin = float(bb["xmin"])
        ymin = float(bb["ymin"])
        xmax = float(bb["xmax"])
        ymax = float(bb["ymax"])
        out.append((_CLS2IDX[name], [xmin, ymin, xmax, ymax]))
    return out


def load_voc_hungarian(
    *,
    year: str = "2007",
    image_set: str = "train",
    input_size: int = 128,
    max_objects: int = 8,
    root: str | Path = "data/torchvision",
    subset_n: int | None = None,
    ignore_difficult: bool = True,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Materialise the whole subset into memory as four ndarrays.

    Args:
        year: '2007' or '2012'.
        image_set: 'train', 'val', 'trainval', 'test'.
        input_size: target H = W (square resize, NOT keep-aspect).
        max_objects: pad/cap object slots per image.  Images with
            more objects keep only the first ``max_objects`` after
            VOC's annotation order (typically area-descending).
        root: torchvision data dir.
        subset_n: if not None, take only the first N images.
        ignore_difficult: drop ``difficult=1`` annotations.
        download: pass-through to VOCDetection.

    Returns:
        X, boxes, classes, counts, VOC_CLASSES
    """
    Path(root).mkdir(parents=True, exist_ok=True)
    ds = VOCDetection(
        root=str(root),
        year=year,
        image_set=image_set,
        download=download,
    )
    n_total = len(ds) if subset_n is None else min(subset_n, len(ds))

    X = np.zeros((n_total, 3, input_size, input_size), dtype=np.float32)
    boxes = np.zeros((n_total, max_objects, 4), dtype=np.float32)
    classes = -np.ones((n_total, max_objects), dtype=np.int64)
    counts = np.zeros((n_total,), dtype=np.int64)

    for i in range(n_total):
        img, ann = ds[i]
        w_orig, h_orig = img.size  # PIL: (W, H)
        # Resize to (input_size, input_size).
        img_resized = img.resize(
            (input_size, input_size), resample=2,  # PIL.Image.BILINEAR == 2
        )
        x = to_tensor(img_resized)  # (3, H, W) in [0, 1]
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        X[i] = x.numpy()
        objs = _parse_objects(ann, ignore_difficult)
        # Normalise box coords to [0, 1] then take first max_objects.
        kept = 0
        for cls_idx, bb in objs:
            if kept >= max_objects:
                break
            x0 = bb[0] / w_orig
            y0 = bb[1] / h_orig
            x1 = bb[2] / w_orig
            y1 = bb[3] / h_orig
            # Clamp to [0, 1] — VOC occasionally has off-by-one
            # annotations that extend slightly past the image.
            x0 = max(0.0, min(1.0, x0))
            y0 = max(0.0, min(1.0, y0))
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes[i, kept] = [x0, y0, x1, y1]
            classes[i, kept] = cls_idx
            kept += 1
        counts[i] = kept

    return X, boxes, classes, counts, VOC_CLASSES


def voc_class_names() -> tuple[str, ...]:
    """Convenience accessor — the canonical 20-class VOC list."""
    return VOC_CLASSES


def _smoke() -> None:
    """Tiny smoke that downloads a 10-image slice and prints shape."""
    X, boxes, classes, counts, names = load_voc_hungarian(
        year="2007", image_set="train", input_size=64, max_objects=4,
        subset_n=10, download=True,
    )
    print(f"X={X.shape}  boxes={boxes.shape}  counts.mean={counts.mean():.2f}")
    print(f"classes 0..3 of img0 = {classes[0]}, counts = {counts[0]}")
    print(f"VOC classes: {names}")


if __name__ == "__main__":
    _smoke()
