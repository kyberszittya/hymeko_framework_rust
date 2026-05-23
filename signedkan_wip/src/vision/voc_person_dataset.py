"""Single-class person-only loader for the Stage H rapport-vision detector.

Filters the VOC2007 trainval / test splits to images containing at
least one `person` GT, emits only the person bounding boxes, and
maps the class label to 0 (the only non-background class in the
single-class regime). Drop-in for ``load_voc_hungarian`` in
the training pipeline.

Plan: docs/plans/2026-05-19-stage-h-voc-eyes-for-rapport/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import VOCDetection
from torchvision.transforms.functional import to_tensor


# Single-class regime: index 0 = 'person'. The class-head emits
# n_classes + 1 logits — index 0 = person, index 1 = no-object.
PERSON_CLASSES: tuple[str, ...] = ("person",)
_PERSON_CLS_IDX = 0


def _parse_person_objects(ann: dict, ignore_difficult: bool
                           ) -> list[list[float]]:
    """Extract person bounding boxes (xmin, ymin, xmax, ymax) from a
    VOCDetection annotation dict. Non-person classes are silently
    dropped."""
    objs_field = ann["annotation"].get("object", [])
    if isinstance(objs_field, dict):
        objs_field = [objs_field]
    out: list[list[float]] = []
    for o in objs_field:
        if ignore_difficult and int(o.get("difficult", "0")) == 1:
            continue
        if o["name"] != "person":
            continue
        bb = o["bndbox"]
        out.append([
            float(bb["xmin"]), float(bb["ymin"]),
            float(bb["xmax"]), float(bb["ymax"]),
        ])
    return out


def load_voc_person_hungarian(
    *,
    year: str = "2007",
    image_set: str = "trainval",
    input_size: int = 224,
    max_objects: int = 6,
    root: str | Path = "data/torchvision",
    subset_n: int | None = None,
    ignore_difficult: bool = True,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Materialise VOC's person-only split into the Hungarian format.

    Images without any person GT are dropped (post-difficult-filter).
    Returns the same (X, boxes, classes, counts, class_names) tuple
    as ``load_voc_hungarian``, but ``classes`` only ever contains
    ``0`` (or ``-1`` for padding rows) and ``class_names`` is just
    ``("person",)``.

    The output ``n_total`` is the number of images that survived
    the person-filter, NOT ``len(VOCDetection)``.
    """
    Path(root).mkdir(parents=True, exist_ok=True)
    ds = VOCDetection(
        root=str(root), year=year, image_set=image_set, download=download,
    )

    # First pass: collect indices of images with at least one person GT.
    keep_idx: list[int] = []
    boxes_per_image: list[list[list[float]]] = []
    for i in range(len(ds)):
        _img_pil, ann = ds[i]
        bbs = _parse_person_objects(ann, ignore_difficult)
        if not bbs:
            continue
        keep_idx.append(i)
        boxes_per_image.append(bbs)
        if subset_n is not None and len(keep_idx) >= subset_n:
            break

    n_total = len(keep_idx)
    X = np.zeros((n_total, 3, input_size, input_size), dtype=np.float32)
    boxes = np.zeros((n_total, max_objects, 4), dtype=np.float32)
    classes = -np.ones((n_total, max_objects), dtype=np.int64)
    counts = np.zeros((n_total,), dtype=np.int64)

    for out_i, raw_idx in enumerate(keep_idx):
        img_pil, _ann = ds[raw_idx]
        w_orig, h_orig = img_pil.size
        img_resized = img_pil.resize(
            (input_size, input_size), resample=2,
        )
        x = to_tensor(img_resized)
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1)
        X[out_i] = x.numpy()
        kept = 0
        for bb in boxes_per_image[out_i]:
            if kept >= max_objects:
                break
            x0 = max(0.0, min(1.0, bb[0] / w_orig))
            y0 = max(0.0, min(1.0, bb[1] / h_orig))
            x1 = max(0.0, min(1.0, bb[2] / w_orig))
            y1 = max(0.0, min(1.0, bb[3] / h_orig))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes[out_i, kept] = [x0, y0, x1, y1]
            classes[out_i, kept] = _PERSON_CLS_IDX
            kept += 1
        counts[out_i] = kept

    return X, boxes, classes, counts, PERSON_CLASSES


def voc_person_class_names() -> tuple[str, ...]:
    return PERSON_CLASSES
