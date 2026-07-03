"""Cluttered MNIST: deterministic synthetic detection dataset.

Each image is a fixed-size canvas (default 64×64) with 1-3 MNIST
digits pasted at random non-overlapping locations.  Bounding boxes
are computed from the digit's bbox after pasting.  The detection
task: localise + classify each digit.

This is the Phase 0 / Phase 1 feasibility dataset for HyMeYOLO ---
small enough to train in ~60 min on a consumer GPU, structured
enough that an honest detection model can learn the task.

Usage:
    from hymeko_yolo.synthetic import ClutteredMNIST
    ds = ClutteredMNIST(n_samples=5000, canvas=64, max_digits=3,
                        seed=0, download=True)
    img, bboxes, labels = ds[0]   # img: (1, 64, 64), bboxes: (N, 4), labels: (N,)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


_DEFAULT_CACHE = Path.home() / ".cache" / "hymeko_yolo" / "mnist"


@dataclass(frozen=True)
class ClutteredSample:
    """One synthetic image + its annotations.

    image  : (1, canvas, canvas) float32 in [0, 1]
    bboxes : (N, 4) float32, [x_min, y_min, x_max, y_max] in pixel coords
    labels : (N,)   int64, MNIST digit class (0-9)
    """
    image: torch.Tensor
    bboxes: torch.Tensor
    labels: torch.Tensor


class ClutteredMNIST(Dataset):
    """Deterministic Cluttered MNIST detection dataset.

    Parameters
    ----------
    n_samples : int
        Number of synthetic images to generate.
    canvas : int
        Canvas side length in pixels (square).  Default 64.
    max_digits : int
        Max digits per image (sampled uniformly from {1, ..., max_digits}).
    min_digits : int
        Min digits per image.  Default 1.
    seed : int
        Master seed for reproducible generation.  Each sample is
        further seeded by its index, so ``ds[i]`` is deterministic.
    train : bool
        Use MNIST train split (60k) if True, else test (10k).
    cache_dir : Path | None
        Where to cache the MNIST source (downloads on first run).
    download : bool
        Allow torchvision to download MNIST if not cached.
    iou_tolerance : float
        Maximum allowed IoU between any two pasted digits.  Rejection
        sampling redraws positions until satisfied.  Default 0.10.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        canvas: int = 64,
        max_digits: int = 3,
        min_digits: int = 1,
        seed: int = 0,
        train: bool = True,
        cache_dir: Optional[Path] = None,
        download: bool = True,
        iou_tolerance: float = 0.10,
    ):
        assert 1 <= min_digits <= max_digits
        assert canvas >= 32, "canvas must be ≥ 32 to fit MNIST digits"
        self.n_samples = n_samples
        self.canvas = canvas
        self.max_digits = max_digits
        self.min_digits = min_digits
        self.seed = int(seed)
        self.iou_tolerance = float(iou_tolerance)

        # Lazy-load MNIST.
        from torchvision.datasets import MNIST
        cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        cache.mkdir(parents=True, exist_ok=True)
        ds = MNIST(root=str(cache), train=train, download=download)
        self._digits = ds.data.numpy()   # (N, 28, 28) uint8
        self._labels = ds.targets.numpy()  # (N,) int
        self._n_digits = len(self._digits)

    def __len__(self) -> int:
        return self.n_samples

    def _rng(self, idx: int) -> np.random.Generator:
        # Per-sample deterministic RNG: seeded by master_seed + idx.
        return np.random.default_rng(self.seed * 10_000 + idx)

    def _try_place(self, rng: np.random.Generator, digit_size: int = 28
                    ) -> Optional[tuple[int, int]]:
        """Sample a random (x, y) top-left corner where a digit_size×digit_size
        digit fits inside the canvas."""
        max_xy = self.canvas - digit_size
        if max_xy < 0:
            return None
        return (int(rng.integers(0, max_xy + 1)),
                int(rng.integers(0, max_xy + 1)))

    @staticmethod
    def _iou(box_a: tuple[int, int, int, int],
              box_b: tuple[int, int, int, int]) -> float:
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b
        ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
        iw = max(0, ix1 - ix0)
        ih = max(0, iy1 - iy0)
        inter = iw * ih
        a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = a + b - inter
        return inter / max(1, union)

    def __getitem__(self, idx: int) -> ClutteredSample:
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(idx)
        rng = self._rng(idx)
        n_d = int(rng.integers(self.min_digits, self.max_digits + 1))
        canvas = np.zeros((self.canvas, self.canvas), dtype=np.float32)
        bboxes: list[tuple[int, int, int, int]] = []
        labels: list[int] = []

        for _ in range(n_d):
            # Pick a random digit from MNIST.
            di = int(rng.integers(0, self._n_digits))
            digit = self._digits[di]  # (28, 28) uint8
            label = int(self._labels[di])

            # Try up to 32 placements to find a low-IoU spot.
            placed = False
            for _attempt in range(32):
                place = self._try_place(rng)
                if place is None:
                    break
                x, y = place
                cand = (x, y, x + 28, y + 28)
                if all(self._iou(cand, b) <= self.iou_tolerance
                        for b in bboxes):
                    bboxes.append(cand)
                    labels.append(label)
                    # Paste using max() so overlaps don't darken.
                    canvas[y:y + 28, x:x + 28] = np.maximum(
                        canvas[y:y + 28, x:x + 28],
                        digit.astype(np.float32) / 255.0,
                    )
                    placed = True
                    break
            # If we couldn't place after 32 tries, just skip this digit.
            if not placed:
                continue

        img = torch.from_numpy(canvas).unsqueeze(0)   # (1, H, W)
        if not bboxes:
            # Degenerate but legal: empty annotations.
            bbox_t = torch.zeros((0, 4), dtype=torch.float32)
            lbl_t = torch.zeros((0,), dtype=torch.int64)
        else:
            bbox_t = torch.tensor(bboxes, dtype=torch.float32)
            lbl_t = torch.tensor(labels, dtype=torch.int64)
        return ClutteredSample(image=img, bboxes=bbox_t, labels=lbl_t)


def collate_cluttered(batch: list[ClutteredSample]):
    """Default collate for ``DataLoader``: stacks images, keeps bboxes
    + labels as a list (variable length per sample)."""
    images = torch.stack([s.image for s in batch], dim=0)
    bboxes = [s.bboxes for s in batch]
    labels = [s.labels for s in batch]
    return images, bboxes, labels


def make_cluttered_mnist_hungarian_format(
    n: int,
    canvas: int = 32,
    max_objects: int = 3,
    seed: int = 0,
    rgb: bool = True,
):
    """Cluttered MNIST in the format consumed by
    ``hymeyolo_hungarian.py`` (DETR-style 4-corner queries).

    Drop-in replacement for ``make_synthetic_multi_rectangles`` --- so
    HyMeYOLOMulti can train on Cluttered MNIST without touching the
    model or loss.

    Returns
    -------
    X       : (n, 3 or 1, H, W) float32 in [0, 1]
    boxes   : (n, max_objects, 4) normalised (x0, y0, x1, y1) in [0, 1]²;
              padded with zeros for absent objects.
    classes : (n, max_objects) int64; 10 = "no-object" pad class.
    counts  : (n,) int64 — true object count per image.
    """
    n_classes = 10
    ds = ClutteredMNIST(
        n_samples=n, canvas=canvas, max_digits=max_objects,
        min_digits=1, seed=seed, download=True,
    )
    c = 3 if rgb else 1
    X = np.zeros((n, c, canvas, canvas), dtype=np.float32)
    boxes = np.zeros((n, max_objects, 4), dtype=np.float32)
    classes = np.full((n, max_objects), n_classes, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    for i in range(n):
        s = ds[i]
        if rgb:
            X[i] = s.image.numpy().repeat(3, axis=0)
        else:
            X[i] = s.image.numpy()
        n_obj = min(len(s.bboxes), max_objects)
        counts[i] = n_obj
        if n_obj > 0:
            norm = s.bboxes[:n_obj].numpy() / float(canvas)
            boxes[i, :n_obj] = norm
            classes[i, :n_obj] = s.labels[:n_obj].numpy()
    return X, boxes, classes, counts


if __name__ == "__main__":
    # Smoke test: generate and visualise 4 samples.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/cluttered_mnist_demo")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--canvas", type=int, default=64)
    parser.add_argument("--max-digits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    ds = ClutteredMNIST(
        n_samples=args.n, canvas=args.canvas,
        max_digits=args.max_digits, seed=args.seed,
    )
    print(f"generated dataset of {len(ds)} samples")
    for i in range(args.n):
        s = ds[i]
        print(f"  [{i}] image={tuple(s.image.shape)}  "
              f"n_bboxes={len(s.bboxes)}  labels={s.labels.tolist()}")
        # Write PNG via PIL if available.
        try:
            from PIL import Image, ImageDraw
            arr = (s.image[0].numpy() * 255).astype(np.uint8)
            im = Image.fromarray(arr, mode="L").convert("RGB")
            draw = ImageDraw.Draw(im)
            for (x0, y0, x1, y1), lbl in zip(s.bboxes.tolist(),
                                              s.labels.tolist()):
                draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0))
                draw.text((x0, max(0, y0 - 10)), str(lbl), fill=(255, 0, 0))
            im.save(f"{args.out}/sample_{i}.png")
        except ImportError:
            pass
    print(f"wrote samples to {args.out}/")
