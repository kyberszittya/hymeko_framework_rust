"""Single-digit Cluttered-MNIST classification adapter.

Wraps ``ClutteredMNIST(max_digits=1)`` into a ``(image, label)`` classification
Dataset: one MNIST digit pasted at a *random position* on a ``canvas``×``canvas``
field. Because the digit's position is random, an absolute-position readout
(flatten) cannot use position as the class signal — the task that discriminates
a content-attention pool from raw flatten (plan
``docs/plans/2026-06-29-soma-position-aware-readout-program/``).

Reuses ``ClutteredMNIST`` for all generation (no duplicate synthesis — §6.1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from hymeko_neuro.experiments.vision.cluttered_mnist import ClutteredMNIST


class ClutteredMNISTClassification(Dataset):
    """One randomly-placed MNIST digit per ``canvas``×``canvas`` image; classify it.

    Preconditions
    -------------
    * ``canvas >= 28`` — a 28×28 digit must fit (then ``ClutteredMNIST`` always
      places it: ``max_xy = canvas - 28 >= 0`` and there is no other box to
      collide with, so the single-digit annotation is guaranteed non-empty).

    Postconditions
    --------------
    * ``__getitem__(i)`` → ``(image (1, canvas, canvas) float32 in [0,1],
      label int in [0, 9])``, deterministic in ``i``.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        canvas: int = 48,
        seed: int = 0,
        train: bool = True,
        cache_dir: Optional[Path] = None,
        download: bool = True,
        cache: bool = True,
    ) -> None:
        if canvas < 28:
            raise ValueError(f"canvas must be >= 28 to fit a digit; got {canvas}")
        self.canvas = canvas
        self._ds = ClutteredMNIST(
            n_samples=n_samples, canvas=canvas, max_digits=1, min_digits=1,
            seed=seed, train=train, cache_dir=cache_dir, download=download,
        )
        # The samples are deterministic in their index, so precompute them once
        # into tensors. Per-item synthesis (numpy paste + rejection sampling) is
        # otherwise re-run every epoch and starves a batched GPU forward; caching
        # makes data loading a cheap tensor index. ~46 MB for 5000×48×48.
        self._imgs: torch.Tensor | None = None
        self._labels: list[int] | None = None
        if cache:
            samples = [self._sample(i) for i in range(n_samples)]
            self._imgs = torch.stack([img for img, _ in samples])
            self._labels = [lab for _, lab in samples]

    def _sample(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self._ds[idx]
        if s.labels.numel() == 0:
            # Unreachable under the canvas>=28 precondition (single digit always
            # places); guard rather than silently return a wrong label.
            raise RuntimeError(
                f"no digit placed for idx {idx}; canvas={self.canvas} too small?"
            )
        return s.image, int(s.labels[0])

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self._imgs is not None and self._labels is not None:
            return self._imgs[idx], self._labels[idx]      # cached tensor index
        return self._sample(idx)
