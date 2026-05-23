"""Synthetic Cichy-92-like data generator for the GömbSoma cortical
benchmark.

The real Cichy 92 dataset (Cichy, Pantazis & Oliva 2014) is 92
greyscale stimuli × 16 subjects × MEG/fMRI ROI signals over V1/V2/V4.
The 2026-05-16 cortical-benchmark plan named it as the right
``first viable'' target. This module provides a synthetic
generator with the same data SHAPE so the full scoring pipeline
can be validated end-to-end without network dependency.

The synthetic stimuli have a four-category structure (faces,
objects, places, scrambled) and per-subject ROI signals are
generated as category-tuned linear filters of pixel statistics
plus calibrated Gaussian noise — enough structure that a real
feature extractor's $r^2$ is meaningfully non-zero, but enough
noise that the noise-ceiling correction is non-trivial.

Object-oriented commitment: :class:`SyntheticCorticalDataset` is
a frozen dataclass; the generator :func:`make_synthetic_cichy_like`
is a pure function of its arguments and the random seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch

# Default ROI voxel counts mirror the Cichy 92 paper's ROI sizes.
_DEFAULT_VOXELS: dict[str, int] = {"V1": 100, "V2": 80, "V4": 60}

# Four canonical category groups — Cichy 92 has 12 fine-grained
# super-categories, but the four-class structure is sufficient for
# the synthetic benchmark's stimulus-to-signal coupling.
_CATEGORIES: tuple[str, ...] = ("faces", "objects", "places", "scrambled")


@dataclass(frozen=True)
class SyntheticCorticalDataset:
    """A Cichy-92-shaped synthetic benchmark.

    Attributes
    ----------
    images : Tensor
        ``[n_images, C, H, W]`` greyscale or RGB stimuli.
    roi_signals : dict[str, Tensor]
        For each ROI name, a tensor of shape
        ``[n_subjects, n_images, n_voxels]`` of fMRI-like
        category-tuned responses with iid Gaussian noise.
    image_classes : Tensor
        ``[n_images]`` integer category labels in
        ``range(len(category_names))``.
    category_names : tuple[str, ...]
        Names of the categories, in label-order.
    snr : float
        The SNR used to generate the synthetic ROI signals
        (signal_var / (signal_var + noise_var)).
    """

    images: torch.Tensor
    roi_signals: dict[str, torch.Tensor]
    image_classes: torch.Tensor
    category_names: tuple[str, ...]
    snr: float = field(default=0.3)

    @property
    def n_images(self) -> int:
        return int(self.images.shape[0])

    @property
    def n_subjects(self) -> int:
        first_roi = next(iter(self.roi_signals.values()))
        return int(first_roi.shape[0])

    @property
    def roi_names(self) -> tuple[str, ...]:
        return tuple(self.roi_signals.keys())

    def __post_init__(self) -> None:
        # Postconditions (light validation; deeper checks happen in
        # callers that consume the dataset).
        if self.images.ndim != 4:
            raise ValueError(
                f"images must be 4D [N,C,H,W]; got {tuple(self.images.shape)}"
            )
        n = self.n_images
        if self.image_classes.shape != (n,):
            raise ValueError(
                f"image_classes must be shape ({n},); got {tuple(self.image_classes.shape)}"
            )
        for roi, sig in self.roi_signals.items():
            if sig.ndim != 3:
                raise ValueError(
                    f"roi_signals[{roi!r}] must be 3D [S,N,V]; got {tuple(sig.shape)}"
                )
            if sig.shape[1] != n:
                raise ValueError(
                    f"roi_signals[{roi!r}].shape[1] must be n_images={n}; "
                    f"got {sig.shape[1]}"
                )


def _make_category_filter(
    rng: np.random.Generator,
    category: int,
    image_h: int,
    image_w: int,
    n_voxels: int,
) -> np.ndarray:
    """Per-voxel linear filter from image pixels.

    Each voxel has a Gaussian receptive-field-like filter on a
    random spatial centre, with category-dependent orientation bias.
    The result is a ``(n_voxels, image_h * image_w)`` matrix.
    """
    grid_y, grid_x = np.meshgrid(
        np.arange(image_h), np.arange(image_w), indexing="ij"
    )
    centres_y = rng.uniform(0, image_h, size=n_voxels)
    centres_x = rng.uniform(0, image_w, size=n_voxels)
    sigma = max(2.0, min(image_h, image_w) / 8.0)
    filters = np.zeros((n_voxels, image_h * image_w), dtype=np.float32)
    # Category-specific orientation factor in [-1, 1] biases the
    # filter's directional sensitivity.
    cat_theta = category * (np.pi / len(_CATEGORIES))
    for v in range(n_voxels):
        d2 = (grid_y - centres_y[v]) ** 2 + (grid_x - centres_x[v]) ** 2
        gauss = np.exp(-d2 / (2 * sigma * sigma))
        # Inject orientation tuning via a sinusoidal modulation.
        orient = np.cos(2 * cat_theta) * grid_y / image_h \
            + np.sin(2 * cat_theta) * grid_x / image_w
        filters[v] = (gauss * (0.5 + 0.5 * orient)).flatten()
    # Normalise so each voxel filter has unit L2 (calibration anchor).
    norms = np.linalg.norm(filters, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return filters / norms


def make_synthetic_cichy_like(
    n_images: int = 92,
    n_subjects: int = 16,
    image_size: int = 64,
    in_channels: int = 1,
    n_voxels: Mapping[str, int] | None = None,
    snr: float = 0.3,
    seed: int = 0,
) -> SyntheticCorticalDataset:
    """Generate a Cichy-92-shaped synthetic cortical benchmark.

    Parameters
    ----------
    n_images
        Number of stimuli (default 92, matching the real dataset).
    n_subjects
        Number of subjects (default 16, matching the real dataset).
    image_size
        Side length of the square stimuli (default 64; the real
        dataset is 175×175, but 64 is fast for synthetic smoke).
    in_channels
        1 for greyscale, 3 for RGB.
    n_voxels
        Per-ROI voxel counts. Defaults to ``{"V1":100, "V2":80, "V4":60}``.
    snr
        Signal-to-noise ratio for the ROI-signal generator.
        ``signal_var / (signal_var + noise_var)``; lower = more
        noise.
    seed
        RNG seed for full determinism.

    Returns
    -------
    SyntheticCorticalDataset
        Shape-faithful with the real dataset; safe drop-in for the
        scoring pipeline.

    Notes
    -----
    * The images are pure synthetic random gradients with
      category-dependent low-frequency structure — not realistic
      photographs. The intent is to give the feature extractor
      something to fit, not to mimic real visual stimuli.
    * The ROI signals have a category-tuned linear filter
      structure with per-voxel Gaussian receptive fields, plus
      iid Gaussian noise calibrated to the requested SNR.
    """
    if n_voxels is None:
        n_voxels = _DEFAULT_VOXELS

    rng = np.random.default_rng(seed)

    # ─── 1. Image generation: per-category low-freq structure. ──
    # Each image gets a category label; the image is a smooth
    # Gaussian random field with a category-specific orientation
    # mean. This gives a feature extractor something predictable
    # to latch on to.
    n_categories = len(_CATEGORIES)
    classes = np.repeat(np.arange(n_categories), n_images // n_categories + 1)
    classes = classes[:n_images]
    rng.shuffle(classes)

    images = np.zeros(
        (n_images, in_channels, image_size, image_size), dtype=np.float32
    )
    for i in range(n_images):
        cat = int(classes[i])
        theta = cat * (np.pi / n_categories)
        # Low-frequency directional field + iid pixel noise.
        ys = np.linspace(-1, 1, image_size)
        xs = np.linspace(-1, 1, image_size)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        base = np.cos(2 * np.pi * (np.cos(theta) * gx + np.sin(theta) * gy))
        per_image_jitter = rng.standard_normal((image_size, image_size)) * 0.2
        img = base + per_image_jitter
        # Normalise to [0, 1] for stability across categories.
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        for c in range(in_channels):
            images[i, c] = img + 0.05 * c

    # ─── 2. ROI signals: category-tuned linear filters + noise. ──
    roi_signals: dict[str, torch.Tensor] = {}
    for roi, n_v in n_voxels.items():
        # Per-ROI filter banks: one filter per voxel, biased by the
        # category of the image being viewed. We construct the
        # filter banks once per (ROI, category) and apply them per
        # image.
        cat_filters = [
            _make_category_filter(rng, c, image_size, image_size, n_v)
            for c in range(n_categories)
        ]  # list of (n_v, image_h*image_w) arrays
        flat_imgs = images.mean(axis=1).reshape(n_images, -1)  # (N, H*W)
        signal = np.zeros((n_subjects, n_images, n_v), dtype=np.float32)
        for s in range(n_subjects):
            # Per-subject filter perturbation: ~5% additive noise
            # on the filter banks. Models subject-to-subject voxel
            # variability that the noise-ceiling uses.
            subject_noise_scale = 0.05
            for i in range(n_images):
                c = int(classes[i])
                filt = cat_filters[c] + subject_noise_scale * rng.standard_normal(
                    cat_filters[c].shape
                ).astype(np.float32)
                response = flat_imgs[i] @ filt.T  # (n_v,)
                signal[s, i, :] = response
        # Calibrate per-ROI noise to the requested SNR.
        signal_var = float(signal.var())
        # noise_var solves snr = signal_var / (signal_var + noise_var)
        noise_var = signal_var * (1.0 / max(snr, 1e-6) - 1.0)
        noise = rng.standard_normal(signal.shape).astype(np.float32) * np.sqrt(
            max(noise_var, 0.0)
        )
        roi_signals[roi] = torch.from_numpy(signal + noise)

    return SyntheticCorticalDataset(
        images=torch.from_numpy(images),
        roi_signals=roi_signals,
        image_classes=torch.from_numpy(classes.astype(np.int64)),
        category_names=_CATEGORIES,
        snr=float(snr),
    )


__all__ = [
    "SyntheticCorticalDataset",
    "make_synthetic_cichy_like",
]
