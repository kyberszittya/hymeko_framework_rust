"""Tests for the GömbSoma cortical-benchmark package shipped in
``signedkan_wip/src/cortical/`` (Slice 1 of the cortical-benchmark
implementation plan, 2026-05-19).

Synthetic-data only — no GPU, no Cichy 92 download. Validates:
  - dataset generator shape correctness
  - feature extractor binning correctness
  - BrainScorer extremes (perfect fit, pure noise, capped at 1.0)
  - noise-ceiling sanity
  - parameter-match assertion utility
  - ResNet baseline produces shape-compatible features
  - end-to-end smoke (BrainScorer + ResNet baseline + synthetic)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from signedkan_wip.src.cortical import (
    BinningConfig,
    BrainScore,
    BrainScorer,
    PerDepthFeatures,
    ResNetTinyCortical,
    SyntheticCorticalDataset,
    assert_param_match,
    count_parameters,
    make_synthetic_cichy_like,
    score_all_rois,
)


# ─── Synthetic dataset shape ─────────────────────────────────────────


def test_synthetic_dataset_default_shape():
    ds = make_synthetic_cichy_like(seed=0)
    assert ds.n_images == 92
    assert ds.n_subjects == 16
    assert ds.images.shape == (92, 1, 64, 64)
    assert ds.image_classes.shape == (92,)
    # ROI voxel counts match the Cichy paper defaults.
    assert tuple(ds.roi_names) == ("V1", "V2", "V4")
    assert ds.roi_signals["V1"].shape == (16, 92, 100)
    assert ds.roi_signals["V2"].shape == (16, 92, 80)
    assert ds.roi_signals["V4"].shape == (16, 92, 60)


def test_synthetic_dataset_deterministic():
    a = make_synthetic_cichy_like(seed=42, n_images=20, n_subjects=4)
    b = make_synthetic_cichy_like(seed=42, n_images=20, n_subjects=4)
    assert torch.equal(a.images, b.images)
    assert torch.equal(a.image_classes, b.image_classes)
    for roi in a.roi_names:
        assert torch.equal(a.roi_signals[roi], b.roi_signals[roi])


def test_synthetic_dataset_custom_image_size():
    ds = make_synthetic_cichy_like(image_size=48, n_images=8, n_subjects=2, seed=1)
    assert ds.images.shape == (8, 1, 48, 48)


def test_synthetic_dataset_rgb_channels():
    ds = make_synthetic_cichy_like(in_channels=3, n_images=4, n_subjects=2, seed=1)
    assert ds.images.shape[1] == 3


# ─── BrainScorer extremes ────────────────────────────────────────────


def test_brain_scorer_perfect_fit_when_features_equal_signal():
    rng = np.random.default_rng(0)
    n_images, n_voxels = 30, 5
    Y = rng.standard_normal((n_images, n_voxels)).astype(np.float32)
    X = Y.copy()  # features = signal
    scorer = BrainScorer(n_pls_components=4, n_cv_folds=5, seed=0)
    score = scorer.score(X, Y, roi="test")
    # Identity mapping should give very high r²; allow some CV slack.
    assert score.r_squared > 0.9, f"perfect fit got r²={score.r_squared}"


def test_brain_scorer_zero_signal_when_features_are_pure_noise():
    rng = np.random.default_rng(0)
    n_images, n_voxels = 60, 5
    X = rng.standard_normal((n_images, 8)).astype(np.float32)
    Y = rng.standard_normal((n_images, n_voxels)).astype(np.float32)
    scorer = BrainScorer(n_pls_components=4, n_cv_folds=5, seed=0)
    score = scorer.score(X, Y, roi="test")
    # Pure noise: r² should be near zero (within noise band).
    assert abs(score.r_squared) < 0.3, f"expected near-zero r², got {score.r_squared}"


def test_brain_scorer_corrected_clamped_to_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4)).astype(np.float32)
    Y = X.copy()
    scorer = BrainScorer(n_pls_components=2, n_cv_folds=4)
    # Pretend the noise ceiling is 0.5; r² will be ~1.0, so
    # corrected = min(1.0, r²/0.5) = 1.0 (clamped).
    score = scorer.score(X, Y, roi="test", noise_ceiling=0.5)
    assert 0.0 <= score.noise_ceiling_corrected <= 1.0


def test_brain_scorer_raises_on_image_axis_mismatch():
    X = np.zeros((10, 4))
    Y = np.zeros((8, 5))
    with pytest.raises(ValueError, match="image-axis mismatch"):
        BrainScorer().score(X, Y, roi="test")


def test_brain_scorer_raises_on_too_few_images():
    X = np.zeros((4, 4))
    Y = np.zeros((4, 5))
    with pytest.raises(ValueError, match="need n_images"):
        BrainScorer(n_cv_folds=5).score(X, Y, roi="test")


# ─── Noise ceiling ───────────────────────────────────────────────────


def test_noise_ceiling_high_when_subjects_agree():
    # All subjects share the same signal: noise ceiling should be near 1.
    n_subjects, n_images, n_voxels = 8, 30, 6
    rng = np.random.default_rng(0)
    base = rng.standard_normal((n_images, n_voxels)).astype(np.float32)
    Y = np.tile(base[None, :, :], (n_subjects, 1, 1))
    Y += 0.01 * rng.standard_normal(Y.shape).astype(np.float32)
    scorer = BrainScorer(seed=0)
    nc = scorer.noise_ceiling(Y, roi="test")
    assert nc > 0.9, f"high-agreement ceiling should be ~1.0, got {nc}"


def test_noise_ceiling_low_when_subjects_independent():
    # Each subject is pure independent noise: ceiling should be near 0.
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((8, 30, 6)).astype(np.float32)
    scorer = BrainScorer(seed=0)
    nc = scorer.noise_ceiling(Y, roi="test")
    # Pure noise → expect ceiling near 0 but possibly slightly above
    # due to chance correlations on a small sample.
    assert nc < 0.5, f"pure-noise ceiling should be small, got {nc}"


def test_noise_ceiling_single_subject_is_unity():
    # Degenerate case: only 1 subject → no split possible → return 1.0.
    Y = torch.zeros((1, 30, 6))
    nc = BrainScorer().noise_ceiling(Y, roi="test")
    assert nc == 1.0


# ─── Feature extractor binning ──────────────────────────────────────


def test_binning_config_default_depths():
    cfg = BinningConfig()
    assert cfg.depths == (0, 1, 2)
    assert cfg.n_bins(0) == 4    # 2x2
    assert cfg.n_bins(1) == 16   # 4x4
    assert cfg.n_bins(2) == 64   # 8x8


def test_bin_features_vectorised_matches_loop_reference():
    """Regression for the 2026-05-19 scatter_add_ vectorisation of
    :meth:`CorticalFeatureExtractor._bin_features`. Compares against a
    loop reference for an explicit handful of (position, feature)
    pairs; ensures bin assignment + mean averaging is unchanged."""
    from signedkan_wip.src.cortical.features import CorticalFeatureExtractor

    class _Dummy(torch.nn.Module):
        def forward(self, x):  # noqa: D401
            return torch.empty(0), None

    extr = CorticalFeatureExtractor(
        backbone=_Dummy(), image_h=8, image_w=8, d_hidden=3,
        binning_config=BinningConfig(bins_per_depth={0: (2, 2)}),
    )
    # Anchors at the four quadrant centres + duplicates in bin 0.
    positions = torch.tensor([
        [1.0, 1.0],   # bin 0 (top-left)
        [1.0, 5.0],   # bin 1 (top-right)
        [5.0, 1.0],   # bin 2 (bottom-left)
        [5.0, 5.0],   # bin 3 (bottom-right)
        [2.0, 2.0],   # bin 0 again
    ])
    features = torch.tensor([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
    ])
    out = extr._bin_features(features, positions, n_h=2, n_w=2)
    # bin 0 averages rows 0 + 4 = (3.0, 3.0, 3.0)
    # bin 1 = row 1; bin 2 = row 2; bin 3 = row 3.
    expected = torch.tensor([
        [3.0, 3.0, 3.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
    ])
    assert torch.allclose(out, expected, atol=1e-6), out


def test_bin_features_handles_empty_input():
    """Pre-condition: zero anchors → all-zero bin output, no NaN."""
    from signedkan_wip.src.cortical.features import CorticalFeatureExtractor

    class _Dummy(torch.nn.Module):
        def forward(self, x):
            return torch.empty(0), None

    extr = CorticalFeatureExtractor(
        backbone=_Dummy(), image_h=8, image_w=8, d_hidden=4,
        binning_config=BinningConfig(bins_per_depth={0: (2, 2)}),
    )
    out = extr._bin_features(
        torch.empty(0, 4), torch.empty(0, 2), n_h=2, n_w=2,
    )
    assert out.shape == (4, 4)
    assert torch.all(out == 0)


# ─── ResNet baseline ─────────────────────────────────────────────────


def test_resnet_tiny_emits_per_depth_features():
    cfg = BinningConfig(bins_per_depth={0: (2, 2), 1: (4, 4)})
    model = ResNetTinyCortical(
        image_h=32, image_w=32, in_channels=1, d_hidden=8,
        binning_config=cfg,
    )
    img = torch.randn(1, 32, 32)
    pdf = model.extract_one(img)
    assert isinstance(pdf, PerDepthFeatures)
    assert set(pdf.per_depth.keys()) == {0, 1}
    assert pdf.per_depth[0].shape == (4, 8)
    assert pdf.per_depth[1].shape == (16, 8)
    # Flat is concatenation of all per_depth: 4*8 + 16*8 = 160.
    assert pdf.flat.shape == (160,)


def test_resnet_tiny_batch_shape():
    model = ResNetTinyCortical(
        image_h=32, image_w=32, in_channels=1, d_hidden=4,
    )
    imgs = torch.randn(5, 1, 32, 32)
    feats = model.extract_batch(imgs)
    assert feats.shape[0] == 5
    assert feats.shape[1] == model.total_d


def test_assert_param_match_succeeds_for_identical():
    a = ResNetTinyCortical(image_h=32, image_w=32, in_channels=1, d_hidden=8)
    b = ResNetTinyCortical(image_h=32, image_w=32, in_channels=1, d_hidden=8)
    assert_param_match(a, b, factor=1.05)


def test_assert_param_match_raises_for_mismatch():
    a = ResNetTinyCortical(image_h=32, image_w=32, in_channels=1, d_hidden=4)
    b = ResNetTinyCortical(image_h=32, image_w=32, in_channels=1, d_hidden=32)
    with pytest.raises(AssertionError):
        assert_param_match(a, b, factor=1.5)


# ─── End-to-end smoke ────────────────────────────────────────────────


def test_end_to_end_smoke_resnet_synthetic():
    """Full pipeline: synthetic data → ResNet features → BrainScorer →
    BrainScore per ROI. All shapes and types correct, all scores
    finite, corrected scores in [0, 1]."""
    ds = make_synthetic_cichy_like(
        n_images=20, n_subjects=4, image_size=32, seed=0
    )
    cfg = BinningConfig(bins_per_depth={0: (2, 2), 1: (4, 4), 2: (8, 8)})
    resnet = ResNetTinyCortical(
        image_h=32, image_w=32, in_channels=1, d_hidden=8,
        binning_config=cfg,
    )
    with torch.no_grad():
        features = resnet.extract_batch(ds.images)
    assert features.shape[0] == ds.n_images
    scorer = BrainScorer(n_pls_components=4, n_cv_folds=4, seed=0)
    results = score_all_rois(scorer, features, ds.roi_signals)
    assert set(results.keys()) == {"V1", "V2", "V4"}
    for roi, score in results.items():
        assert isinstance(score, BrainScore)
        assert score.roi == roi
        assert np.isfinite(score.r_squared)
        assert np.isfinite(score.noise_ceiling)
        assert 0.0 <= score.noise_ceiling_corrected <= 1.0
        assert score.n_voxels > 0


def test_brain_score_dataclass_is_frozen():
    score = BrainScore(
        roi="V1", r_squared=0.5, noise_ceiling=0.7,
        noise_ceiling_corrected=0.5/0.7, n_voxels=100,
        n_pls_components=25, n_cv_folds=5,
    )
    with pytest.raises(Exception):
        score.roi = "V2"  # type: ignore[misc]
