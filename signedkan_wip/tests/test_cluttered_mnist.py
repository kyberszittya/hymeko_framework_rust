"""Unit tests for the Cluttered MNIST generator."""
from __future__ import annotations
import pytest
import torch

from signedkan_wip.src.vision.cluttered_mnist import ClutteredMNIST, ClutteredSample, collate_cluttered


@pytest.fixture(scope="module")
def ds():
    return ClutteredMNIST(n_samples=64, canvas=64, max_digits=3,
                            min_digits=1, seed=42, download=True)


def test_dataset_length(ds):
    assert len(ds) == 64


def test_sample_shape(ds):
    s = ds[0]
    assert isinstance(s, ClutteredSample)
    assert s.image.shape == (1, 64, 64)
    assert s.image.dtype == torch.float32
    assert s.bboxes.shape[1] == 4
    assert s.labels.shape[0] == s.bboxes.shape[0]


def test_image_pixel_range(ds):
    s = ds[0]
    assert s.image.min() >= 0.0
    assert s.image.max() <= 1.0
    # Non-empty image (some digit got pasted).
    assert s.image.max() > 0.0


def test_bbox_in_canvas(ds):
    """All bboxes must be inside [0, canvas]² with positive width/height."""
    for i in range(16):
        s = ds[i]
        for (x0, y0, x1, y1) in s.bboxes.tolist():
            assert 0 <= x0 < x1 <= 64, f"sample {i}: bad x range {x0}..{x1}"
            assert 0 <= y0 < y1 <= 64, f"sample {i}: bad y range {y0}..{y1}"


def test_labels_in_range(ds):
    for i in range(16):
        s = ds[i]
        for lbl in s.labels.tolist():
            assert 0 <= lbl <= 9


def test_n_digits_in_range(ds):
    """Every sample has between min_digits and max_digits annotations."""
    for i in range(32):
        s = ds[i]
        # Floor is min_digits (= 1) unless rejection sampling failed,
        # in which case we may have fewer.  Ceil is max_digits.
        assert 0 <= len(s.bboxes) <= 3


def test_determinism_by_index(ds):
    """ds[i] is deterministic: two calls return identical tensors."""
    s1 = ds[7]
    s2 = ds[7]
    assert torch.equal(s1.image, s2.image)
    assert torch.equal(s1.bboxes, s2.bboxes)
    assert torch.equal(s1.labels, s2.labels)


def test_determinism_across_seeds():
    """Different master seeds produce different samples."""
    ds_a = ClutteredMNIST(n_samples=4, canvas=64, seed=0, download=False)
    ds_b = ClutteredMNIST(n_samples=4, canvas=64, seed=1, download=False)
    # At least one sample differs.
    diffs = sum(
        not torch.equal(ds_a[i].image, ds_b[i].image)
        for i in range(4)
    )
    assert diffs >= 1


def test_collate_stacks_images(ds):
    batch = [ds[i] for i in range(4)]
    imgs, bboxes, labels = collate_cluttered(batch)
    assert imgs.shape == (4, 1, 64, 64)
    assert len(bboxes) == 4
    assert len(labels) == 4
    # Each entry is a tensor (possibly empty).
    assert all(b.shape[1] == 4 for b in bboxes)


def test_iou_constraint(ds):
    """No two bboxes within a sample exceed the iou_tolerance."""
    from signedkan_wip.src.vision.cluttered_mnist import ClutteredMNIST as CM
    for i in range(16):
        s = ds[i]
        for a in range(len(s.bboxes)):
            for b in range(a + 1, len(s.bboxes)):
                iou = CM._iou(tuple(s.bboxes[a].int().tolist()),
                                tuple(s.bboxes[b].int().tolist()))
                assert iou <= 0.10 + 1e-6, f"iou too high in sample {i}: {iou}"


def test_smaller_canvas_rejected():
    with pytest.raises(AssertionError):
        ClutteredMNIST(n_samples=4, canvas=16)


def test_invalid_digit_range_rejected():
    with pytest.raises(AssertionError):
        ClutteredMNIST(n_samples=4, max_digits=2, min_digits=5)
