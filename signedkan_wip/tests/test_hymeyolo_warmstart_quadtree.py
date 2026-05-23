"""Unit tests for the quadtree-driven HymeYOLO warmstart."""

from __future__ import annotations

import torch

from signedkan_wip.src.vision.hymeyolo_warmstart_quadtree import (
    quadtree_centres,
    warmstart_query_corners_quadtree,
)


def _blob_image(H: int, W: int, blobs: list[tuple[int, int, int, int, float]]
                ) -> torch.Tensor:
    """Build a (1, 3, H, W) image with rectangular blobs.

    Each blob is (y0, y1, x0, x1, amplitude).  Background is 0.01.
    """
    img = torch.zeros(1, 3, H, W) + 0.01
    for y0, y1, x0, x1, amp in blobs:
        img[0, :, y0:y1, x0:x1] = amp
    return img


# ---------------- quadtree_centres ---------------------------------


def test_centres_shape_and_range():
    img = _blob_image(224, 224, [(50, 170, 50, 170, 1.0)])
    out = quadtree_centres(img, m=8, patch_size_initial=56, patch_size_min=14)
    assert out.shape == (8, 2)
    assert out.dtype == torch.float32
    assert (out >= 0.05).all()
    assert (out <= 0.95).all()


def test_centres_concentrate_on_two_blobs():
    """Two distinct blobs ⇒ at least one centre near each."""
    img = _blob_image(
        224, 224,
        [(40, 80, 40, 80, 1.0), (160, 200, 160, 200, 0.7)],
    )
    out = quadtree_centres(img, m=6, patch_size_initial=56, patch_size_min=14)
    pts = out * torch.tensor([224.0, 224.0])

    def near(p, cy, cx, tol=40):
        return ((p[0] - cy).abs() <= tol) and ((p[1] - cx).abs() <= tol)

    hits1 = sum(1 for p in pts if near(p, 60, 60, tol=40))
    hits2 = sum(1 for p in pts if near(p, 180, 180, tol=40))
    assert hits1 >= 1, f"expected >=1 centre near blob1; got pts={pts.tolist()}"
    assert hits2 >= 1, f"expected >=1 centre near blob2; got pts={pts.tolist()}"


def test_centres_deterministic_for_same_input():
    img = _blob_image(224, 224, [(50, 170, 50, 170, 1.0)])
    a = quadtree_centres(img, m=4, patch_size_initial=56, patch_size_min=14, seed=0)
    b = quadtree_centres(img, m=4, patch_size_initial=56, patch_size_min=14, seed=0)
    assert torch.equal(a, b)


def test_centres_handles_uniform_image_via_fallback():
    """Uniform image ⇒ quadtree degenerates; uniform fallback is used."""
    img = torch.full((1, 3, 224, 224), 0.5)
    out = quadtree_centres(img, m=4, patch_size_initial=56, patch_size_min=14)
    assert out.shape == (4, 2)
    assert (out >= 0.05).all() and (out <= 0.95).all()


def test_centres_padding_when_quadtree_underprovides():
    """Request m larger than quadtree produces ⇒ pad with uniform."""
    img = _blob_image(224, 224, [(50, 170, 50, 170, 1.0)])
    # patch_size_initial=224 (single root cell), no subdivision possible
    # under patch_size_min=224 ⇒ at most 1 anchor.
    out = quadtree_centres(img, m=4, patch_size_initial=224, patch_size_min=224)
    assert out.shape == (4, 2)


def test_centres_rejects_bad_input():
    import pytest

    with pytest.raises(ValueError):
        quadtree_centres(torch.zeros(3, 224, 224), m=4)  # ndim != 4
    with pytest.raises(ValueError):
        quadtree_centres(torch.zeros(0, 3, 224, 224), m=4)  # empty batch
    with pytest.raises(ValueError):
        quadtree_centres(torch.zeros(1, 3, 224, 224), m=0)  # m < 1


# ---------------- warmstart_query_corners_quadtree -----------------


class _MockModel(torch.nn.Module):
    def __init__(self, n_box: int = 6, n_circle: int = 0, circle_k: int = 8):
        super().__init__()
        self.n_box_queries = n_box
        self.n_circle_queries = n_circle
        self.circle_k = circle_k
        if n_box > 0:
            self.box_corners = torch.nn.Parameter(torch.zeros(n_box, 4, 2))
        if n_circle > 0:
            self.circle_corners = torch.nn.Parameter(
                torch.zeros(n_circle, circle_k, 2)
            )


def test_warmstart_writes_box_corners_in_place():
    model = _MockModel(n_box=6, n_circle=0)
    img = _blob_image(224, 224, [(50, 170, 50, 170, 1.0)])
    old = model.box_corners.detach().clone()
    out = warmstart_query_corners_quadtree(
        model, img, patch_size_initial=56, patch_size_min=14,
    )
    new = model.box_corners.detach()
    assert not torch.equal(old, new), "warmstart did not modify box_corners"
    assert out["box_corners"].shape == (6, 4, 2)
    # all coords in [0.05, 0.95]+box_size offset ⇒ widen a bit
    assert (new >= -0.05).all() and (new <= 1.05).all()


def test_warmstart_handles_circle_corners():
    model = _MockModel(n_box=4, n_circle=2, circle_k=8)
    img = _blob_image(224, 224, [(50, 170, 50, 170, 1.0)])
    out = warmstart_query_corners_quadtree(
        model, img, patch_size_initial=56, patch_size_min=14,
    )
    assert out["box_corners"].shape == (4, 4, 2)
    assert out["circle_corners"].shape == (2, 8, 2)


def test_warmstart_noop_when_no_queries():
    model = _MockModel(n_box=0, n_circle=0)
    img = _blob_image(224, 224, [(50, 170, 50, 170, 1.0)])
    out = warmstart_query_corners_quadtree(model, img)
    assert out["box_corners"].numel() == 0
    assert out["circle_corners"].numel() == 0
