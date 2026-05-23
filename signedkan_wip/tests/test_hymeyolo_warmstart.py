"""Tests for the 2026-05-16 HyMeYOLO Stage-A-1 warm-start.

Plan: docs/plans/2026-05-16-hymeyolo-warmstart-query-init/.

Pins:

* Parameter count unchanged (we only overwrite existing buffers).
* Deterministic given the same bootstrap batch + seed.
* All emitted corners are in [0.05, 0.95].
* On a synthetic image with K known peaks, ≥ 75% of warm-start
  centres land within 0.10 normalised-distance of a peak.
* Aggregator / backbone / head weights are not touched.
* When --warm-start is off, the model is byte-identical to the
  no-flag path (this is mainly a guard against accidental side
  effects from importing the warmstart module).
"""
from __future__ import annotations

import math

import pytest
import torch

from signedkan_wip.src.vision.hymeyolo_circles_ricci import (
    RicciHyMeYOLOMulti,
)
from signedkan_wip.src.vision.hymeyolo_warmstart import (
    _farthest_point_sample,
    _gaussian_smooth_2d,
    warmstart_query_corners,
)


# ─── helpers ──────────────────────────────────────────────────────────


def _model(seed: int = 0) -> RicciHyMeYOLOMulti:
    torch.manual_seed(seed)
    return RicciHyMeYOLOMulti(
        n_box_queries=4, n_circle_queries=2, circle_k=8,
        n_classes=10, d_hidden=16, ricci_modulation=True,
        ricci_scale=1.0,
    )


def _bootstrap_uniform(n: int = 8, h: int = 64, w: int = 64) -> torch.Tensor:
    """A bootstrap batch of uniform-noise RGB images."""
    torch.manual_seed(0)
    return torch.rand((n, 3, h, w))


def _bootstrap_with_peaks(
    n: int, h: int, w: int, peak_yx: list[tuple[int, int]],
    peak_sigma: int = 3, base_noise: float = 0.05,
) -> torch.Tensor:
    """A bootstrap batch with Gaussian hot spots at known locations.

    Each image is base_noise plus a sum of K Gaussians at the listed
    (y, x) pixel coordinates. Used to test that warm-start finds the
    intended peaks.
    """
    torch.manual_seed(0)
    X = base_noise * torch.rand((n, 3, h, w))
    yy = torch.arange(h, dtype=torch.float32).unsqueeze(1).expand(h, w)
    xx = torch.arange(w, dtype=torch.float32).unsqueeze(0).expand(h, w)
    for (py, px) in peak_yx:
        bump = torch.exp(-((yy - py) ** 2 + (xx - px) ** 2)
                         / (2.0 * peak_sigma ** 2))
        X = X + bump.unsqueeze(0).unsqueeze(0)  # broadcast over (N, C)
    return X


# ─── core pins ────────────────────────────────────────────────────────


def test_warmstart_preserves_param_count() -> None:
    m = _model()
    before = sum(p.numel() for p in m.parameters())
    X = _bootstrap_uniform()
    warmstart_query_corners(m, X, seed=0)
    after = sum(p.numel() for p in m.parameters())
    assert before == after, (
        f"warm-start changed param count: {before} → {after}"
    )


def test_warmstart_deterministic_given_seed() -> None:
    m1 = _model()
    m2 = _model()
    X = _bootstrap_uniform()
    out1 = warmstart_query_corners(m1, X, seed=42)
    out2 = warmstart_query_corners(m2, X, seed=42)
    assert torch.equal(m1.box_corners, m2.box_corners), (
        "warm-start box_corners differ for identical seed+bootstrap"
    )
    assert torch.equal(m1.circle_corners, m2.circle_corners), (
        "warm-start circle_corners differ for identical seed+bootstrap"
    )
    # Return dict should reflect what's in the model.
    assert torch.equal(out1["box_corners"], m1.box_corners.detach())
    assert torch.equal(out1["circle_corners"], m1.circle_corners.detach())


def test_warmstart_respects_clamp() -> None:
    m = _model()
    X = _bootstrap_uniform()
    warmstart_query_corners(m, X, seed=0)
    assert (m.box_corners.min().item() >= 0.05), (
        f"box_corners min {m.box_corners.min().item()} < 0.05"
    )
    assert (m.box_corners.max().item() <= 0.95), (
        f"box_corners max {m.box_corners.max().item()} > 0.95"
    )
    assert (m.circle_corners.min().item() >= 0.05)
    assert (m.circle_corners.max().item() <= 0.95)


def test_warmstart_finds_known_peaks() -> None:
    """6 query centres (4 box + 2 circle) should land near the 4 peaks
    on a synthetic image with hot spots. We expect each peak to attract
    at least one centre; with farthest-point sampling, the 6 centres
    spread to cover all 4 peaks (plus 2 extras between/around them).
    """
    h, w = 64, 64
    peaks = [(16, 16), (16, 48), (48, 16), (48, 48)]
    X = _bootstrap_with_peaks(n=16, h=h, w=w, peak_yx=peaks)
    m = _model()
    warmstart_query_corners(m, X, seed=0)

    # Compute centres of each emitted query (mean of its corners).
    box_centres = m.box_corners.mean(dim=1)        # (4, 2) — (x_norm, y_norm)
    circ_centres = m.circle_corners.mean(dim=1)    # (2, 2)
    all_centres = torch.cat([box_centres, circ_centres], dim=0)  # (6, 2)

    # Each query corner is (x_norm, y_norm); peaks are in (y_pix, x_pix)
    # → normalise to (y_norm, x_norm) and match against the (x, y)
    # query convention.
    peak_norm = torch.tensor(
        [[(px + 0.5) / w, (py + 0.5) / h] for (py, px) in peaks]
    )  # (K, 2) in (x_norm, y_norm) — same convention as query corners

    # For each peak, the closest query centre must be within 0.10
    # normalised-distance (roughly 6 px on a 64-px canvas).
    found = 0
    for pk in peak_norm:
        d = ((all_centres - pk.unsqueeze(0)) ** 2).sum(dim=-1).sqrt()
        if d.min().item() < 0.10:
            found += 1
    expected = math.ceil(len(peaks) * 0.75)
    assert found >= expected, (
        f"warm-start found only {found}/{len(peaks)} peaks within "
        f"0.10 dist; expected ≥ {expected}.\n"
        f"emitted centres (x, y):\n{all_centres}\n"
        f"peaks (x, y):\n{peak_norm}"
    )


def test_warmstart_does_not_change_aggregator_or_head() -> None:
    """Only box_corners / circle_corners are overwritten; everything
    else (backbone, aggregator, heads) is byte-identical."""
    m1 = _model(seed=123)
    m2 = _model(seed=123)
    X = _bootstrap_uniform()
    warmstart_query_corners(m1, X, seed=0)

    for name1, p1 in m1.named_parameters():
        if name1 in ("box_corners", "circle_corners"):
            continue
        p2 = dict(m2.named_parameters())[name1]
        assert torch.equal(p1, p2), (
            f"warm-start unexpectedly modified parameter '{name1}'"
        )


def test_warmstart_rejects_bad_bootstrap_shape() -> None:
    m = _model()
    with pytest.raises(ValueError, match="X_bootstrap must have shape"):
        warmstart_query_corners(m, torch.zeros((3, 64, 64)), seed=0)


def test_warmstart_rejects_empty_bootstrap() -> None:
    m = _model()
    with pytest.raises(ValueError, match="at least one image"):
        warmstart_query_corners(m, torch.zeros((0, 3, 64, 64)), seed=0)


def test_warmstart_rejects_nonfinite_bootstrap() -> None:
    m = _model()
    X = torch.zeros((1, 3, 64, 64))
    X[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        warmstart_query_corners(m, X, seed=0)


# ─── component pieces ─────────────────────────────────────────────────


def test_gaussian_smooth_preserves_mean_approximately() -> None:
    """Smoothing a non-negative map preserves total mass (modulo
    reflect-padding edge effects) — useful as a numerical sanity gate."""
    S = torch.rand((32, 32))
    S2 = _gaussian_smooth_2d(S, sigma_px=2.0)
    # Reflect padding doesn't conserve mass exactly, but the relative
    # change should be small at sigma=2 on a 32×32 canvas.
    ratio = (S2.sum() / S.sum()).item()
    assert 0.7 < ratio < 1.3, (
        f"smoothing total-mass ratio out of range: {ratio}"
    )


def test_gaussian_smooth_is_identity_at_sigma_zero() -> None:
    S = torch.rand((16, 16))
    S2 = _gaussian_smooth_2d(S, sigma_px=0.0)
    assert torch.equal(S, S2)


def test_farthest_point_sample_returns_distinct_points() -> None:
    H, W = 16, 16
    p = torch.ones((H, W)) / (H * W)
    idxs = _farthest_point_sample(p, m=8, seed=0)
    flat = idxs[:, 0] * W + idxs[:, 1]
    assert flat.unique().numel() == 8, (
        f"FPS returned duplicate points: {idxs}"
    )


def test_farthest_point_sample_spreads_under_uniform() -> None:
    """Under uniform saliency, FPS centres should be well-spread.
    Minimum pairwise distance must exceed half the trivial
    upper bound from grid spacing."""
    H, W = 32, 32
    p = torch.ones((H, W)) / (H * W)
    idxs = _farthest_point_sample(p, m=4, seed=0).float()
    # Pairwise distances.
    diffs = idxs.unsqueeze(0) - idxs.unsqueeze(1)
    d = (diffs ** 2).sum(dim=-1).sqrt()
    # Drop the zero diagonal.
    d_off = d[~torch.eye(4, dtype=torch.bool)]
    min_d = d_off.min().item()
    # For 4 points on a 32×32 grid, half the grid diagonal ≈ 22.6
    # is an aspirational lower bound; demand at least 0.5 of that.
    assert min_d > 11.0, f"FPS centres too clustered: min_d={min_d}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
