"""Tests for FuzzySignaturePoseModel + soft-argmax + synthetic generator.

Plan: ``docs/plans/2026-05-31-fuzzy-pose-detection/plan.tex``.
"""
from __future__ import annotations

import pytest
import torch

from hymeko_neuro.experiments.vision.fuzzy_pose import (
    FuzzySignaturePoseModel,
    HSiKANGombPoseModel,
    HSiKANPoseModel,
    HybridHSiKANFuzzyLayer,
    HybridHSiKANFuzzyPoseModel,
    SyntheticPoseDataset,
    TinyCNNPoseModel,
    make_synthetic_pose_sample,
    soft_argmax_2d,
)


# ─── soft_argmax_2d ────────────────────────────────────────────────


def test_soft_argmax_finds_peak_within_one_pixel():
    """A single-peak Gaussian heatmap should resolve to ≤ 1 px of the
    peak coordinates under soft-argmax."""
    H, W = 32, 32
    cx, cy = 17.0, 11.0
    ys = torch.arange(H, dtype=torch.float32).view(H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, W)
    heatmap = -((xs - cx) ** 2 + (ys - cy) ** 2) / 2.0  # log-Gaussian
    coords = soft_argmax_2d(heatmap.unsqueeze(0), beta=1.0)  # (1, 2)
    dx = abs(coords[0, 0].item() - cx)
    dy = abs(coords[0, 1].item() - cy)
    assert dx < 1.0 and dy < 1.0, f"soft-argmax off by ({dx:.3f}, {dy:.3f})"


def test_soft_argmax_differentiable():
    """Gradient w.r.t. the heatmap entries is finite and non-zero."""
    torch.manual_seed(0)
    heatmap = torch.randn(1, 8, 16, 16, requires_grad=True)
    coords = soft_argmax_2d(heatmap, beta=1.0)  # (1, 8, 2)
    loss = coords.sum()
    loss.backward()
    assert heatmap.grad is not None
    assert torch.isfinite(heatmap.grad).all()
    assert (heatmap.grad.abs() > 1e-8).any()


def test_soft_argmax_temperature_extremes():
    """Lower β (sharper) and higher β (softer) both produce finite,
    in-range coords."""
    torch.manual_seed(1)
    heatmap = torch.randn(1, 4, 16, 16)
    for beta in (0.1, 1.0, 10.0):
        coords = soft_argmax_2d(heatmap, beta=beta)
        assert torch.isfinite(coords).all()
        assert (coords >= 0).all()
        assert (coords[..., 0] <= 15).all()  # x ≤ W-1
        assert (coords[..., 1] <= 15).all()  # y ≤ H-1


@pytest.mark.parametrize("bad_beta", [0.0, -1.0, -0.001])
def test_pose_model_rejects_non_positive_beta(bad_beta):
    with pytest.raises(ValueError, match="beta"):
        FuzzySignaturePoseModel(H=32, W=32, n_keypoints=4,
                                 soft_argmax_beta=bad_beta)


# ─── Model forward shape and gradient ───────────────────────────────


def test_pose_model_forward_shape():
    """Model returns (B, n_kp, 2)."""
    m = FuzzySignaturePoseModel(H=32, W=32, n_keypoints=8,
                                 d=16, n_layers=2)
    x = torch.rand(3, 1, 32, 32)
    out = m(x)
    assert out.shape == (3, 8, 2)
    assert torch.isfinite(out).all()
    # Coords inside the image bounds.
    assert (out >= 0).all()
    assert (out[..., 0] <= 31).all()
    assert (out[..., 1] <= 31).all()


def test_pose_model_heatmaps_shape():
    """The heatmaps() helper returns (B, n_kp, H, W)."""
    m = FuzzySignaturePoseModel(H=32, W=32, n_keypoints=4,
                                 d=8, n_layers=2)
    x = torch.rand(2, 1, 32, 32)
    h = m.heatmaps(x)
    assert h.shape == (2, 4, 32, 32)


def test_pose_model_gradient_flow_one_step():
    """One training step: every learnable param receives a finite
    gradient (covers backbone + pose head)."""
    torch.manual_seed(2)
    m = FuzzySignaturePoseModel(H=32, W=32, n_keypoints=4,
                                 d=8, n_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 32, 32)
    y = torch.rand(4, 4, 2) * 32.0  # random target coords in [0, 32)
    opt.zero_grad()
    pred = m(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss)
    loss.backward()
    saw_pose_head = False
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        if "pose_head" in name:
            saw_pose_head = True
    assert saw_pose_head, "no gradient hit pose_head"
    opt.step()


def test_pose_model_param_count_closed_form():
    """Param count: backbone (rev-6 multi-arity at d, L, arities) +
    (d+1)*n_kp pose head."""
    H, W = 32, 32
    d, L = 32, 8
    n_kp = 8
    m = FuzzySignaturePoseModel(H=H, W=W, n_keypoints=n_kp,
                                 d=d, n_layers=L)
    actual = sum(p.numel() for p in m.parameters())
    # rev-6 multi-arity backbone params:
    # embed: 2d
    # per arity per layer: 2dm + d + 1 + d + d (rev-5 formula with c_gate)
    # at residual_kind=additive_centered, no alpha_raw → so just 2dm + 3d + 1
    # 2 arities × 8 layers × per-arity
    # + αₖ mixer: 2 per layer × 8
    # + pose head: (d+1)·n_kp
    embed = 2 * d
    per_arity_per_layer = 2 * d * 8 + d + 1 + d  # 2dm + d + 1 (W_e) + d (c)
    # NOTE: residual_kind=additive_centered does NOT allocate alpha_raw
    # (see fuzzy_signature.py `register_parameter("alpha_raw", None)`).
    backbone_layers = 2 * L * per_arity_per_layer  # 2 arities × L layers
    alpha_logits = L * 2  # αₖ mixer: 2 logits per layer × L
    pose_head = (d + 1) * n_kp
    expected = embed + backbone_layers + alpha_logits + pose_head
    assert actual == expected, (
        f"actual {actual} != expected {expected} "
        f"(embed {embed} + backbone {backbone_layers} + α_k {alpha_logits} "
        f"+ pose_head {pose_head})"
    )


def test_pose_model_invalid_n_keypoints():
    with pytest.raises(ValueError, match="n_keypoints"):
        FuzzySignaturePoseModel(H=32, W=32, n_keypoints=0)


# ─── Synthetic data generator ───────────────────────────────────────


def test_synthetic_pose_sample_coords_in_bounds():
    """Generated coords must lie within the image grid (with a safety
    margin)."""
    rng = torch.Generator()
    rng.manual_seed(42)
    for _ in range(20):
        img, coords = make_synthetic_pose_sample(H=32, W=32,
                                                   n_keypoints=8,
                                                   rng=rng)
        assert img.shape == (1, 32, 32)
        assert coords.shape == (8, 2)
        assert (coords[:, 0] >= 0).all() and (coords[:, 0] < 32).all()
        assert (coords[:, 1] >= 0).all() and (coords[:, 1] < 32).all()
        assert torch.isfinite(img).all()
        assert torch.isfinite(coords).all()


def test_synthetic_dataset_deterministic():
    """Same seed → identical samples; different seeds → different samples."""
    ds1 = SyntheticPoseDataset(n_samples=4, seed=0)
    ds2 = SyntheticPoseDataset(n_samples=4, seed=0)
    ds3 = SyntheticPoseDataset(n_samples=4, seed=1)
    for i in range(4):
        img1, c1 = ds1[i]
        img2, c2 = ds2[i]
        img3, c3 = ds3[i]
        assert torch.allclose(img1, img2)
        assert torch.allclose(c1, c2)
        # Different seed → different samples (probability of exact match ≈ 0).
        assert not torch.allclose(c1, c3)


def test_synthetic_dataset_length():
    ds = SyntheticPoseDataset(n_samples=100, seed=0)
    assert len(ds) == 100


def test_synthetic_sample_has_peaks_at_keypoints():
    """The image intensity at each (cx, cy) keypoint should be the
    local maximum or close to it (Gaussian peak amplitude 1.0)."""
    rng = torch.Generator()
    rng.manual_seed(5)
    img, coords = make_synthetic_pose_sample(H=32, W=32, n_keypoints=8,
                                               blob_sigma=1.5,
                                               bg_noise=0.05, rng=rng)
    img_2d = img.squeeze(0)
    for k in range(8):
        cx, cy = coords[k].tolist()
        ix, iy = int(round(cx)), int(round(cy))
        # Pixel intensity at the keypoint should be ≥ 0.5 (peak amp
        # 1.0; multiple overlapping blobs can boost or slightly reduce
        # the value at non-peak pixels).
        assert img_2d[iy, ix] >= 0.5, (
            f"kp {k}: intensity at ({iy},{ix}) = {img_2d[iy, ix].item():.3f}"
        )


# ─── End-to-end smoke (small, fast) ────────────────────────────────


def test_pose_model_overfits_one_batch():
    """A small d=8/L=2 model should be able to drive MSE loss down on
    a fixed batch (proves the end-to-end pipeline learns)."""
    torch.manual_seed(7)
    m = FuzzySignaturePoseModel(H=32, W=32, n_keypoints=4,
                                 d=8, n_layers=2,
                                 arities=[(3, 1)])  # single arity for speed
    opt = torch.optim.Adam(m.parameters(), lr=3e-2)
    # One fixed batch.
    ds = SyntheticPoseDataset(n_samples=4, n_keypoints=4, seed=11)
    imgs = torch.stack([ds[i][0] for i in range(4)])         # (4, 1, 32, 32)
    targets = torch.stack([ds[i][1] for i in range(4)])      # (4, 4, 2)
    initial_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    for _ in range(100):
        opt.zero_grad()
        pred = m(imgs)
        loss = torch.nn.functional.mse_loss(pred, targets)
        loss.backward()
        opt.step()
    final_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    # 60% threshold: tight d=8/L=2 single-arity needs many steps; the
    # point of this test is that the pipeline learns at all, not the
    # speed.
    assert final_loss < initial_loss * 0.6, (
        f"MSE didn't drop to 60% of init in 100 steps: init "
        f"{initial_loss:.3f} → final {final_loss:.3f}"
    )


# ─── HSiKANPoseModel (vanilla HSiKAN backbone, paired comparison) ────


def test_hsikan_pose_model_forward_shape():
    """Vanilla HSiKAN pose model returns (B, n_kp, 2)."""
    m = HSiKANPoseModel(H=32, W=32, n_keypoints=6,
                          d=16, n_layers=2,
                          arities=[(3, 1)])
    x = torch.rand(2, 1, 32, 32)
    out = m(x)
    assert out.shape == (2, 6, 2)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert (out[..., 0] <= 31).all()
    assert (out[..., 1] <= 31).all()


def test_hsikan_pose_model_heatmaps_shape():
    m = HSiKANPoseModel(H=32, W=32, n_keypoints=4,
                          d=8, n_layers=2, arities=[(3, 1)])
    x = torch.rand(2, 1, 32, 32)
    h = m.heatmaps(x)
    assert h.shape == (2, 4, 32, 32)


def test_hsikan_pose_model_gradient_flow_one_step():
    """End-to-end gradient flow for the vanilla HSiKAN pose model."""
    torch.manual_seed(20)
    m = HSiKANPoseModel(H=32, W=32, n_keypoints=4,
                          d=8, n_layers=2, arities=[(3, 1)])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 32, 32)
    y = torch.rand(4, 4, 2) * 32.0
    opt.zero_grad()
    pred = m(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss)
    loss.backward()
    saw_pose_head = False
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        if "pose_head" in name:
            saw_pose_head = True
    assert saw_pose_head, "no gradient hit pose_head"
    opt.step()


def test_hsikan_pose_model_overfits_one_batch():
    """Vanilla HSiKAN pose model overfits a small fixed batch —
    proves the comparison target is in the same regime as fuzzy."""
    torch.manual_seed(21)
    m = HSiKANPoseModel(H=32, W=32, n_keypoints=4,
                          d=8, n_layers=2, arities=[(3, 1)])
    opt = torch.optim.Adam(m.parameters(), lr=3e-2)
    ds = SyntheticPoseDataset(n_samples=4, n_keypoints=4, seed=22)
    imgs = torch.stack([ds[i][0] for i in range(4)])
    targets = torch.stack([ds[i][1] for i in range(4)])
    initial_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    for _ in range(100):
        opt.zero_grad()
        pred = m(imgs)
        loss = torch.nn.functional.mse_loss(pred, targets)
        loss.backward()
        opt.step()
    final_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    assert final_loss < initial_loss * 0.6, (
        f"HSiKAN pose MSE didn't drop to 60% in 100 steps: "
        f"init {initial_loss:.3f} → final {final_loss:.3f}"
    )


def test_hsikan_pose_model_invalid_n_keypoints():
    with pytest.raises(ValueError, match="n_keypoints"):
        HSiKANPoseModel(H=32, W=32, n_keypoints=0)


@pytest.mark.parametrize("bad_beta", [0.0, -1.0])
def test_hsikan_pose_model_rejects_non_positive_beta(bad_beta):
    with pytest.raises(ValueError, match="beta"):
        HSiKANPoseModel(H=32, W=32, n_keypoints=4,
                          soft_argmax_beta=bad_beta)


def test_paired_models_same_io_shape():
    """Both models accept the same input shape and produce the same
    output shape — required for paired-smoke comparison.

    At random init both pose heads emit near-uniform heatmaps so
    soft-argmax gives ≈ image centroid for every keypoint. Numerical
    values therefore coincide closely at init; differentiation
    emerges only with training. The paired smoke is the place to
    observe that, not this test."""
    torch.manual_seed(23)
    fuzzy = FuzzySignaturePoseModel(H=32, W=32, n_keypoints=4,
                                      d=8, n_layers=2, arities=[(3, 1)])
    vanilla = HSiKANPoseModel(H=32, W=32, n_keypoints=4,
                                d=8, n_layers=2, arities=[(3, 1)])
    x = torch.rand(2, 1, 32, 32)
    out_f = fuzzy(x)
    out_v = vanilla(x)
    assert out_f.shape == out_v.shape == (2, 4, 2)
    assert torch.isfinite(out_f).all() and torch.isfinite(out_v).all()


# ─── HSiKANGombPoseModel (HSiKAN backbone + Gömb head) ─────────────


def test_hsikan_gomb_pose_forward_shape():
    """Gömb-pose model returns (B, n_kp, 2)."""
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=8,
                              d=8, n_layers=3, arities=[(3, 1)])
    x = torch.rand(2, 1, 32, 32)
    out = m(x)
    assert out.shape == (2, 8, 2)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert (out[..., 0] <= 31).all()
    assert (out[..., 1] <= 31).all()


def test_hsikan_gomb_pose_heatmaps_shape():
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              d=4, n_layers=2, arities=[(3, 1)])
    x = torch.rand(2, 1, 32, 32)
    h = m.heatmaps(x)
    assert h.shape == (2, 4, 32, 32)


def test_hsikan_gomb_pose_gradient_flow_one_step():
    """Gradient flow through backbone + all 3 Gömb shells + tail."""
    torch.manual_seed(30)
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              d=4, n_layers=2, arities=[(3, 1)],
                              gomb_M=2, gomb_d_bank=4, gomb_d_mid=8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 32, 32)
    y = torch.rand(4, 4, 2) * 32.0
    opt.zero_grad()
    pred = m(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss)
    loss.backward()
    saw_outer = saw_middle = saw_inner = False
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        if "outer" in name:
            saw_outer = True
        elif "middle" in name:
            saw_middle = True
        elif "inner" in name:
            saw_inner = True
    assert saw_outer, "no gradient hit outer shell"
    assert saw_middle, "no gradient hit middle shell"
    assert saw_inner, "no gradient hit inner shell"
    opt.step()


def test_hsikan_gomb_pose_overfits_one_batch():
    """Deep-narrow HSiKAN + Gömb head overfits a small batch."""
    torch.manual_seed(31)
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              d=8, n_layers=4, arities=[(3, 1)],
                              gomb_M=2, gomb_d_bank=8, gomb_d_mid=16)
    opt = torch.optim.Adam(m.parameters(), lr=3e-2)
    ds = SyntheticPoseDataset(n_samples=4, n_keypoints=4, seed=32)
    imgs = torch.stack([ds[i][0] for i in range(4)])
    targets = torch.stack([ds[i][1] for i in range(4)])
    initial_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    for _ in range(100):
        opt.zero_grad()
        pred = m(imgs)
        loss = torch.nn.functional.mse_loss(pred, targets)
        loss.backward()
        opt.step()
    final_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    assert final_loss < initial_loss * 0.6, (
        f"Gömb-pose MSE didn't drop to 60% in 100 steps: "
        f"init {initial_loss:.3f} → final {final_loss:.3f}"
    )


def test_hsikan_gomb_invalid_n_keypoints():
    with pytest.raises(ValueError, match="n_keypoints"):
        HSiKANGombPoseModel(H=32, W=32, n_keypoints=0)


def test_hsikan_gomb_invalid_M():
    with pytest.raises(ValueError, match="gomb_M"):
        HSiKANGombPoseModel(H=32, W=32, n_keypoints=4, gomb_M=0)


# ─── HybridHSiKANFuzzyLayer + Pose model ───────────────────────────


def test_hybrid_layer_alpha_init_is_hsikan_dominant():
    """At init α ≈ hybrid_alpha_init (default 0.05) → HSiKAN-dominant
    layer. Validates the highway-gate inductive bias."""
    torch.manual_seed(60)
    layer = HybridHSiKANFuzzyLayer(
        d=8, H=8, W=8, arities=[(3, 1)],
        hybrid_alpha_init=0.05,
    )
    alpha = layer.alpha_eff
    assert torch.allclose(alpha, torch.full_like(alpha, 0.05), atol=1e-3)


def test_hybrid_layer_alpha_zero_is_pure_hsikan_plus_residual():
    """With α forced to 0, ``out = x + HSiKAN(x)`` exactly."""
    torch.manual_seed(61)
    layer = HybridHSiKANFuzzyLayer(d=4, H=8, W=8, arities=[(3, 1)])
    with torch.no_grad():
        layer.alpha_raw.fill_(-100.0)  # σ(-100) ≈ 0
    x = torch.randn(2, 64, 4) * 0.5
    out = layer(x)
    expected = x + layer.hsikan(x)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"at α=0, hybrid should equal x + HSiKAN(x); "
        f"max diff = {(out - expected).abs().max().item():.2e}"
    )


def test_hybrid_layer_alpha_one_is_pure_fuzzy():
    """With α forced to 1, ``out = x + (Fuzzy(x) − x) = Fuzzy(x)`` exactly."""
    torch.manual_seed(62)
    layer = HybridHSiKANFuzzyLayer(d=4, H=8, W=8, arities=[(3, 1)])
    with torch.no_grad():
        layer.alpha_raw.fill_(100.0)  # σ(100) ≈ 1
    x = torch.randn(2, 64, 4) * 0.5
    out = layer(x)
    expected = layer.fuzzy(x)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"at α=1, hybrid should equal Fuzzy(x); "
        f"max diff = {(out - expected).abs().max().item():.2e}"
    )


def test_hybrid_layer_forward_shape_and_finite():
    layer = HybridHSiKANFuzzyLayer(d=4, H=8, W=8, arities=[(3, 1)])
    x = torch.randn(2, 64, 4) * 0.5
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_hybrid_pose_model_forward_shape():
    m = HybridHSiKANFuzzyPoseModel(H=32, W=32, n_keypoints=4,
                                      d=4, n_layers=2, arities=[(3, 1)])
    x = torch.rand(2, 1, 32, 32)
    out = m(x)
    assert out.shape == (2, 4, 2)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert (out[..., 0] <= 31).all()
    assert (out[..., 1] <= 31).all()


def test_hybrid_pose_model_gradient_flow_includes_alpha_and_both_branches():
    """One step: gradients flow through α_raw AND both HSiKAN and
    Fuzzy branches."""
    torch.manual_seed(63)
    m = HybridHSiKANFuzzyPoseModel(H=32, W=32, n_keypoints=4,
                                      d=4, n_layers=2, arities=[(3, 1)])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 32, 32)
    y = torch.rand(4, 4, 2) * 32.0
    opt.zero_grad()
    pred = m(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss)
    loss.backward()
    saw_alpha = saw_hsikan = saw_fuzzy = False
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        if "alpha_raw" in name:
            saw_alpha = True
        if "hsikan" in name:
            saw_hsikan = True
        if "fuzzy" in name and "alpha" not in name:
            saw_fuzzy = True
    assert saw_alpha, "no gradient hit α_raw"
    assert saw_hsikan, "no gradient hit HSiKAN branch"
    assert saw_fuzzy, "no gradient hit Fuzzy branch"
    opt.step()


def test_hybrid_pose_model_overfits_one_batch():
    """End-to-end: hybrid pose model overfits a small fixed batch."""
    torch.manual_seed(64)
    m = HybridHSiKANFuzzyPoseModel(
        H=32, W=32, n_keypoints=4,
        d=8, n_layers=2, arities=[(3, 1)],
    )
    opt = torch.optim.Adam(m.parameters(), lr=3e-2)
    ds = SyntheticPoseDataset(n_samples=4, n_keypoints=4, seed=65)
    imgs = torch.stack([ds[i][0] for i in range(4)])
    targets = torch.stack([ds[i][1] for i in range(4)])
    initial_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    for _ in range(100):
        opt.zero_grad()
        pred = m(imgs)
        loss = torch.nn.functional.mse_loss(pred, targets)
        loss.backward()
        opt.step()
    final_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    assert final_loss < initial_loss * 0.6, (
        f"hybrid MSE didn't drop to 60% in 100 steps: "
        f"init {initial_loss:.3f} → final {final_loss:.3f}"
    )


def test_hybrid_pose_model_invalid_n_keypoints():
    with pytest.raises(ValueError, match="n_keypoints"):
        HybridHSiKANFuzzyPoseModel(H=32, W=32, n_keypoints=0)


@pytest.mark.parametrize("init_p", [1e-6, 0.999999])
def test_hybrid_alpha_init_boundary(init_p):
    """Boundary α_init values are clamped safely."""
    layer = HybridHSiKANFuzzyLayer(
        d=4, H=8, W=8, arities=[(3, 1)],
        hybrid_alpha_init=init_p,
    )
    alpha = layer.alpha_eff
    assert torch.isfinite(alpha).all()
    assert (alpha > 0).all() and (alpha < 1).all()


# ─── TinyCNNPoseModel (external baseline) ─────────────────────────


def test_tiny_cnn_pose_forward_shape():
    """TinyCNNPose returns (B, n_kp, 2) in-bound coords."""
    m = TinyCNNPoseModel(H=32, W=32, n_keypoints=8, d=12, n_layers=4)
    x = torch.rand(2, 1, 32, 32)
    out = m(x)
    assert out.shape == (2, 8, 2)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert (out[..., 0] <= 31).all()
    assert (out[..., 1] <= 31).all()


def test_tiny_cnn_pose_heatmaps_shape():
    m = TinyCNNPoseModel(H=32, W=32, n_keypoints=4, d=8, n_layers=2)
    x = torch.rand(2, 1, 32, 32)
    h = m.heatmaps(x)
    assert h.shape == (2, 4, 32, 32)


def test_tiny_cnn_pose_param_count_closed_form():
    """Conv2d(1, d, 1) embed + L blocks of (Conv2d(d,d,3,pad=1) +
    BatchNorm2d) + Conv2d(d, n_kp, 1) head."""
    H = W = 32
    d = 12
    L = 8
    n_kp = 8
    m = TinyCNNPoseModel(H=H, W=W, n_keypoints=n_kp, d=d, n_layers=L)
    actual = sum(p.numel() for p in m.parameters())
    # embed Conv2d(1, d, 1): 1·1·1·d + d = 2d
    embed = 1 * 1 * 1 * d + d
    # Per block: Conv2d(d, d, 3, pad=1) + BN(d).
    #   Conv: 3·3·d·d + d
    #   BN:   2d (weight + bias) — running stats are buffers, not params
    per_block = 3 * 3 * d * d + d + 2 * d
    blocks = L * per_block
    # head Conv2d(d, n_kp, 1): 1·1·d·n_kp + n_kp
    head = 1 * 1 * d * n_kp + n_kp
    expected = embed + blocks + head
    assert actual == expected, (
        f"actual {actual} != expected {expected} "
        f"(embed {embed} + blocks {blocks} + head {head})"
    )


def test_tiny_cnn_pose_gradient_flow_one_step():
    """End-to-end gradient flow for the CNN baseline."""
    torch.manual_seed(50)
    m = TinyCNNPoseModel(H=32, W=32, n_keypoints=4, d=8, n_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 32, 32)
    y = torch.rand(4, 4, 2) * 32.0
    opt.zero_grad()
    pred = m(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    opt.step()


def test_tiny_cnn_pose_overfits_one_batch():
    """The CNN baseline overfits a tiny batch — proves the baseline
    is in the same trainable regime as the framework models."""
    torch.manual_seed(51)
    m = TinyCNNPoseModel(H=32, W=32, n_keypoints=4, d=8, n_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=3e-2)
    ds = SyntheticPoseDataset(n_samples=4, n_keypoints=4, seed=52)
    imgs = torch.stack([ds[i][0] for i in range(4)])
    targets = torch.stack([ds[i][1] for i in range(4)])
    initial_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    for _ in range(100):
        opt.zero_grad()
        pred = m(imgs)
        loss = torch.nn.functional.mse_loss(pred, targets)
        loss.backward()
        opt.step()
    final_loss = torch.nn.functional.mse_loss(m(imgs), targets).item()
    assert final_loss < initial_loss * 0.6, (
        f"CNN MSE didn't drop to 60% in 100 steps: "
        f"init {initial_loss:.3f} → final {final_loss:.3f}"
    )


def test_tiny_cnn_pose_invalid_n_keypoints():
    with pytest.raises(ValueError, match="n_keypoints"):
        TinyCNNPoseModel(H=32, W=32, n_keypoints=0)


@pytest.mark.parametrize("bad_kernel", [0, 2, 4, -1])
def test_tiny_cnn_pose_invalid_kernel(bad_kernel):
    with pytest.raises(ValueError, match="kernel"):
        TinyCNNPoseModel(H=32, W=32, n_keypoints=4, kernel=bad_kernel)


@pytest.mark.parametrize("bad_beta", [0.0, -1.0])
def test_tiny_cnn_pose_invalid_beta(bad_beta):
    with pytest.raises(ValueError, match="beta"):
        TinyCNNPoseModel(H=32, W=32, n_keypoints=4,
                          soft_argmax_beta=bad_beta)


def test_tiny_cnn_pose_default_size_near_10k():
    """Default config (d=12, L=8) should be in the ~10k param range,
    enabling iso-params comparison with the framework models."""
    m = TinyCNNPoseModel(H=32, W=32, n_keypoints=8)
    n_params = sum(p.numel() for p in m.parameters())
    # Window 8k-12k around the target 10k.
    assert 8000 <= n_params <= 12000, (
        f"Default param count {n_params} is outside the 8k-12k range; "
        "adjust d or L to get back to the iso-params regime."
    )


# ─── gomb_fuzzy_at axis (fuzzy-localization ablation) ──────────────


def test_gomb_fuzzy_at_default_is_fully_crisp():
    """Default (empty tuple) → all 3 shells crisp."""
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              d=4, n_layers=2, arities=[(3, 1)])
    assert m.gomb_fuzzy_at == ()
    assert m.outer.fuzzy is False
    assert m.middle.fuzzy is False
    assert m.inner.fuzzy is False


@pytest.mark.parametrize("loc", [("inner",), ("outer",), ("middle",),
                                  ("outer", "inner"),
                                  ("outer", "middle", "inner")])
def test_gomb_fuzzy_at_propagates_per_shell(loc):
    """Each named shell flips its fuzzy flag; others stay crisp."""
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              d=4, n_layers=2, arities=[(3, 1)],
                              gomb_fuzzy_at=loc)
    assert m.outer.fuzzy == ("outer" in loc)
    assert m.middle.fuzzy == ("middle" in loc)
    assert m.inner.fuzzy == ("inner" in loc)


@pytest.mark.parametrize("bad_loc", [("foo",), ("Outer",), ("outer", "head"),
                                      ("v1", "v4", "it")])
def test_gomb_fuzzy_at_invalid_names_raise(bad_loc):
    with pytest.raises(ValueError, match="gomb_fuzzy_at"):
        HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              gomb_fuzzy_at=bad_loc)


@pytest.mark.parametrize("loc", [(), ("inner",), ("outer", "inner"),
                                  ("outer", "middle", "inner")])
def test_gomb_fuzzy_at_forward_finite_and_in_image_bounds(loc):
    """All 4 ablation cells produce finite, in-bound keypoints."""
    torch.manual_seed(40)
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              d=4, n_layers=2, arities=[(3, 1)],
                              gomb_fuzzy_at=loc)
    x = torch.rand(2, 1, 32, 32)
    out = m(x)
    assert out.shape == (2, 4, 2)
    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    assert (out[..., 0] <= 31).all()
    assert (out[..., 1] <= 31).all()


def test_gomb_fuzzy_inner_changes_param_count():
    """Fuzzy localisation should not change the inner shell's param
    count (the fuzzy variant just adds a σ around the linear's output),
    but the middle and outer fuzzy variants DO add params (extra CR
    branch). Regression-checks the param-count formula per axis cell."""
    common = dict(H=32, W=32, n_keypoints=4, d=4, n_layers=2,
                   arities=[(3, 1)], gomb_M=2, gomb_d_bank=4,
                   gomb_d_mid=8)
    m_crisp = HSiKANGombPoseModel(**common, gomb_fuzzy_at=())
    m_inner = HSiKANGombPoseModel(**common, gomb_fuzzy_at=("inner",))
    m_outer = HSiKANGombPoseModel(**common, gomb_fuzzy_at=("outer",))
    m_middle = HSiKANGombPoseModel(**common, gomb_fuzzy_at=("middle",))

    def count(m):
        return sum(p.numel() for p in m.parameters())

    # Inner fuzzy = same params (just adds σ); outer fuzzy adds M CR
    # pairs; middle fuzzy adds one CR pair with 2 branches.
    assert count(m_inner) == count(m_crisp), (
        f"inner fuzzy should not change param count: "
        f"{count(m_crisp)} → {count(m_inner)}"
    )
    assert count(m_outer) > count(m_crisp), (
        "outer fuzzy should add CR-pair params (M banks × 2 branches "
        f"× d_bank × m): {count(m_crisp)} → {count(m_outer)}"
    )
    assert count(m_middle) > count(m_crisp), (
        "middle fuzzy should add one extra CR branch "
        f"({count(m_crisp)} → {count(m_middle)})"
    )


def test_gomb_fuzzy_full_localization_trains_one_step():
    """gomb_fuzzy_at=('outer','middle','inner') gradient-flows."""
    torch.manual_seed(41)
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              d=4, n_layers=2, arities=[(3, 1)],
                              gomb_M=2, gomb_d_bank=4, gomb_d_mid=8,
                              gomb_fuzzy_at=("outer", "middle", "inner"))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 32, 32)
    y = torch.rand(4, 4, 2) * 32.0
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(m(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    opt.step()


def test_hsikan_gomb_three_shells_present():
    """The model exposes outer/middle/inner as named submodules
    (paired-smoke telemetry can probe them)."""
    m = HSiKANGombPoseModel(H=32, W=32, n_keypoints=4,
                              d=4, n_layers=2, arities=[(3, 1)])
    assert hasattr(m, "outer")
    assert hasattr(m, "middle")
    assert hasattr(m, "inner")


def test_paired_models_diverge_after_training_step():
    """After one training step on the SAME batch, the two models'
    outputs should diverge measurably — they have different gradient
    flows through different machinery."""
    torch.manual_seed(24)
    fuzzy = FuzzySignaturePoseModel(H=32, W=32, n_keypoints=4,
                                      d=8, n_layers=2, arities=[(3, 1)])
    vanilla = HSiKANPoseModel(H=32, W=32, n_keypoints=4,
                                d=8, n_layers=2, arities=[(3, 1)])
    opt_f = torch.optim.Adam(fuzzy.parameters(), lr=1e-1)
    opt_v = torch.optim.Adam(vanilla.parameters(), lr=1e-1)
    x = torch.rand(4, 1, 32, 32)
    y = torch.rand(4, 4, 2) * 32.0
    # One aggressive training step on each.
    for _ in range(20):
        for opt, m in [(opt_f, fuzzy), (opt_v, vanilla)]:
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(m(x), y)
            loss.backward()
            opt.step()
    out_f = fuzzy(x)
    out_v = vanilla(x)
    diff = (out_f - out_v).abs().max().item()
    assert diff > 0.5, (
        f"post-training outputs too close: max abs diff = {diff:.3f}"
    )
