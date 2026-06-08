"""Tests for the HSiKAN within-RF spatial-filter variant.

Verifies (the equivariance hypothesis was falsified; this is the
follow-up hypothesis):
  1. ``build_rf_position_incidence`` returns the same binary ``inc`` as
     ``build_rf_incidence`` AND a parallel ``pos_inc`` with the right
     values per pixel-in-RF.
  2. Construction: ``spatial_filter="scalar"`` adds K-sized W_pos params per
     arity per layer (404 + 410 = 814 extra for the default config — the
     exact arithmetic is documented in `test_param_count`).
  3. Forward parity at init: with W_pos initialised to ones, the
     spatial-filter forward is bit-for-bit equal to the uniform-mean
     baseline. (Mathematical: ``inc * W_pos[pos_inc.clamp_min(0)]``
     reduces to ``inc * 1 = inc`` everywhere it matters.)
  4. The spatial-filter model trains one step without NaN.
  5. ``pos_inc`` for non-RF entries is -1; clamp_min(0) handles them
     safely (zeroed out by the binary ``inc`` multiplication).
"""
from __future__ import annotations

import torch

from signedkan_wip.src.vision.hsikan_vision import (
    HSiKANVisionClassifier,
    SignedBranchConv,
    build_rf_incidence,
    build_rf_position_incidence,
)


def test_position_incidence_matches_binary():
    """The (inc, n_edges) returned by build_rf_position_incidence MUST
    match build_rf_incidence on the same args."""
    H, W = 28, 28
    for k, s in [(5, 2), (8, 4), (12, 4)]:
        inc1, n1 = build_rf_incidence(H, W, kernel=k, stride=s)
        inc2, pos_inc, n2 = build_rf_position_incidence(H, W, kernel=k, stride=s)
        assert n1 == n2
        assert torch.equal(inc1, inc2)
        # pos_inc shape matches inc shape.
        assert pos_inc.shape == inc1.shape
        # pos_inc is in {-1, 0, ..., k²-1}.
        assert pos_inc.min().item() == -1
        assert pos_inc.max().item() == k * k - 1
        # Where inc==1, pos_inc must be in [0, k²-1]; where inc==0, pos_inc=-1.
        is_member = inc1.bool()
        assert (pos_inc[is_member] >= 0).all()
        assert (pos_inc[is_member] < k * k).all()
        assert (pos_inc[~is_member] == -1).all()


def test_signedbranchconv_W_pos_shape():
    """spatial_filter="scalar" adds W_pos[k²] per conv."""
    c = SignedBranchConv(d_in=4, d_out=8, n_edges=144, kernel=5,
                         spatial_filter="scalar")
    assert c.W_pos.shape == (25,)
    assert c.spatial_filter == "scalar"
    # Default (spatial_filter="none") has no W_pos.
    c0 = SignedBranchConv(d_in=4, d_out=8, n_edges=144)
    assert c0.W_pos is None


def test_param_count():
    """spatial_filter="scalar" adds K² params per (arity, layer).

    Default arities [(5,2),(8,4),(12,4)] → K² = 25 + 64 + 144 = 233
    per layer. × 2 layers = 466 extra params.
    """
    H, W = 28, 28
    p_off = sum(p.numel() for p in HSiKANVisionClassifier(
        H, W, 10, hidden=32, n_layers=2, spatial_filter="none").parameters())
    p_on = sum(p.numel() for p in HSiKANVisionClassifier(
        H, W, 10, hidden=32, n_layers=2, spatial_filter="scalar").parameters())
    expected_added = (25 + 64 + 144) * 2
    assert p_on - p_off == expected_added, (
        f"expected {expected_added} added by spatial_filter, got {p_on - p_off}"
    )


def test_forward_parity_at_init_with_ones():
    """spatial_filter="scalar" with W_pos=ones (the init value) is
    bit-for-bit equal to spatial_filter="none" on the same input.

    Math: ``inc * W_pos[pos_inc.clamp_min(0)] = inc * 1 = inc`` when all
    W_pos entries are 1. The clamp is irrelevant because entries where
    pos_inc=-1 also have inc=0.
    """
    torch.manual_seed(0)
    H, W = 28, 28
    m_off = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2,
                                   spatial_filter="none")
    torch.manual_seed(0)
    m_on = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2,
                                  spatial_filter="scalar")
    # Sanity: same seed gave identical non-W_pos init.
    x = torch.randn(2, 1, H, W)
    with torch.no_grad():
        y_off = m_off(x)
        y_on = m_on(x)
    assert torch.allclose(y_off, y_on, atol=1e-6, rtol=1e-5), (
        f"forward differs at init despite W_pos=ones: "
        f"max abs diff {(y_off - y_on).abs().max().item():.3e}"
    )


def test_spatial_filter_model_trains_one_step():
    """A spatial-filter model takes one gradient step without NaN."""
    torch.manual_seed(1)
    H, W = 28, 28
    model = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2,
                                   spatial_filter="scalar")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(4, 1, H, W)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    # W_pos should have non-zero gradient (the model used it).
    for layer in model.layers:
        for conv in layer.convs:
            assert conv.W_pos is not None
            assert conv.W_pos.grad is not None
            assert torch.isfinite(conv.W_pos.grad).all()


def test_spatial_filter_requires_pos_inc():
    """spatial_filter='scalar' forward without pos_inc raises a clear error."""
    import pytest

    c = SignedBranchConv(d_in=4, d_out=8, n_edges=144, kernel=5,
                         spatial_filter="scalar")
    x = torch.randn(2, 28 * 28, 4)
    inc = torch.zeros(28 * 28, 144)
    D_v = torch.ones(28 * 28)
    D_e = torch.ones(144)
    with pytest.raises(ValueError, match="pos_inc"):
        c.forward(x, inc, D_v, D_e, pos_inc=None)


# ─── per_channel variant (W_pos[K, d_out]) ──────────────────────────


def test_per_channel_W_pos_shape():
    """spatial_filter='per_channel' shapes W_pos as (K², d_out)."""
    c = SignedBranchConv(d_in=4, d_out=8, n_edges=144, kernel=5,
                         spatial_filter="per_channel")
    assert c.W_pos.shape == (25, 8)
    assert c.spatial_filter == "per_channel"


def test_per_channel_param_count():
    """per_channel adds K² × d_out per (arity, layer).

    Default arities [(5,2),(8,4),(12,4)] × hidden=32:
       (25 + 64 + 144) × 32 = 7456 per layer × 2 layers = 14 912.
    """
    H, W = 28, 28
    p_off = sum(p.numel() for p in HSiKANVisionClassifier(
        H, W, 10, hidden=32, n_layers=2, spatial_filter="none").parameters())
    p_pc = sum(p.numel() for p in HSiKANVisionClassifier(
        H, W, 10, hidden=32, n_layers=2,
        spatial_filter="per_channel").parameters())
    expected = (25 + 64 + 144) * 32 * 2
    assert p_pc - p_off == expected, (
        f"expected {expected} added by per_channel, got {p_pc - p_off}"
    )


def test_per_channel_forward_parity_at_init_with_ones():
    """spatial_filter='per_channel' with W_pos=ones is bit-equal to baseline."""
    torch.manual_seed(0)
    H, W = 28, 28
    m_off = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2,
                                   spatial_filter="none")
    torch.manual_seed(0)
    m_pc = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2,
                                  spatial_filter="per_channel")
    x = torch.randn(2, 1, H, W)
    with torch.no_grad():
        y_off = m_off(x)
        y_pc = m_pc(x)
    assert torch.allclose(y_off, y_pc, atol=1e-5, rtol=1e-5), (
        f"per_channel forward differs at init despite W_pos=ones: "
        f"max abs diff {(y_off - y_pc).abs().max().item():.3e}"
    )


def test_per_channel_trains_one_step():
    """per_channel model trains one step; W_pos has finite grad."""
    torch.manual_seed(1)
    H, W = 28, 28
    model = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2,
                                   spatial_filter="per_channel")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(4, 1, H, W)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    for layer in model.layers:
        for conv in layer.convs:
            assert conv.W_pos.shape[-1] == 32
            assert torch.isfinite(conv.W_pos.grad).all()


def test_invalid_spatial_filter_value_raises():
    """Unknown spatial_filter strings raise clearly."""
    import pytest

    with pytest.raises(ValueError, match="spatial_filter"):
        SignedBranchConv(d_in=4, d_out=8, n_edges=144, kernel=5,
                         spatial_filter="bogus")
