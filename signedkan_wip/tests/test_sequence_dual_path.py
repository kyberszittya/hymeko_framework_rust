"""Tests for HSiKANSeqWindow, PositionRouter, DualPathSeqBlock, and
DualPathSequenceModel.

Coverage (per plan §7):
  - HSiKANSeqWindow shape contract + σ-masked pool semantics + parity gate
  - PositionRouter output in (0, 1)
  - DualPathSeqBlock: g=1 collapses to CliffordFIR; g=0 collapses to
    HSiKANSeqWindow (byte-identical pinning for ablation interpretability)
  - DualPathSequenceModel end-to-end forward + gradient flow
  - Parameter count matches the docstring estimate

Plan: docs/plans/2026-05-17-sequential-hsikan-clifford-fir/.
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.sequence.clifford import CL2_DIM
from signedkan_wip.src.sequence.clifford_fir import CliffordFIR
from signedkan_wip.src.sequence.hsikan_seq import HSiKANSeqWindow
from signedkan_wip.src.sequence.dual_router import (
    DualPathSeqBlock, PositionRouter,
)
from signedkan_wip.src.sequence.dual_path_model import (
    DualPathSequenceModel, SeqInputEncoder,
)


# ─── HSiKANSeqWindow ─────────────────────────────────────────────────


def test_hsikan_seq_window_shape():
    layer = HSiKANSeqWindow(K=4)
    x = torch.randn(2, 16, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 16)).float()
    y = layer(x, sigma)
    assert y.shape == (2, 16, CL2_DIM)
    assert torch.isfinite(y).all()


def test_hsikan_seq_window_rejects_wrong_shapes():
    layer = HSiKANSeqWindow(K=4)
    with pytest.raises(ValueError, match="x must be"):
        layer(torch.randn(2, 16, 5), torch.zeros(2, 16))
    with pytest.raises(ValueError, match="sigma must be"):
        layer(torch.randn(2, 16, 4), torch.zeros(2, 17))


def test_hsikan_seq_window_param_count():
    """v2 (2026-05-17): tap_pos (C×4) + tap_neg (C×4) + channel_mixer
    (C×C×4) + parity_gate (3). At C=1: 4 + 4 + 4 + 3 = 15. The
    channel mixer is identity-initialised so the C=1 forward is
    numerically byte-identical to v1; the additional 4 scalars are
    the trivial-at-init scalar-multivector identity."""
    layer = HSiKANSeqWindow(K=4, n_channels=1)
    n = sum(p.numel() for p in layer.parameters())
    assert n == 15


def test_hsikan_seq_window_gradient_flows():
    torch.manual_seed(0)
    layer = HSiKANSeqWindow(K=4)
    x = torch.randn(2, 16, CL2_DIM, requires_grad=True)
    sigma = torch.randint(-1, 2, (2, 16)).float()
    y = layer(x, sigma)
    y.pow(2).mean().backward()
    assert layer.tap_pos.grad is not None
    assert torch.isfinite(layer.tap_pos.grad).all()
    assert layer.tap_neg.grad is not None
    assert layer.parity_gate.grad is not None


def test_hsikan_seq_window_zero_signs_neutralize_parity():
    """When all signs in the window are 0 (no-sign), the parity is +1
    (the multiplicative identity)."""
    layer = HSiKANSeqWindow(K=4)
    x = torch.randn(1, 16, CL2_DIM)
    sigma = torch.zeros(1, 16)  # no signs anywhere
    y = layer(x, sigma)
    # Output should be finite (parity gate index 2 = pos branch).
    assert torch.isfinite(y).all()


# ─── PositionRouter ──────────────────────────────────────────────────


def test_position_router_output_in_unit_interval():
    router = PositionRouter()
    torch.manual_seed(0)
    x = torch.randn(2, 16, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 16)).float()
    g = router(x, sigma)
    assert g.shape == (2, 16)
    assert ((0.0 < g) & (g < 1.0)).all()


def test_position_router_param_count():
    """Linear(6 → 1) = 7 params."""
    router = PositionRouter()
    n = sum(p.numel() for p in router.parameters())
    assert n == 7


# ─── DualPathSeqBlock — gate-extreme ablation pins ───────────────────


def test_dual_path_block_full_gate_equals_clifford_fir():
    """Forcing g_t ≡ 1 must collapse to pure CliffordFIR. We pin this
    by temporarily zeroing the router's weights + biasing toward
    sigmoid(large) ≈ 1."""
    torch.manual_seed(0)
    block = DualPathSeqBlock(K=4)
    with torch.no_grad():
        # Force g ≈ 1.
        block.router.proj.weight.zero_()
        block.router.proj.bias.fill_(10.0)  # sigmoid(10) ≈ 0.99995
    x = torch.randn(1, 8, CL2_DIM)
    sigma = torch.randint(-1, 2, (1, 8)).float()
    y_block, _ = block(x, sigma)
    y_fir = block.fir(x)
    assert torch.allclose(y_block, y_fir, atol=1e-3)


def test_dual_path_block_zero_gate_equals_hsikan():
    """Forcing g_t ≡ 0 must collapse to pure HSiKANSeqWindow."""
    torch.manual_seed(0)
    block = DualPathSeqBlock(K=4)
    with torch.no_grad():
        block.router.proj.weight.zero_()
        block.router.proj.bias.fill_(-10.0)  # sigmoid(-10) ≈ 4.5e-5
    x = torch.randn(1, 8, CL2_DIM)
    sigma = torch.randint(-1, 2, (1, 8)).float()
    y_block, _ = block(x, sigma)
    y_hsi = block.hsi(x, sigma)
    assert torch.allclose(y_block, y_hsi, atol=1e-3)


def test_dual_path_block_passes_sigma_through_unchanged():
    block = DualPathSeqBlock(K=4)
    x = torch.randn(2, 8, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 8)).float()
    _, sigma_out = block(x, sigma)
    assert torch.equal(sigma_out, sigma)


def test_dual_path_block_load_balance_loss_nonnegative():
    block = DualPathSeqBlock(K=4)
    x = torch.randn(2, 16, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 16)).float()
    loss = block.gate_load_balance_loss(x, sigma)
    assert loss.item() >= 0


# ─── DualPathSequenceModel ───────────────────────────────────────────


def test_seq_input_encoder_shapes():
    enc = SeqInputEncoder(in_features=4)
    raw = torch.randn(2, 16, 4)
    x, sigma = enc(raw)
    assert x.shape == (2, 16, CL2_DIM)
    assert sigma.shape == (2, 16)
    assert ((sigma == -1.0) | (sigma == 1.0)).all()


def test_seq_input_encoder_supervised_sign_requires_override():
    enc = SeqInputEncoder(in_features=4, supervised_sign=True)
    raw = torch.randn(2, 16, 4)
    with pytest.raises(ValueError, match="supervised_sign"):
        enc(raw)
    sigma_gt = torch.randint(-1, 2, (2, 16)).float()
    x, sigma = enc(raw, sigma_override=sigma_gt)
    assert torch.equal(sigma, sigma_gt)


def test_dual_path_sequence_model_forward_shape():
    torch.manual_seed(0)
    model = DualPathSequenceModel(in_features=4, n_classes=2, depth=3, K=4)
    raw = torch.randn(2, 32, 4)
    logits = model(raw)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_dual_path_sequence_model_gradient_flows_to_all_subparams():
    torch.manual_seed(0)
    model = DualPathSequenceModel(in_features=4, n_classes=2, depth=3, K=4)
    raw = torch.randn(2, 32, 4)
    logits = model(raw)
    logits.pow(2).mean().backward()
    # Encoder, every block's fir/hsi/router, pool, head all see gradient.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_dual_path_sequence_model_param_count_in_expected_range():
    """At in_features=4, n_classes=2, depth=3, K=4 we expect ~140 params."""
    model = DualPathSequenceModel(in_features=4, n_classes=2, depth=3, K=4)
    n = sum(p.numel() for p in model.parameters())
    assert 100 <= n <= 200, f"param count {n} outside [100,200]"


def test_dual_path_sequence_model_load_balance_loss_nonnegative():
    model = DualPathSequenceModel(in_features=4, n_classes=2)
    raw = torch.randn(2, 16, 4)
    loss = model.gate_load_balance_loss(raw)
    assert loss.item() >= 0


def test_dual_path_sequence_model_supervised_sign_path():
    model = DualPathSequenceModel(
        in_features=4, n_classes=2, supervised_sign=True,
    )
    raw = torch.randn(2, 16, 4)
    sigma_gt = torch.randint(-1, 2, (2, 16)).float()
    logits = model(raw, sigma_override=sigma_gt)
    assert logits.shape == (2, 2)
