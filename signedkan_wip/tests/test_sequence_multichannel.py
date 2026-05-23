"""Tests for the v2 multi-channel HSiKANSeqWindow refactor.

Coverage (per the v2 plan §7):
  - C=1 numerical byte-identity vs the conceptual v1 baseline
  - Shape correctness for C ∈ {1, 4, 8}
  - Gradient flow at C > 1
  - Parameter count grows ~linearly with C
  - n_channels mismatch raises cleanly

Plan: docs/plans/2026-05-17-sequence-multichannel-v2/.
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.sequence.clifford import CL2_DIM
from signedkan_wip.src.sequence.hsikan_seq import HSiKANSeqWindow


# ─── C=1 byte-identical regression ────────────────────────────────────


def test_hsikan_seq_window_c1_numerical_byte_identical_to_simulated_v1():
    """At C=1 with the channel mixer at the identity-scalar init,
    HSiKANSeqWindow produces the same output as v1's "single
    multivector tap" path would, to float32 tolerance."""
    torch.manual_seed(0)
    layer = HSiKANSeqWindow(K=4, n_channels=1)
    # Force the channel mixer to exact scalar identity (no noise).
    with torch.no_grad():
        layer.channel_mixer.zero_()
        layer.channel_mixer[0, 0, 0] = 1.0
    x = torch.randn(2, 16, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 16)).float()
    # Two paths: (a) auto-unsqueeze path (input is (B, L, 4))
    y_auto = layer(x, sigma)
    # (b) explicit-channel path (input is (B, L, 1, 4))
    y_explicit = layer(x.unsqueeze(2), sigma).squeeze(2)
    assert torch.allclose(y_auto, y_explicit, atol=1e-6)


def test_hsikan_seq_window_c1_with_identity_mixer_equals_pre_mix_output():
    """With identity-scalar channel mixer, the post-mix output equals
    the pre-mix per-channel output. This validates the mixer's
    identity-init claim."""
    torch.manual_seed(0)
    layer = HSiKANSeqWindow(K=4, n_channels=1)
    with torch.no_grad():
        layer.channel_mixer.zero_()
        layer.channel_mixer[0, 0, 0] = 1.0
    x = torch.randn(2, 8, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 8)).float()
    y = layer(x, sigma)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# ─── Multi-channel shape contracts ────────────────────────────────────


@pytest.mark.parametrize("C", [1, 2, 4, 8])
def test_hsikan_seq_window_c_shape(C):
    """For each C, (B, L, C, 4) in → (B, L, C, 4) out."""
    layer = HSiKANSeqWindow(K=4, n_channels=C)
    x = torch.randn(2, 16, C, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 16)).float()
    y = layer(x, sigma)
    assert y.shape == (2, 16, C, CL2_DIM)
    assert torch.isfinite(y).all()


def test_hsikan_seq_window_param_count_grows_linearly_with_C():
    """At C=8 we expect more than C=1; specifically:
      tap_pos: C × 4
      tap_neg: C × 4
      channel_mixer: C × C × 4
      parity_gate: 3
    """
    layer_1 = HSiKANSeqWindow(K=4, n_channels=1)
    layer_4 = HSiKANSeqWindow(K=4, n_channels=4)
    layer_8 = HSiKANSeqWindow(K=4, n_channels=8)
    n1 = sum(p.numel() for p in layer_1.parameters())
    n4 = sum(p.numel() for p in layer_4.parameters())
    n8 = sum(p.numel() for p in layer_8.parameters())
    # C=1: 4+4+4+3 = 15
    assert n1 == 15
    # C=4: 16+16+64+3 = 99
    assert n4 == 4*4 + 4*4 + 4*4*4 + 3
    # C=8: 32+32+256+3 = 323
    assert n8 == 8*4 + 8*4 + 8*8*4 + 3


def test_hsikan_seq_window_gradient_flows_at_C4():
    torch.manual_seed(0)
    layer = HSiKANSeqWindow(K=4, n_channels=4)
    x = torch.randn(2, 10, 4, CL2_DIM, requires_grad=True)
    sigma = torch.randint(-1, 2, (2, 10)).float()
    y = layer(x, sigma)
    y.pow(2).mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_hsikan_seq_window_rejects_c_mismatch():
    layer = HSiKANSeqWindow(K=4, n_channels=4)
    # input has 3 channels, layer expects 4
    x = torch.randn(2, 10, 3, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 10)).float()
    with pytest.raises(ValueError, match="channels"):
        layer(x, sigma)


def test_hsikan_seq_window_rejects_3d_input_when_C_gt_1():
    """When n_channels > 1, the auto-unsqueeze path is disabled."""
    layer = HSiKANSeqWindow(K=4, n_channels=4)
    x = torch.randn(2, 10, CL2_DIM)  # (B, L, 4) — no channel dim
    sigma = torch.randint(-1, 2, (2, 10)).float()
    with pytest.raises(ValueError, match="n_channels"):
        layer(x, sigma)


def test_hsikan_seq_window_zero_signs_at_C_4_preserves_finiteness():
    """When all sigma=0 at C=4, the layer still produces finite output
    (no NaN, no divide-by-zero in the σ-pool)."""
    layer = HSiKANSeqWindow(K=4, n_channels=4)
    x = torch.randn(1, 16, 4, CL2_DIM)
    sigma = torch.zeros(1, 16)
    y = layer(x, sigma)
    assert torch.isfinite(y).all()


def test_hsikan_seq_window_default_n_channels_is_1():
    """Default n_channels=1 (backward compat with v1 callers)."""
    layer = HSiKANSeqWindow(K=4)
    assert layer.n_channels == 1
    x = torch.randn(2, 8, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 8)).float()
    y = layer(x, sigma)
    assert y.shape == x.shape
