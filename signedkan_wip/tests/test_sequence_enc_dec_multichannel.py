"""Tests for the v2 multi-channel DualPathEncoderDecoder.

Plan: docs/plans/2026-05-17-sequence-multichannel-v2/.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from signedkan_wip.src.sequence.clifford import CL2_DIM
from signedkan_wip.src.sequence.cross_dual_path import CrossDualPathBlock
from signedkan_wip.src.sequence.dual_router import DualPathSeqBlock, PositionRouter
from signedkan_wip.src.sequence.encoder_decoder import DualPathEncoderDecoder


# ─── DualPathSeqBlock multi-channel ─────────────────────────────────


@pytest.mark.parametrize("C", [1, 2, 4, 8])
def test_dual_path_seq_block_C_shape(C):
    block = DualPathSeqBlock(K=4, n_channels=C)
    if C == 1:
        x = torch.randn(2, 16, CL2_DIM)
    else:
        x = torch.randn(2, 16, C, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 16)).float()
    y, sigma_out = block(x, sigma)
    assert y.shape == x.shape
    assert torch.equal(sigma_out, sigma)


def test_dual_path_seq_block_C4_param_count_grows():
    n1 = sum(p.numel() for p in DualPathSeqBlock(K=4, n_channels=1).parameters())
    n4 = sum(p.numel() for p in DualPathSeqBlock(K=4, n_channels=4).parameters())
    n8 = sum(p.numel() for p in DualPathSeqBlock(K=4, n_channels=8).parameters())
    assert n4 > n1
    assert n8 > n4


# ─── PositionRouter multi-channel input ─────────────────────────────


def test_position_router_accepts_C4_input():
    router = PositionRouter()
    x = torch.randn(2, 16, 4, CL2_DIM)
    sigma = torch.randint(-1, 2, (2, 16)).float()
    g = router(x, sigma)
    assert g.shape == (2, 16)
    assert ((0.0 < g) & (g < 1.0)).all()


# ─── CrossDualPathBlock multi-channel ───────────────────────────────


@pytest.mark.parametrize("C", [1, 2, 4, 8])
def test_cross_dual_path_C_shape(C):
    block = CrossDualPathBlock(n_channels=C)
    if C == 1:
        q = torch.randn(2, 7, CL2_DIM)
        k = torch.randn(2, 11, CL2_DIM)
    else:
        q = torch.randn(2, 7, C, CL2_DIM)
        k = torch.randn(2, 11, C, CL2_DIM)
    sigma_q = torch.randint(-1, 2, (2, 7)).float()
    sigma_k = torch.randint(-1, 2, (2, 11)).float()
    y = block(q, k, sigma_q, sigma_k)
    assert y.shape == q.shape


def test_cross_dual_path_C_gradient_flows():
    torch.manual_seed(0)
    block = CrossDualPathBlock(n_channels=4)
    q = torch.randn(2, 7, 4, CL2_DIM, requires_grad=True)
    k = torch.randn(2, 11, 4, CL2_DIM, requires_grad=True)
    sigma_q = torch.randint(-1, 2, (2, 7)).float()
    sigma_k = torch.randint(-1, 2, (2, 11)).float()
    y = block(q, k, sigma_q, sigma_k)
    y.pow(2).mean().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# ─── End-to-end DualPathEncoderDecoder multi-channel ────────────────


@pytest.mark.parametrize("C", [1, 4, 8])
def test_encoder_decoder_C_forward_shape(C):
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(
        vocab_size=50, enc_depth=2, dec_depth=2, K=4, max_len=64,
        n_channels=C,
    )
    src = torch.randint(0, 50, (2, 16))
    tgt = torch.randint(0, 50, (2, 16))
    logits = model(src, tgt)
    assert logits.shape == (2, 16, 50)
    assert torch.isfinite(logits).all()


def test_encoder_decoder_C8_gradient_flows_end_to_end():
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(
        vocab_size=50, enc_depth=2, dec_depth=2, K=4, max_len=64,
        n_channels=8,
    )
    src = torch.randint(0, 50, (2, 16))
    tgt = torch.randint(0, 50, (2, 16))
    logits = model(src, tgt)
    loss = F.cross_entropy(logits.view(-1, 50), tgt.view(-1))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_encoder_decoder_param_count_grows_with_C():
    n1 = sum(p.numel() for p in DualPathEncoderDecoder(
        vocab_size=50, n_channels=1, max_len=64).parameters())
    n8 = sum(p.numel() for p in DualPathEncoderDecoder(
        vocab_size=50, n_channels=8, max_len=64).parameters())
    # At V=50, C=1 has ~few hundred params; C=8 has substantially more
    # (dominated by embed: 50 * 8 * 4 = 1600 vs 50 * 1 * 4 = 200).
    assert n8 > 3 * n1


def test_encoder_decoder_C8_generate_runs():
    """Greedy decode at C=8 produces valid token IDs."""
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(
        vocab_size=20, enc_depth=2, dec_depth=2, K=4, max_len=64,
        n_channels=8,
    )
    src = torch.randint(2, 20, (2, 10))
    out = model.generate(src, max_new_tokens=8, sos_token_id=1)
    assert out.shape == (2, 8)
    assert ((0 <= out) & (out < 20)).all()
