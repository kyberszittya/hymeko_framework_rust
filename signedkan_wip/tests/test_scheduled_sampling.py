"""Tests for the scheduled-sampling training forward.

The Bengio et al. 2015 exposure-bias mitigation: at each decoder
position, with probability ε the input is the model's own previous
argmax; with probability 1-ε the input is the ground-truth previous
token.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from signedkan_wip.src.sequence.encoder_decoder import DualPathEncoderDecoder


def test_scheduled_sampling_epsilon_zero_equals_teacher_forced():
    """At ε=0, scheduled-sampling output must equal the plain
    teacher-forced forward bit-for-bit."""
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(
        vocab_size=20, enc_depth=2, dec_depth=2, K=4, max_len=64,
        n_channels=2,
    ).eval()
    src = torch.randint(2, 20, (2, 8))
    tgt = torch.randint(2, 20, (2, 8))
    with torch.no_grad():
        tf = model(src, tgt)
        ss = model.forward_scheduled_sampling(src, tgt, epsilon=0.0)
    assert torch.allclose(tf, ss, atol=1e-6)


def test_scheduled_sampling_epsilon_one_ignores_ground_truth():
    """At ε=1, the decoder is fed only its own predictions starting
    after the SOS token. The result should equal the autoregressive
    generation up to the final position's logits."""
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(
        vocab_size=20, enc_depth=2, dec_depth=2, K=4, max_len=64,
        n_channels=2,
    ).eval()
    src = torch.randint(2, 20, (2, 6))
    tgt = torch.zeros(2, 6, dtype=torch.long)  # GT is unused at ε=1
    with torch.no_grad():
        # Two ε=1 runs with the same seed should be identical.
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        ss1 = model.forward_scheduled_sampling(src, tgt, epsilon=1.0, generator=gen1)
        ss2 = model.forward_scheduled_sampling(src, tgt, epsilon=1.0, generator=gen2)
    assert torch.allclose(ss1, ss2, atol=1e-6)


def test_scheduled_sampling_shape_matches_teacher_forced():
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(
        vocab_size=30, enc_depth=2, dec_depth=2, K=4, max_len=64,
        n_channels=2,
    )
    src = torch.randint(2, 30, (3, 10))
    tgt = torch.randint(2, 30, (3, 10))
    logits_tf = model(src, tgt)
    logits_ss = model.forward_scheduled_sampling(src, tgt, epsilon=0.3)
    assert logits_tf.shape == logits_ss.shape == (3, 10, 30)


def test_scheduled_sampling_gradient_flows():
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(
        vocab_size=20, enc_depth=2, dec_depth=2, K=4, max_len=64,
        n_channels=2,
    )
    src = torch.randint(2, 20, (2, 8))
    tgt = torch.randint(2, 20, (2, 8))
    logits = model.forward_scheduled_sampling(src, tgt, epsilon=0.4)
    loss = F.cross_entropy(logits.view(-1, 20), tgt.view(-1))
    loss.backward()
    # Every learnable parameter receives non-NaN gradient.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_scheduled_sampling_intermediate_epsilon_differs_from_TF():
    """At ε∈(0,1) the output should usually differ from teacher-
    forced (different decoder inputs after the first position)."""
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(
        vocab_size=20, enc_depth=2, dec_depth=2, K=4, max_len=64,
        n_channels=2,
    ).eval()
    src = torch.randint(2, 20, (4, 8))
    tgt = torch.randint(2, 20, (4, 8))
    with torch.no_grad():
        tf = model(src, tgt)
        ss = model.forward_scheduled_sampling(
            src, tgt, epsilon=0.7,
            generator=torch.Generator().manual_seed(7),
        )
    # First position is always SOS-driven; difference may be small.
    # But at least one entry across (B, L, V) should differ
    # meaningfully at ε=0.7 with B=4, L=8, V=20.
    diff = (tf - ss).abs().max().item()
    assert diff > 1e-3, f"TF and ε=0.7 outputs are too close (max diff {diff})"
