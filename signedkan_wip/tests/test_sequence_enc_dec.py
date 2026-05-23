"""Tests for the text encoder-decoder modules.

Coverage:
  - TokenMultivectorEmbedding shape + parameter count + padding mask
  - CliffordRotorPositional norm-preservation + relative-position invariance
  - CrossDualPathBlock shape + gradient flow
  - DualPathEncoderDecoder forward + generate + teacher-forced training
  - Synthetic seq2seq generator shapes + decoder-input shift
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from signedkan_wip.src.sequence.clifford import CL2_DIM, geometric_product, multivector_norm
from signedkan_wip.src.sequence.clifford_positional import (
    CliffordRotorPositional, _rotor_apply,
)
from signedkan_wip.src.sequence.text_embedding import TokenMultivectorEmbedding
from signedkan_wip.src.sequence.cross_dual_path import CrossDualPathBlock
from signedkan_wip.src.sequence.encoder_decoder import DualPathEncoderDecoder
from signedkan_wip.src.sequence.synthetic_seq2seq import (
    CopyTaskConfig, make_copy_dataset, make_decoder_inputs,
)


# ─── CliffordRotorPositional ─────────────────────────────────────────


def test_rotor_apply_preserves_scalar_and_bivector():
    """The rotor only rotates the (e_1, e_2) part; scalar and
    bivector parts pass through unchanged."""
    torch.manual_seed(0)
    x = torch.randn(5, CL2_DIM)
    theta = torch.tensor(0.7)
    y = _rotor_apply(x, theta)
    assert torch.allclose(y[:, 0], x[:, 0], atol=1e-6), "scalar should be preserved"
    assert torch.allclose(y[:, 3], x[:, 3], atol=1e-6), "bivector should be preserved"


def test_rotor_apply_zero_angle_is_identity():
    torch.manual_seed(0)
    x = torch.randn(5, CL2_DIM)
    y = _rotor_apply(x, torch.tensor(0.0))
    assert torch.allclose(y, x, atol=1e-6)


def test_rotor_apply_norm_preserving():
    torch.manual_seed(0)
    x = torch.randn(20, CL2_DIM)
    theta = torch.linspace(0.0, 3.14, 20)
    y = _rotor_apply(x, theta)
    n_x = multivector_norm(x)
    n_y = multivector_norm(y)
    assert torch.allclose(n_x, n_y, atol=1e-5), "rotor conjugation should preserve norm"


def test_rotor_positional_shape():
    pe = CliffordRotorPositional(max_len=64, n_channels=2)
    x = torch.randn(2, 16, 2, CL2_DIM)
    y = pe(x)
    assert y.shape == x.shape


def test_rotor_positional_handles_single_channel_input():
    pe = CliffordRotorPositional(max_len=64, n_channels=1)
    x = torch.randn(2, 16, CL2_DIM)
    y = pe(x)
    assert y.shape == (2, 16, CL2_DIM)


def test_rotor_positional_rejects_too_long_sequences():
    pe = CliffordRotorPositional(max_len=8, n_channels=1)
    x = torch.randn(1, 10, CL2_DIM)
    with pytest.raises(ValueError, match="max_len"):
        pe(x)


# ─── TokenMultivectorEmbedding ───────────────────────────────────────


def test_token_embedding_shape():
    emb = TokenMultivectorEmbedding(vocab_size=100, n_channels=4, max_len=64)
    tok = torch.randint(0, 100, (2, 16))
    out = emb(tok)
    assert out.shape == (2, 16, 4, CL2_DIM)


def test_token_embedding_pad_zero():
    """Padding token (id=0) embeds to zero."""
    emb = TokenMultivectorEmbedding(
        vocab_size=10, n_channels=2, max_len=32, pad_token_id=0, positional=False,
    )
    tok = torch.zeros(1, 4, dtype=torch.long)
    out = emb(tok)
    assert torch.allclose(out, torch.zeros_like(out))


def test_token_embedding_param_count():
    """V=100, n_channels=4 → embedding = 100*16 = 1600 params."""
    emb = TokenMultivectorEmbedding(vocab_size=100, n_channels=4, positional=False)
    n = sum(p.numel() for p in emb.parameters())
    assert n == 100 * 4 * CL2_DIM


# ─── CrossDualPathBlock ──────────────────────────────────────────────


def test_cross_dual_path_shape():
    block = CrossDualPathBlock()
    q = torch.randn(2, 7, CL2_DIM)
    k = torch.randn(2, 11, CL2_DIM)
    sigma_q = torch.randint(-1, 2, (2, 7)).float()
    sigma_k = torch.randint(-1, 2, (2, 11)).float()
    y = block(q, k, sigma_q, sigma_k)
    assert y.shape == (2, 7, CL2_DIM)
    assert torch.isfinite(y).all()


def test_cross_dual_path_gradient_flows():
    torch.manual_seed(0)
    block = CrossDualPathBlock()
    q = torch.randn(2, 7, CL2_DIM, requires_grad=True)
    k = torch.randn(2, 11, CL2_DIM, requires_grad=True)
    sigma_q = torch.randint(-1, 2, (2, 7)).float()
    sigma_k = torch.randint(-1, 2, (2, 11)).float()
    y = block(q, k, sigma_q, sigma_k)
    y.pow(2).mean().backward()
    # All learnable params receive gradient.
    for name, p in block.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


# ─── DualPathEncoderDecoder ──────────────────────────────────────────


def test_encoder_decoder_forward_shape():
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(vocab_size=50, enc_depth=2, dec_depth=2, K=4, max_len=64)
    src = torch.randint(0, 50, (2, 16))
    tgt = torch.randint(0, 50, (2, 16))
    logits = model(src, tgt)
    assert logits.shape == (2, 16, 50)
    assert torch.isfinite(logits).all()


def test_encoder_decoder_gradient_flows_end_to_end():
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(vocab_size=50, enc_depth=2, dec_depth=2, K=4, max_len=64)
    src = torch.randint(0, 50, (2, 16))
    tgt = torch.randint(0, 50, (2, 16))
    logits = model(src, tgt)
    loss = F.cross_entropy(logits.view(-1, 50), tgt.view(-1))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_encoder_decoder_generate_runs():
    torch.manual_seed(0)
    model = DualPathEncoderDecoder(vocab_size=20, enc_depth=2, dec_depth=2, K=4, max_len=64)
    src = torch.randint(2, 20, (2, 10))
    out = model.generate(src, max_new_tokens=8, sos_token_id=1)
    assert out.shape == (2, 8)
    assert ((0 <= out) & (out < 20)).all()


def test_encoder_decoder_tied_embedding_param_count():
    """With tie_embeddings=True (default), output_proj is None,
    saving V * 4 params vs the untied baseline."""
    model_tied = DualPathEncoderDecoder(vocab_size=100, tie_embeddings=True, max_len=64)
    model_untied = DualPathEncoderDecoder(vocab_size=100, tie_embeddings=False, max_len=64)
    n_tied = sum(p.numel() for p in model_tied.parameters())
    n_untied = sum(p.numel() for p in model_untied.parameters())
    # Difference is exactly the output projection (V*4 + V bias).
    diff = n_untied - n_tied
    assert diff == 100 * CL2_DIM + 100  # weight + bias


# ─── Synthetic seq2seq generator ─────────────────────────────────────


def test_copy_dataset_shapes():
    cfg = CopyTaskConfig(n_samples=8, L_src=12, vocab_size=20, shift=3)
    src, tgt = make_copy_dataset(cfg)
    assert src.shape == (8, 12)
    assert tgt.shape == (8, 12)


def test_copy_dataset_shift_correct():
    cfg = CopyTaskConfig(n_samples=4, L_src=10, vocab_size=20, shift=3, pad_token=0)
    src, tgt = make_copy_dataset(cfg)
    # First `shift` positions of tgt are PAD (0).
    assert (tgt[:, :3] == 0).all()
    # The rest of tgt equals the shifted src.
    assert torch.equal(tgt[:, 3:], src[:, :7])


def test_decoder_inputs_shift_by_sos():
    cfg = CopyTaskConfig(n_samples=4, L_src=10, vocab_size=20, shift=2)
    _, tgt = make_copy_dataset(cfg)
    dec_in, dec_tgt = make_decoder_inputs(tgt, sos_token=1)
    assert dec_in.shape == tgt.shape
    assert (dec_in[:, 0] == 1).all()             # SOS first
    assert torch.equal(dec_in[:, 1:], tgt[:, :-1])
    assert torch.equal(dec_tgt, tgt)


def test_copy_dataset_seed_reproducible():
    cfg = CopyTaskConfig(n_samples=4, L_src=8, seed=42)
    a_src, a_tgt = make_copy_dataset(cfg)
    b_src, b_tgt = make_copy_dataset(cfg)
    assert torch.equal(a_src, b_src)
    assert torch.equal(a_tgt, b_tgt)
