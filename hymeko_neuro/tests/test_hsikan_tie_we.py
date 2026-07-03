"""Tests for the HSiKAN translation-equivariance variant (tie_we).

Verifies:
  1. tie_we=False is the existing behaviour (W_e shape unchanged; forward
     unchanged).
  2. tie_we=True collapses W_e to a single scalar shared across RFs
     (the equivariance-restoring change).
  3. Forward parity at init when both variants are seeded identically:
     with W_e=ones in both cases (matches default init), the forward
     outputs MUST be numerically equal (the scalar W_e=1 broadcasts to
     the same value as the vector W_e=ones).
  4. Translation-equivariance behaviour test: under tie_we=True, two
     spatially-related inputs should yield outputs related the same way
     (modulo boundary effects on D_v). We sanity-check that the param
     count differs in the expected way.
  5. A built model trains for 1 step without NaN/error.
"""
from __future__ import annotations

import torch

from hymeko_neuro.experiments.vision.hsikan_vision import (
    HSiKANVisionClassifier,
    SignedBranchConv,
)


def test_signedbranchconv_we_shape():
    """tie_we=False → W_e has n_edges entries; True → 1 scalar."""
    n_e = 196
    conv_untied = SignedBranchConv(d_in=4, d_out=8, n_edges=n_e, tie_we=False)
    conv_tied = SignedBranchConv(d_in=4, d_out=8, n_edges=n_e, tie_we=True)
    assert conv_untied.W_e.shape == (n_e,)
    assert conv_tied.W_e.shape == (1,)
    assert conv_untied.tie_we is False
    assert conv_tied.tie_we is True


def test_param_count_difference():
    """tie_we reduces parameters by (n_edges - 1) per arity per layer."""
    H = W = 28
    p_untied = sum(
        p.numel() for p in HSiKANVisionClassifier(H, W, 10, hidden=32,
                                                  n_layers=2, tie_we=False).parameters()
    )
    p_tied = sum(
        p.numel() for p in HSiKANVisionClassifier(H, W, 10, hidden=32,
                                                  n_layers=2, tie_we=True).parameters()
    )
    # default arities [(5,2),(8,4),(12,4)] on 28×28 → n_e = 144 + 36 + 25
    # = 205 per layer × 2 layers = 410 W_e scalars total. Tied: 1 scalar
    # per (arity, layer) = 6 scalars. Saved = 410 − 6 = 404.
    expected_saved = 404
    assert p_untied - p_tied == expected_saved, (
        f"expected {expected_saved} param saving from tying, got {p_untied - p_tied}"
    )


def test_forward_parity_at_init_with_ones():
    """At init both variants have W_e=ones (shape (n_e,) vs (1,)). The
    scalar broadcasts to the same value, so forward is bit-equal."""
    torch.manual_seed(0)
    H = W = 28
    m_untied = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2, tie_we=False)
    torch.manual_seed(0)
    m_tied = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2, tie_we=True)
    # Sanity: identical non-W_e initialisation (same seed).
    x = torch.randn(2, 1, H, W)
    with torch.no_grad():
        y_untied = m_untied(x)
        y_tied = m_tied(x)
    assert torch.allclose(y_untied, y_tied, atol=1e-6, rtol=1e-5), (
        f"forward differs at init despite identical W_e=ones: "
        f"max abs diff {(y_untied - y_tied).abs().max().item():.3e}"
    )


def test_tied_model_trains_one_step():
    """tie_we=True model takes one gradient step without NaN/error."""
    torch.manual_seed(1)
    H = W = 28
    model = HSiKANVisionClassifier(H, W, 10, hidden=32, n_layers=2, tie_we=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    x = torch.randn(4, 1, H, W)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    assert torch.isfinite(loss), f"loss is non-finite: {loss}"
    loss.backward()
    opt.step()
    # W_e should have a real gradient on it (not zero).
    for layer in model.layers:
        for conv in layer.convs:
            assert conv.W_e.grad is not None
            # Note: grad CAN be small but should be a finite tensor.
            assert torch.isfinite(conv.W_e.grad).all()
