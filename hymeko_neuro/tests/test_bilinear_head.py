"""Tests for the ComplEx-style complex diagonal edge head.
Plan: docs/plans/2026-06-17-link-head-ablation/.

Run: pytest -p no:randomly hymeko_neuro/tests/test_bilinear_head.py
"""
from __future__ import annotations

import pytest
import torch

from hymeko_neuro.hyperedge.bilinear_head import ComplexDiagonalHead


def test_rejects_odd_dimension():
    with pytest.raises(ValueError):
        ComplexDiagonalHead(d=7)


def test_dead_start_is_off():
    head = ComplexDiagonalHead(d=8)            # gamma_init=1e-3
    out = head(torch.randn(5, 8), torch.randn(5, 8))
    assert out.shape == (5,)
    assert out.abs().max() < 0.05


def test_symmetric_when_imaginary_weight_zero():
    """Im(w)=0 → ComplEx is symmetric, φ(u,v)=φ(v,u) (DistMult-like)."""
    torch.manual_seed(0)
    head = ComplexDiagonalHead(d=8)
    with torch.no_grad():
        head.gamma.fill_(1.0)
        head.w_im.zero_()
    u, v = torch.randn(6, 8), torch.randn(6, 8)
    assert torch.allclose(head(u, v), head(v, u), atol=1e-6)


def test_asymmetric_with_imaginary_weight():
    """Im(w)≠0 → asymmetric (can model directed/antisymmetric sign structure)."""
    torch.manual_seed(1)
    head = ComplexDiagonalHead(d=8)
    with torch.no_grad():
        head.gamma.fill_(1.0)
        head.w_im.fill_(0.5)
    u, v = torch.randn(6, 8), torch.randn(6, 8)
    assert not torch.allclose(head(u, v), head(v, u), atol=1e-4)


def test_gradient_flows():
    head = ComplexDiagonalHead(d=8)
    head(torch.randn(4, 8), torch.randn(4, 8)).pow(2).sum().backward()
    assert head.w_re.grad is not None and head.w_im.grad is not None
    assert head.gamma.grad is not None
