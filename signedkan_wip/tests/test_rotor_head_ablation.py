"""Tests for the fair head-algebra ablation harness (all heads on the same
propagated rotors q). Plan: docs/plans/2026-06-17-link-head-ablation/.

Run: pytest -p no:randomly signedkan_wip/tests/test_rotor_head_ablation.py
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.experiments.runs.run_rotor_head_ablation import HEADS, RotorHeadModel

EDGES = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3], [3, 4]], dtype=torch.long)
SIGNS = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0])
STRUCT_DIM = 6


def _model(kind, prop_rounds=1):
    torch.manual_seed(0)
    return RotorHeadModel(hidden=12, head_kind=kind, prop_rounds=prop_rounds,
                          prop_self_weight=4.0, edges=EDGES, signs=SIGNS)


@pytest.mark.parametrize("kind", HEADS)
def test_forward_shape_and_finite(kind):
    """Every head scores the same propagated rotors → finite (E,) logits."""
    out = _model(kind)(torch.randn(6, STRUCT_DIM), EDGES)
    assert out.shape == (EDGES.shape[0],)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("kind", HEADS)
def test_gradient_flows(kind):
    m = _model(kind)
    m(torch.randn(6, STRUCT_DIM), EDGES).pow(2).sum().backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rejects_unknown_head():
    with pytest.raises(ValueError):
        RotorHeadModel(12, "octonion", 1, 4.0, EDGES, SIGNS)
