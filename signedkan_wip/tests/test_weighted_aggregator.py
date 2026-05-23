"""Tests for the K_g-class continuous-weight WeightedHSiKANAggregator."""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.hypergraph.weighted_aggregator import (
    WeightedHSiKANAggregator, _catmull_rom_eval,
)


# ─── Catmull-Rom primitive ──────────────────────────────────────────


def test_catmull_rom_passes_through_control_values_at_knots():
    """For x at knot positions, the interpolant must (approximately)
    pass through the control values. We test at the inner knots
    (excluding the two boundary knots whose extrapolation behaviour
    differs)."""
    G = 7
    w_min, w_max = -1.0, 1.0
    torch.manual_seed(0)
    y_knots = torch.randn(G)
    for g in range(1, G - 2):
        x = torch.tensor(w_min + g / (G - 1) * (w_max - w_min))
        y_predicted = _catmull_rom_eval(y_knots, x, w_min, w_max)
        assert abs(y_predicted.item() - y_knots[g].item()) < 1e-5, (
            f"knot {g}: predicted {y_predicted.item():.4f} != y_knots[{g}]={y_knots[g].item():.4f}"
        )


def test_catmull_rom_handles_batched_input():
    G = 7
    y_knots = torch.linspace(-1.0, 1.0, G)
    x = torch.linspace(-1.0, 1.0, 13).view(13, 1).expand(13, 5)
    y = _catmull_rom_eval(y_knots, x, -1.0, 1.0)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_catmull_rom_clamps_extrapolation():
    """Inputs outside [w_min, w_max] should evaluate finitely (the
    `t.clamp(...)` step holds them at the boundary)."""
    y_knots = torch.linspace(-1.0, 1.0, 7)
    out = _catmull_rom_eval(
        y_knots, torch.tensor([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]),
        -1.0, 1.0,
    )
    assert torch.isfinite(out).all()
    # In-range mid-point should give y_knots[3] (=0.0).
    mid = _catmull_rom_eval(y_knots, torch.tensor(0.0), -1.0, 1.0)
    assert abs(mid.item()) < 1e-5


# ─── WeightedHSiKANAggregator ───────────────────────────────────────


def test_weighted_aggregator_forward_shape():
    agg = WeightedHSiKANAggregator(d_in=8, d_hidden=16, K_g=2)
    h = torch.randn(4, 6, 8)
    w = torch.rand(4, 6) * 2 - 1  # in [-1, 1]
    z = agg(h, w)
    assert z.shape == (4, 16)
    assert torch.isfinite(z).all()


@pytest.mark.parametrize("K_g", [2, 4, 6])
def test_weighted_aggregator_K_g_classes(K_g):
    agg = WeightedHSiKANAggregator(d_in=8, d_hidden=16, K_g=K_g)
    h = torch.randn(3, 5, 8)
    w = torch.rand(3, 5) * 2 - 1
    z = agg(h, w)
    assert z.shape == (3, 16)


def test_weighted_aggregator_gradient_flows():
    torch.manual_seed(0)
    agg = WeightedHSiKANAggregator(d_in=8, d_hidden=16, K_g=2)
    h = torch.randn(4, 6, 8, requires_grad=True)
    w = (torch.rand(4, 6) * 2 - 1).requires_grad_(True)
    z = agg(h, w)
    z.pow(2).mean().backward()
    for name, p in agg.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_weighted_aggregator_K_g_2_param_count_smaller_than_K_g_6():
    a2 = WeightedHSiKANAggregator(d_in=8, d_hidden=16, K_g=2)
    a6 = WeightedHSiKANAggregator(d_in=8, d_hidden=16, K_g=6)
    n2 = a2.num_params()
    n6 = a6.num_params()
    assert n6 > 2 * n2


def test_weighted_aggregator_rejects_bad_shapes():
    agg = WeightedHSiKANAggregator(d_in=8, d_hidden=16, K_g=2)
    with pytest.raises(ValueError, match="h_corners must be"):
        agg(torch.randn(4, 8), torch.rand(4))
    with pytest.raises(ValueError, match="w_corners must be"):
        agg(torch.randn(4, 6, 8), torch.rand(4, 5))


def test_weighted_aggregator_rejects_K_g_lt_2():
    with pytest.raises(ValueError, match="K_g must be"):
        WeightedHSiKANAggregator(d_in=8, K_g=1)


def test_gate_values_per_class_shape():
    agg = WeightedHSiKANAggregator(d_in=8, d_hidden=16, K_g=4)
    w = torch.rand(3, 5) * 2 - 1
    gates = agg.gate_values(w)
    assert gates.shape == (4, 3, 5)
    assert torch.isfinite(gates).all()


def test_weighted_aggregator_binary_responds_to_sign_flip():
    """At K_g=2 with the default Gaussian-class init, flipping all
    arc weights from +1 to -1 should produce a meaningfully different
    output (the two classes route differently for opposite-signed
    weights)."""
    torch.manual_seed(0)
    agg = WeightedHSiKANAggregator(d_in=8, d_hidden=16, K_g=2)
    agg.eval()
    h = torch.randn(2, 6, 8)
    w_pos = torch.ones(2, 6)
    w_neg = -torch.ones(2, 6)
    with torch.no_grad():
        z_pos = agg(h, w_pos)
        z_neg = agg(h, w_neg)
    diff = (z_pos - z_neg).abs().max().item()
    assert diff > 1e-3, f"K_g=2 aggregator did not distinguish +1 vs -1 weights (max diff {diff})"
