"""Phase-1 isolation tests for the signed-KAN core (no consumer touched yet).

Covers: the dense backend aggregation, Catmull-Rom parity with ``signedkan_wip``, the skip/highway modes, and the
three incidence modes (incl. the ``weighted`` real-arc-weight parity + sparsity-mask guard).
"""
from __future__ import annotations

import pytest
import torch

from signed_kan import (
    CatmullRomActivation,
    DenseBatchedBackend,
    HighwaySkip,
    SignedKANBackbone,
    SignedKANLayer,
    catmull_rom,
)

# A fixed 3-vertex signed adjacency (a tiny chain) — no MuJoCo / consumer needed.
_A_POS = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
_A_NEG = _A_POS * 0.3


# ── aggregation backend ──────────────────────────────────────────────────────
def test_dense_backend_aggregates_neighbours() -> None:
    h = torch.randn(4, 3, 5)
    agg_pos, agg_neg = DenseBatchedBackend().aggregate(_A_POS, _A_NEG, h)
    assert agg_pos.shape == h.shape and agg_neg.shape == h.shape
    # receiver 0 reads only neighbour 1 (a_pos[0,1]=1); receiver 1 reads neighbours 0 and 2.
    assert torch.allclose(agg_pos[:, 0, :], h[:, 1, :])
    assert torch.allclose(agg_pos[:, 1, :], h[:, 0, :] + h[:, 2, :])
    assert torch.allclose(agg_neg[:, 0, :], 0.3 * h[:, 1, :])


def test_dense_backend_rejects_non_batched() -> None:
    with pytest.raises(ValueError, match="B, N, d"):
        DenseBatchedBackend().aggregate(_A_POS, _A_NEG, torch.randn(3, 5))


# ── Catmull-Rom parity ───────────────────────────────────────────────────────
def test_catmull_rom_parity_with_signedkan() -> None:
    """The core CR matches ``signedkan_wip``'s canonical eval (the cross-impl anchor that survives unification)."""
    try:
        from signedkan_wip.src.core.splines import _catmull_rom_eval
    except Exception:
        pytest.skip("signedkan_wip not importable in this environment")
    torch.manual_seed(0)
    coef = torch.randn(6, 5)
    x = torch.linspace(-1.5, 1.5, 64).unsqueeze(-1).expand(64, 6).contiguous()
    assert torch.allclose(catmull_rom(coef, x, 5), _catmull_rom_eval(coef, x, 5), atol=1e-6)


def test_catmull_rom_activation_shape_and_domain() -> None:
    act = CatmullRomActivation(8, grid=5)
    out = act(torch.randn(4, 3, 8) * 10.0)   # large inputs are clamped into [-1, 1]
    assert out.shape == (4, 3, 8) and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        CatmullRomActivation(0)


# ── layer + skip/highway ─────────────────────────────────────────────────────
def test_layer_skip_modes_forward() -> None:
    x = torch.randn(4, 3, 6)
    for skip in ("none", "residual", "highway"):
        layer = SignedKANLayer(6, 6, activation="cr", skip=skip)
        out = layer(x, _A_POS, _A_NEG)
        assert out.shape == (4, 3, 6) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="skip must be one of"):
        SignedKANLayer(6, 6, skip="bogus")


def test_highway_is_carry_dominant_at_init() -> None:
    """Schmidhuber highway init: gate bias = -2 → T = σ(-2) ≈ 0.12, so the carry path dominates at the start."""
    hw = HighwaySkip(6, 6)
    assert torch.allclose(hw.gate.bias, torch.full((6,), -2.0))
    x = torch.randn(100, 6)
    t = torch.sigmoid(hw.gate(x))
    assert t.mean() < 0.3   # carry-dominant (information flows through the highway initially)


def test_highway_projects_on_dim_mismatch() -> None:
    hw = HighwaySkip(4, 8)   # in != out → carry must be projected
    out = hw(torch.randn(5, 8), torch.randn(5, 4))
    assert out.shape == (5, 8) and torch.isfinite(out).all()


# ── backbone + incidence modes ───────────────────────────────────────────────
def test_backbone_forward_and_pool() -> None:
    for pool in ("mean", "sum"):
        bb = SignedKANBackbone(2, 8, 2, _A_POS, _A_NEG, pool=pool)
        assert bb(torch.randn(4, 3, 2)).shape == (4, 8)
    with pytest.raises(ValueError, match="pool must be one of"):
        SignedKANBackbone(2, 8, 2, _A_POS, _A_NEG, pool="bogus")


def test_incidence_fixed_is_buffer_learned_is_param() -> None:
    fixed = SignedKANBackbone(2, 8, 1, _A_POS, _A_NEG, incidence="fixed")
    learned = SignedKANBackbone(2, 8, 1, _A_POS, _A_NEG, incidence="learned")
    assert not isinstance(fixed.a_pos, torch.nn.Parameter)
    assert isinstance(learned.a_pos, torch.nn.Parameter) and learned.a_pos.requires_grad
    with pytest.raises(ValueError, match="incidence must be one of"):
        SignedKANBackbone(2, 8, 1, _A_POS, _A_NEG, incidence="bogus")


def test_weighted_incidence_real_weights_parity_and_mask() -> None:
    """``weighted``: a fixed structural mask (buffer) × a learned per-arc weight (init 1.0 → effective adjacency
    equals the fixed structure, parity), and the gradient stays on the real arcs only (sparsity preserved)."""
    bb = SignedKANBackbone(2, 8, 1, _A_POS, _A_NEG, incidence="weighted")
    assert not isinstance(bb.a_pos, torch.nn.Parameter)
    assert isinstance(bb.w_pos_arc, torch.nn.Parameter)
    eff_pos, eff_neg = bb._effective_adj()
    assert torch.allclose(eff_pos, _A_POS) and torch.allclose(eff_neg, _A_NEG)   # init parity
    bb(torch.randn(4, 3, 2)).pow(2).mean().backward()
    g = bb.w_pos_arc.grad
    mask = _A_POS != 0
    assert g is not None and torch.isfinite(g).all()
    assert g[mask].abs().sum() > 0                                # real arcs learn
    assert torch.allclose(g[~mask], torch.zeros_like(g[~mask]))   # non-arcs inert (masked)


def test_backbone_rejects_bad_adjacency() -> None:
    with pytest.raises(ValueError, match="square"):
        SignedKANBackbone(2, 8, 1, torch.randn(3, 4), torch.randn(3, 4))
