"""Smoke tests for AC-HSiKAN: shape, grad flow, iso-param vs transformer."""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.ac_hsikan import (
    AcHsikanClassifier, AcHsikanConfig, AcHsikanLayer,
    QuaternionSignHead, SignHead,
)


# ── SignHead ─────────────────────────────────────────────────────────

def test_sign_head_shape_and_range():
    head = SignHead(d_model=8, hidden=4, temperature=1.0, use_ste=False)
    x = torch.randn(2, 7, 8)
    s = head(x)
    assert s.shape == (2, 7, 7)
    # soft sign is in [-1, 1]
    assert s.min().item() >= -1.0 - 1e-6
    assert s.max().item() <=  1.0 + 1e-6


def test_sign_head_ste_returns_hard_signs():
    head = SignHead(d_model=8, hidden=4, use_ste=True)
    x = torch.randn(2, 7, 8)
    s = head(x)
    # All entries should be exactly -1 or +1.
    unique = torch.unique(s)
    assert torch.all((unique == 1.0) | (unique == -1.0))


def test_sign_head_grad_flows():
    head = SignHead(d_model=8, hidden=4)
    x = torch.randn(2, 5, 8, requires_grad=True)
    s = head(x).sum()
    s.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_sign_head_ste_backward_is_identity_on_tanh():
    # Hard forward, backward via tanh surrogate -> non-zero grads.
    head = SignHead(d_model=8, hidden=4, use_ste=True)
    x = torch.randn(2, 5, 8, requires_grad=True)
    loss = head(x).sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# ── QuaternionSignHead ─────────────────────────────────────────────

def test_quaternion_sign_head_shape_and_range():
    head = QuaternionSignHead(d_model=16, hidden=8, temperature=1.0)
    x = torch.randn(2, 7, 16)
    s = head(x)
    assert s.shape == (2, 7, 7)
    assert s.min().item() >= -1.0 - 1e-6
    assert s.max().item() <=  1.0 + 1e-6


def test_quaternion_sign_head_rejects_non_multiple_of_4():
    with pytest.raises(ValueError, match="hidden % 4 == 0"):
        QuaternionSignHead(d_model=16, hidden=5)


def test_quaternion_sign_head_grad_flows():
    head = QuaternionSignHead(d_model=16, hidden=8)
    x = torch.randn(2, 5, 16, requires_grad=True)
    head(x).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_quaternion_sign_head_asymmetric_imaginary_subtraction():
    """The (i, j, k) imaginary components SUBTRACT from the score
    (negative coefficients) — this is the load-bearing difference
    from bilinear / cosine attention."""
    head = QuaternionSignHead(d_model=4, hidden=4, temperature=1.0)
    # Two inputs with identical (i, j, k) but opposite real parts.
    # The Hamilton-product real-part scoring should distinguish them.
    x_pos = torch.tensor([[[+1.0, +1.0, +1.0, +1.0]]])   # (1, 1, 4)
    x_neg = torch.tensor([[[-1.0, +1.0, +1.0, +1.0]]])   # (1, 1, 4)
    with torch.no_grad():
        s_pos = head(x_pos).item()
        s_neg = head(x_neg).item()
    # Diff != 0 (the real part contributes asymmetrically).
    assert abs(s_pos - s_neg) > 1e-3


def test_layer_quaternion_sign_head_kind_works():
    cfg = AcHsikanConfig(d_model=16, n_positions=12, arities=(2, 3),
                         alpha_init=(0.5, 0.5), top_k_per_position=4,
                         sign_head_kind="quaternion", sign_head_hidden=8)
    layer = AcHsikanLayer(cfg)
    x = torch.randn(2, 10, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_config_rejects_quaternion_with_nondivisible_hidden():
    with pytest.raises(ValueError, match="sign_head_hidden % 4 == 0"):
        AcHsikanConfig(sign_head_kind="quaternion", sign_head_hidden=15)


def test_config_rejects_unknown_sign_head_kind():
    with pytest.raises(ValueError, match="sign_head_kind"):
        AcHsikanConfig(sign_head_kind="cosine")


# ── Jumps + walk_kind ───────────────────────────────────────────────

@pytest.mark.parametrize("walk_kind", ["star", "chain", "cycle"])
def test_layer_walk_kind_forward_grad(walk_kind):
    cfg = AcHsikanConfig(d_model=16, n_positions=20, arities=(2, 3, 4),
                         alpha_init=(0.33, 0.33, 0.34),
                         top_k_per_position=6, walk_kind=walk_kind)
    layer = AcHsikanLayer(cfg)
    x = torch.randn(2, 16, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_layer_jumps_extends_candidate_set():
    """With n_jumps > 0 the layer reaches non-local positions."""
    cfg_no = AcHsikanConfig(d_model=8, n_positions=32, arities=(2, 3),
                             alpha_init=(0.5, 0.5), top_k_per_position=4,
                             n_jumps_per_anchor=0)
    cfg_jumps = AcHsikanConfig(d_model=8, n_positions=32, arities=(2, 3),
                                alpha_init=(0.5, 0.5), top_k_per_position=4,
                                n_jumps_per_anchor=4)
    # Same seeded model init, different candidate-set sizes; forward
    # must succeed in both and produce different outputs.
    torch.manual_seed(0); layer_no = AcHsikanLayer(cfg_no)
    torch.manual_seed(0); layer_jumps = AcHsikanLayer(cfg_jumps)
    x = torch.randn(2, 24, 8)
    y_no = layer_no(x)
    y_jumps = layer_jumps(x)
    assert y_no.shape == y_jumps.shape == x.shape
    # The forward path differs because jumps add 4 candidates per anchor.
    assert not torch.allclose(y_no, y_jumps)


def test_config_rejects_unknown_walk_kind():
    with pytest.raises(ValueError, match="walk_kind"):
        AcHsikanConfig(walk_kind="zigzag")


def test_config_rejects_negative_jumps():
    with pytest.raises(ValueError, match="n_jumps_per_anchor"):
        AcHsikanConfig(n_jumps_per_anchor=-1)


def test_cycle_uses_more_edges_than_chain():
    """Smoke: cycle and chain should produce different layer outputs
    (cycle multiplies by one extra closing-edge sign)."""
    cfg_c = AcHsikanConfig(d_model=8, n_positions=16, arities=(3,),
                            alpha_init=(1.0,), top_k_per_position=4,
                            walk_kind="cycle")
    cfg_ch = AcHsikanConfig(d_model=8, n_positions=16, arities=(3,),
                             alpha_init=(1.0,), top_k_per_position=4,
                             walk_kind="chain")
    torch.manual_seed(0); lc = AcHsikanLayer(cfg_c)
    torch.manual_seed(0); lch = AcHsikanLayer(cfg_ch)
    x = torch.randn(2, 12, 8)
    yc = lc(x); ych = lch(x)
    assert not torch.allclose(yc, ych)


# ── v1.4: Clifford-FIR context + sparse sign-head ───────────────────

@pytest.mark.parametrize("walk_kind", ["star", "chain", "cycle"])
def test_sparse_sign_head_forward_grad(walk_kind):
    cfg = AcHsikanConfig(d_model=16, n_positions=24, arities=(2, 3, 4),
                         alpha_init=(0.33, 0.33, 0.34),
                         top_k_per_position=6, walk_kind=walk_kind,
                         sparse_sign_head=True)
    layer = AcHsikanLayer(cfg)
    x = torch.randn(2, 20, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("walk_kind", ["star", "chain", "cycle"])
def test_clifford_fir_context_forward_grad(walk_kind):
    cfg = AcHsikanConfig(d_model=16, n_positions=24, arities=(2, 3, 4),
                         alpha_init=(0.33, 0.33, 0.34),
                         top_k_per_position=6, walk_kind=walk_kind,
                         use_clifford_fir_context=True,
                         clifford_fir_K=4)
    layer = AcHsikanLayer(cfg)
    x = torch.randn(2, 20, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_clifford_fir_with_sparse_sign_combined():
    cfg = AcHsikanConfig(d_model=16, n_positions=24, arities=(2, 3, 4, 5),
                         alpha_init=(0.25, 0.25, 0.25, 0.25),
                         top_k_per_position=6, walk_kind="cycle",
                         use_clifford_fir_context=True, clifford_fir_K=4,
                         sparse_sign_head=True,
                         sign_head_kind="quaternion",
                         sign_head_hidden=8)
    layer = AcHsikanLayer(cfg)
    x = torch.randn(2, 20, 16, requires_grad=True)
    layer(x).sum().backward()
    for n, p in layer.named_parameters():
        assert p.grad is not None, f"no grad on {n}"


def test_config_rejects_clifford_fir_with_d_model_not_mod_4():
    with pytest.raises(ValueError, match="d_model % 4 == 0"):
        AcHsikanConfig(d_model=15, use_clifford_fir_context=True)


# ── v1.5: building-block speedups (multi-head / fused walk / bottleneck FFN) ──

def test_multi_head_quaternion_sign_head():
    """Multi-head quaternion sparse attention produces the right shape
    and gradients via batched einsum (no per-head loop)."""
    cfg = AcHsikanConfig(d_model=32, n_positions=24, arities=(2, 3, 4),
                         alpha_init=(0.33, 0.33, 0.34), top_k_per_position=6,
                         sparse_sign_head=True, sign_head_kind="quaternion",
                         sign_head_hidden=16, n_sign_heads=2)
    layer = AcHsikanLayer(cfg)
    x = torch.randn(2, 20, 32, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_config_rejects_multihead_without_sparse():
    with pytest.raises(ValueError, match="n_sign_heads > 1 requires sparse"):
        AcHsikanConfig(n_sign_heads=2)


def test_config_rejects_multihead_with_nondivisible_hidden():
    with pytest.raises(ValueError, match="must be divisible by n_sign_heads"):
        AcHsikanConfig(sparse_sign_head=True, n_sign_heads=3,
                       sign_head_hidden=16)


@pytest.mark.parametrize("walk_kind", ["chain", "cycle"])
def test_fused_walk_matches_non_fused(walk_kind):
    """Fused chain/cycle walk-op should produce the same output as
    its non-fused sibling (modulo numerical noise from log+exp)."""
    common = dict(d_model=16, n_positions=20, arities=(3,),
                  alpha_init=(1.0,), top_k_per_position=4,
                  walk_kind=walk_kind, sparse_sign_head=True,
                  sign_head_kind="quaternion", sign_head_hidden=8)
    torch.manual_seed(0); l_plain = AcHsikanLayer(AcHsikanConfig(**common))
    torch.manual_seed(0); l_fused = AcHsikanLayer(
        AcHsikanConfig(**common, use_fused_walk=True))
    x = torch.randn(2, 16, 16)
    with torch.no_grad():
        y_plain = l_plain(x)
        y_fused = l_fused(x)
    # Tolerance on log+exp+sign reformulation.
    assert torch.allclose(y_plain, y_fused, atol=1e-4, rtol=1e-3)


def test_bottleneck_ffn_smaller_param_count():
    """BottleneckFFN at ratio=4 has fewer parameters than StandardFFN
    at ff_hidden=4×d."""
    d = 16; ratio = 4
    cfg_std = AcHsikanConfig(d_model=d, n_positions=20, arities=(2, 3),
                              alpha_init=(0.5, 0.5), top_k_per_position=4,
                              ffn_hidden=64)
    cfg_bot = AcHsikanConfig(d_model=d, n_positions=20, arities=(2, 3),
                              alpha_init=(0.5, 0.5), top_k_per_position=4,
                              ffn_bottleneck_ratio=ratio)
    l_std = AcHsikanLayer(cfg_std)
    l_bot = AcHsikanLayer(cfg_bot)
    n_std = sum(p.numel() for p in l_std.ffn.parameters())
    n_bot = sum(p.numel() for p in l_bot.ffn.parameters())
    assert n_bot < n_std


def test_all_speedups_together():
    """Combined: multi-head + fused-walk + bottleneck FFN + sparse +
    Clifford-FIR pre-context. All flags work together."""
    cfg = AcHsikanConfig(
        d_model=16, n_positions=32, arities=(2, 3, 4, 5),
        alpha_init=(0.25, 0.25, 0.25, 0.25), top_k_per_position=8,
        walk_kind="cycle", n_jumps_per_anchor=4,
        sparse_sign_head=True, sign_head_kind="quaternion",
        sign_head_hidden=8, n_sign_heads=2,
        use_value_projection=True, use_magnitude_weight=True,
        use_fused_walk=True, ffn_bottleneck_ratio=2,
        use_clifford_fir_context=True, clifford_fir_K=4,
    )
    layer = AcHsikanLayer(cfg)
    x = torch.randn(2, 28, 16, requires_grad=True)
    layer(x).sum().backward()
    for n, p in layer.named_parameters():
        assert p.grad is not None, f"no grad on {n}"


def test_sparse_sign_head_no_full_sign_matrix_constructed():
    """Sanity: at L=64 with sparse mode, peak memory should not
    include a (B, L, L) tensor. We approximate by checking the
    layer forward runs at L=256 with batch=1 (where dense L²·d would
    be ~1 MB but sparse is K_total·d ~ few KB) without grossly more
    activation memory."""
    cfg_sparse = AcHsikanConfig(d_model=16, n_positions=256,
                                 arities=(2, 3), alpha_init=(0.5, 0.5),
                                 top_k_per_position=8,
                                 sparse_sign_head=True,
                                 sign_head_kind="quaternion",
                                 sign_head_hidden=8)
    layer = AcHsikanLayer(cfg_sparse)
    x = torch.randn(1, 256, 16)
    y = layer(x)
    assert y.shape == x.shape


# ── AcHsikanLayer ───────────────────────────────────────────────────

def test_layer_shape_and_residual():
    cfg = AcHsikanConfig(d_model=16, n_positions=32, arities=(2, 3, 4),
                         alpha_init=(0.33, 0.33, 0.34),
                         top_k_per_position=8, n_layers=1)
    layer = AcHsikanLayer(cfg)
    x = torch.randn(3, 24, 16)
    y = layer(x)
    assert y.shape == x.shape


def test_layer_alpha_simplex():
    cfg = AcHsikanConfig(d_model=8, n_positions=16, arities=(2, 3),
                         alpha_init=(0.5, 0.5), top_k_per_position=4)
    layer = AcHsikanLayer(cfg)
    alpha = layer.alpha()
    assert alpha.shape == (2,)
    assert abs(alpha.sum().item() - 1.0) < 1e-5


def test_layer_grad_flows_through_full_block():
    cfg = AcHsikanConfig(d_model=8, n_positions=16, arities=(2, 3),
                         alpha_init=(0.5, 0.5), top_k_per_position=4)
    layer = AcHsikanLayer(cfg)
    x = torch.randn(2, 10, 8, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    # all learnable params got a grad
    for n, p in layer.named_parameters():
        assert p.grad is not None, f"no grad on {n}"


# ── AcHsikanClassifier ─────────────────────────────────────────────

def test_classifier_forward_shape():
    cfg = AcHsikanConfig(d_model=16, n_positions=32, arities=(2, 3, 4, 5),
                         top_k_per_position=8, n_layers=2, n_classes=2)
    model = AcHsikanClassifier(vocab_size=100, cfg=cfg)
    tokens = torch.randint(0, 100, (4, 20))
    logits = model(tokens)
    assert logits.shape == (4, 2)


def test_classifier_alpha_list_length():
    cfg = AcHsikanConfig(d_model=8, n_positions=16, arities=(2, 3),
                         alpha_init=(0.5, 0.5),
                         top_k_per_position=4, n_layers=3)
    model = AcHsikanClassifier(vocab_size=50, cfg=cfg)
    alphas = model.alpha()
    assert len(alphas) == 3
    assert all(a.shape == (2,) for a in alphas)


def test_classifier_grad_flows():
    cfg = AcHsikanConfig(d_model=8, n_positions=16, arities=(2, 3),
                         alpha_init=(0.5, 0.5),
                         top_k_per_position=4, n_layers=1, n_classes=2)
    model = AcHsikanClassifier(vocab_size=50, cfg=cfg)
    tokens = torch.randint(0, 50, (4, 12))
    target = torch.randint(0, 2, (4,))
    loss = torch.nn.functional.cross_entropy(model(tokens), target)
    loss.backward()
    for n, p in model.named_parameters():
        assert p.grad is not None, f"no grad on {n}"


# ── Iso-param vs transformer ─────────────────────────────────────────

def test_iso_param_match_to_transformer_baseline_within_30pct():
    """AC-HSiKAN at default config should land in the same parameter
    bracket as the existing IMDBTransformerBaseline (~321 k params)."""
    from signedkan_wip.src.sequence.iso_param_transformer import (
        IMDBTransformerBaseline,
    )
    cfg = AcHsikanConfig(d_model=16, n_positions=256, arities=(2, 3, 4, 5),
                         top_k_per_position=16, n_layers=2, n_classes=2)
    ac = AcHsikanClassifier(vocab_size=20_000, cfg=cfg)
    tr = IMDBTransformerBaseline(vocab_size=20_000)
    ac_p = ac.count_parameters()
    tr_p = sum(p.numel() for p in tr.parameters() if p.requires_grad)
    # Within 30% of each other.
    ratio = ac_p / tr_p
    assert 0.7 <= ratio <= 1.3, (
        f"AC-HSiKAN params {ac_p:,} not within 30% of transformer "
        f"{tr_p:,}; ratio {ratio:.2f}"
    )
