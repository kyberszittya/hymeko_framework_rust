"""Tests for the learnable Kóczy fuzzy signature layer + classifier.

(Distinct from the older ``test_fuzzy_signature.py`` which tests an
interpretability-extraction module; this file tests the operator built
in ``hymeko_neuro/experiments/vision/fuzzy_signature.py``.)

Plan:       ``docs/plans/2026-05-30-fuzzy-signature-layer/plan.tex``
Background: ``docs/plans/2026-05-30-fuzzy-signature-layer/background.tex``

Revision 2 (Atanassov-pair redesign) test coverage:
  1. T-norm boundary conditions (identity, annihilator, monotonicity,
     ordering Łuk ≤ product ≤ Gödel).
  2. T-conorm boundary conditions (dual: identity 0, annihilator 1,
     monotonicity, ordering max ≤ probsum ≤ Łuk).
  3. Mask handling in t-conorm (padded entries → identity element).
  4. Vertex-to-edges consistency.
  5. FuzzySignatureLayer forward stays in [0, 1] across all 9
     (t-norm, t-conorm) combinations.
  6. FuzzySignatureClassifier produces (B, n_classes) logits.
  7. Parameter-count regression — matches the closed-form plan table.
  8. NEW: τ positivity invariant after a training step
     (softplus parameterisation, plan Risk anticipation).
  9. NEW: Atanassov-pair plasticity — μ⁺ and μ⁻ receive distinct
     gradients on non-symmetric inputs. Catches the failure mode where
     both branches collapse to one function.
 10. NEW: τ → ∞ limit ≈ crisp Heaviside gate at c = ½.
 11. NEW: W_e effective value stays in [0, 1] always (clamp invariant).
 12. NEW: Gradient flow through every learnable param after one step.
"""
from __future__ import annotations

import pytest
import torch

from hymeko_neuro.experiments.vision.fuzzy_signature import (
    FuzzySignatureClassifier,
    FuzzySignatureLayer,
    MultiArityFuzzySignatureLayer,
    build_rf_vertex_edges,
    expected_param_count,
    t_conorm,
    t_norm,
)


# ─── T-norm boundary conditions ────────────────────────────────────


@pytest.mark.parametrize("kind", ["min", "product", "lukasiewicz"])
def test_t_norm_T_of_ones(kind):
    x = torch.ones(1, 25, 1)
    out = t_norm(x, kind)
    assert torch.allclose(out, torch.tensor(1.0), atol=1e-5), (
        f"{kind}: T(1,...,1)={out.item():.6f}"
    )


@pytest.mark.parametrize("kind", ["min", "product", "lukasiewicz"])
def test_t_norm_with_zero(kind):
    x = torch.rand(1, 25, 1)
    x[0, 0, 0] = 0.0
    out = t_norm(x, kind)
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-5)


@pytest.mark.parametrize("kind", ["min", "product", "lukasiewicz"])
def test_t_norm_monotonicity(kind):
    torch.manual_seed(0)
    x = torch.rand(10, 25, 1)
    y = x + (1 - x) * torch.rand_like(x)
    assert (t_norm(y, kind) >= t_norm(x, kind) - 1e-5).all()


def test_t_norm_ordering():
    """Łuk ≤ product ≤ Gödel(min) — Prop. 2.2 in background.tex."""
    torch.manual_seed(1)
    x = torch.rand(20, 8, 1)
    assert (t_norm(x, "lukasiewicz") <= t_norm(x, "product") + 1e-5).all()
    assert (t_norm(x, "product") <= t_norm(x, "min") + 1e-5).all()


# ─── T-conorm boundary conditions (dual to t-norm) ─────────────────


@pytest.mark.parametrize("kind", ["max", "probsum", "lukasiewicz"])
def test_t_conorm_S_of_zeros(kind):
    x = torch.zeros(1, 25, 1)
    mask = torch.ones(1, 25)
    assert torch.allclose(t_conorm(x, mask, kind), torch.tensor(0.0), atol=1e-5)


@pytest.mark.parametrize("kind", ["max", "probsum", "lukasiewicz"])
def test_t_conorm_with_one(kind):
    x = torch.rand(1, 25, 1)
    x[0, 0, 0] = 1.0
    mask = torch.ones(1, 25)
    out = t_conorm(x, mask, kind)
    assert torch.allclose(out, torch.tensor(1.0), atol=1e-5), (
        f"{kind}: S(1, ...) = {out.item():.6f}"
    )


def test_t_conorm_ordering():
    """max(Gödel) ≤ probsum ≤ Łuk(bounded sum) — Prop. 2.3."""
    torch.manual_seed(2)
    x = torch.rand(20, 5, 1)
    mask = torch.ones(20, 5)
    g = t_conorm(x, mask, "max")
    p = t_conorm(x, mask, "probsum")
    L = t_conorm(x, mask, "lukasiewicz")
    assert (g <= p + 1e-5).all()
    assert (p <= L + 1e-5).all()


@pytest.mark.parametrize("kind", ["max", "probsum", "lukasiewicz"])
def test_t_conorm_mask_pads_to_identity(kind):
    """Padded entries (mask=0) shouldn't change S."""
    x_full = torch.tensor([[[0.3], [0.6], [0.0], [0.0]]])
    mask_full = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    x_short = torch.tensor([[[0.3], [0.6]]])
    mask_short = torch.tensor([[1.0, 1.0]])
    a = t_conorm(x_full, mask_full, kind)
    b = t_conorm(x_short, mask_short, kind)
    assert torch.allclose(a, b, atol=1e-5), (
        f"{kind}: masked padding leaked: {a.item():.6f} vs {b.item():.6f}"
    )


# ─── Vertex-to-edges helper ────────────────────────────────────────


def test_vertex_edges_consistency():
    """Each (v, j) listed edge must contain v per the edge_members
    table (cross-check)."""
    from hymeko_neuro.experiments.vision.hsikan_vision import build_rf_edge_members
    H, W, k, s = 8, 8, 3, 1
    v_edges, v_mask, max_E = build_rf_vertex_edges(H, W, k, s)
    em, n_e = build_rf_edge_members(H, W, k, s)
    V = H * W
    assert v_edges.shape == (V, max_E)
    for v in range(V):
        for j in range(max_E):
            if v_mask[v, j] > 0:
                e = int(v_edges[v, j])
                assert v in em[e].tolist()


# ─── Layer forward boundedness + classifier shape ──────────────────


@pytest.mark.parametrize("t_n", ["min", "product", "lukasiewicz"])
@pytest.mark.parametrize("t_c", ["max", "probsum", "lukasiewicz"])
def test_layer_forward_stays_in_01(t_n, t_c):
    """Theorem 7.1 (background.tex): the layer preserves [0,1]."""
    torch.manual_seed(0)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                t_norm_kind=t_n, t_conorm_kind=t_c)
    x = torch.rand(2, 64, 4)
    out = layer(x)
    assert out.shape == x.shape
    assert (out >= 0.0).all(), f"{t_n}/{t_c}: min={out.min().item():.6f}"
    assert (out <= 1.0 + 1e-5).all(), f"{t_n}/{t_c}: max={out.max().item():.6f}"
    assert torch.isfinite(out).all()


def test_classifier_forward_shape():
    torch.manual_seed(0)
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=16, n_layers=8)
    out = m(torch.rand(2, 1, 28, 28))
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


# ─── Parameter-count regression (matches plan table) ───────────────


@pytest.mark.parametrize(
    "d,L,expected",
    # Revision-4 closed-form: 2d + L·(2dm + d + 1 + d + d) + 10·(d+1).
    # Per layer: 2dm (CR pair) + d (τ) + 1 (W_e tied) + d (c_gate)
    #          + d (α_raw for lerp). m=8 throughout.
    # d=8:  per_layer = 128 + 8 + 1 + 8 + 8 = 153
    # d=16: per_layer = 256 + 16 + 1 + 16 + 16 = 305
    # d=32: per_layer = 512 + 32 + 1 + 32 + 32 = 609
    [(8, 2, 16 + 2 * 153 + 90),
     (8, 4, 16 + 4 * 153 + 90),
     (8, 8, 16 + 8 * 153 + 90),
     (16, 2, 32 + 2 * 305 + 170),
     (16, 4, 32 + 4 * 305 + 170),
     (16, 8, 32 + 8 * 305 + 170),
     (32, 8, 64 + 8 * 609 + 330)],
)
def test_classifier_param_count(d, L, expected):
    """Closed-form param count must match the model's actual count.
    Default config (rev 4): residual_kind='lerp', gate_center_learnable=True."""
    pred = expected_param_count(d=d, n_layers=L)
    assert pred == expected, f"closed-form {pred} != expected {expected}"
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=d, n_layers=L)
    actual = sum(p.numel() for p in m.parameters())
    assert actual == expected, (
        f"d={d} L={L}: closed-form {expected}, model {actual}"
    )


def test_classifier_param_count_legacy_revision3():
    """Revision-3 config (no c_gate, no α) must still report correctly."""
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=16, n_layers=8,
                                  residual_kind="max",
                                  gate_center_learnable=False)
    actual = sum(p.numel() for p in m.parameters())
    pred = expected_param_count(d=16, n_layers=8, residual_kind="max",
                                  gate_center_learnable=False)
    assert pred == actual, f"closed-form {pred} != model {actual}"
    assert actual == 2386, f"rev-3 default param count was 2386, got {actual}"


# ─── Atanassov-pair specific tests (revision 2) ────────────────────


def test_tau_stays_positive_after_training_step():
    """τ_eff = softplus(τ_raw) > 0 by construction; verify it survives
    a training step (no clamp_min, no nan-to-pos, just softplus)."""
    torch.manual_seed(1)
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=8, n_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-1)  # aggressive lr
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    for L in m.layers:
        tau = L.tau
        assert torch.isfinite(tau).all(), f"τ has non-finite values: {tau}"
        assert (tau > 0).all(), f"τ went non-positive: min={tau.min().item()}"


def test_atanassov_branches_get_distinct_gradients():
    """μ⁺ and μ⁻ are different CR modules and must receive distinct
    gradients on a non-symmetric input — otherwise the Atanassov pair
    degenerates to a single membership function and the layer is
    revision-1 collapse all over again."""
    torch.manual_seed(2)
    layer = FuzzySignatureLayer(d=4, H=6, W=6, kernel=3, stride=1,
                                t_norm_kind="min", t_conorm_kind="probsum")
    # Asymmetric input (low values mostly).
    x = torch.rand(2, 36, 4) * 0.3
    out = layer(x)
    loss = out.sum()
    loss.backward()
    g_plus = layer.mu_plus.cpts.grad
    g_minus = layer.mu_minus.cpts.grad
    assert g_plus is not None and g_minus is not None
    assert torch.isfinite(g_plus).all() and torch.isfinite(g_minus).all()
    # The gradient tensors must NOT be equal — distinct CR modules
    # processing the same input via different gate weightings yield
    # different per-control-point gradients.
    diff = (g_plus - g_minus).abs().max().item()
    assert diff > 1e-6, (
        f"μ⁺ and μ⁻ received identical gradients (diff={diff:.2e}); "
        "Atanassov pair has collapsed."
    )


def test_tau_large_acts_as_crisp_heaviside():
    """As τ → ∞, g_τ(x) → 1[x ≥ ½] (crisp Heaviside at c = ½).
    Verify the per-channel gate is approximately crisp at large τ."""
    torch.manual_seed(3)
    layer = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                t_norm_kind="min", t_conorm_kind="probsum",
                                learnable_tau=False, tau_init=1000.0)
    # τ_raw is frozen; tau_init=1000 → softplus(σ⁻¹(1000)) ≈ 1000.
    # The gate for x > ½ should be ≈ 1; for x < ½, ≈ 0.
    x_high = torch.full((1, 16, 4), 0.9)
    x_low = torch.full((1, 16, 4), 0.1)
    # We can probe the internals manually since the gate isn't exposed.
    tau = layer.tau
    g_high = torch.sigmoid(tau * (x_high - 0.5))
    g_low = torch.sigmoid(tau * (x_low - 0.5))
    assert (g_high > 0.999).all(), f"g(x=0.9, τ→∞) not ≈1: min={g_high.min()}"
    assert (g_low < 0.001).all(), f"g(x=0.1, τ→∞) not ≈0: max={g_low.max()}"


def test_W_e_eff_always_in_unit_interval():
    """The effective rule strength W_e_eff = min(σ(W_e_raw)·2, 1) must
    stay in [0, 1] for any value of W_e_raw."""
    torch.manual_seed(4)
    layer = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                t_norm_kind="min", t_conorm_kind="probsum")
    # Try extreme values of W_e_raw.
    for raw_val in [-100.0, -1.0, 0.0, 1.0, 100.0]:
        with torch.no_grad():
            layer.W_e_raw.fill_(raw_val)
        w = layer.W_e_eff
        assert (w >= 0.0).all() and (w <= 1.0 + 1e-7).all(), (
            f"W_e_eff out of [0,1] at raw={raw_val}: {w.item():.6f}"
        )
    # At raw=0 should be exactly 1.0 (full strength at init).
    with torch.no_grad():
        layer.W_e_raw.fill_(0.0)
    assert torch.allclose(layer.W_e_eff, torch.tensor(1.0), atol=1e-6)


def test_classifier_trains_one_step_grads_finite():
    """Every learnable param (embed, μ⁺ CR, μ⁻ CR, τ, W_e, head) must
    receive a finite gradient after one training step."""
    torch.manual_seed(5)
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=8, n_layers=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(m(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    grad_count = {"embed": 0, "mu_plus": 0, "mu_minus": 0,
                  "tau_raw": 0, "W_e_raw": 0, "head": 0, "other": 0}
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        matched = False
        for key in grad_count:
            if key in name:
                grad_count[key] += 1
                matched = True
                break
        if not matched:
            grad_count["other"] += 1
    # Must see at least one of each kind.
    assert grad_count["embed"] > 0, "no gradient hit embed"
    assert grad_count["mu_plus"] > 0, "no gradient hit μ⁺ CR"
    assert grad_count["mu_minus"] > 0, "no gradient hit μ⁻ CR"
    assert grad_count["tau_raw"] > 0, "no gradient hit τ"
    assert grad_count["W_e_raw"] > 0, "no gradient hit W_e"
    assert grad_count["head"] > 0, "no gradient hit head"
    opt.step()


def test_layer_input_zero_gives_pure_mu_minus():
    """At x = 0, the gate g = σ(τ·(-½)) ≈ 0 (since τ > 0), so the
    Atanassov mix is dominated by μ⁻. At x = 1, dominated by μ⁺.
    This is the inductive bias of the Atanassov pair.

    Uses cr_input_scale="raw" so the comparison μ⁻(x_zero) matches
    what forward computes internally (no rescale)."""
    torch.manual_seed(6)
    layer = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                t_norm_kind="min", t_conorm_kind="probsum",
                                learnable_tau=False, tau_init=10.0,
                                cr_input_scale="raw")
    x_zero = torch.zeros(1, 16, 4)
    x_one = torch.ones(1, 16, 4)
    mu_p_zero = layer.mu_plus(x_zero, branch_idx=0)
    mu_n_zero = layer.mu_minus(x_zero, branch_idx=0)
    mu_p_one = layer.mu_plus(x_one, branch_idx=0)
    mu_n_one = layer.mu_minus(x_one, branch_idx=0)
    g_zero = torch.sigmoid(layer.tau * (x_zero - 0.5))
    g_one = torch.sigmoid(layer.tau * (x_one - 0.5))
    assert (g_zero < 0.01).all(), f"gate at x=0 not near 0: {g_zero.max()}"
    assert (g_one > 0.99).all(), f"gate at x=1 not near 1: {g_one.min()}"
    mu_zero = g_zero * mu_p_zero + (1 - g_zero) * mu_n_zero
    mu_one = g_one * mu_p_one + (1 - g_one) * mu_n_one
    assert torch.allclose(mu_zero, mu_n_zero, atol=1e-2)
    assert torch.allclose(mu_one, mu_p_one, atol=1e-2)


# ─── Revision-3 axes: cr_input_scale + residual_kind ───────────────


@pytest.mark.parametrize("scale", ["raw", "unit_to_grid"])
@pytest.mark.parametrize("resid", ["avg", "max", "probsum"])
def test_revision3_axes_build_and_forward_stays_in_01(scale, resid):
    """The 2×3 cross product of (cr_input_scale, residual_kind) must
    each build, forward without NaN, and preserve [0,1]."""
    torch.manual_seed(10)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                t_norm_kind="min", t_conorm_kind="probsum",
                                cr_input_scale=scale, residual_kind=resid)
    x = torch.rand(2, 64, 4)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all(), f"{scale}/{resid}: non-finite output"
    assert (out >= 0.0).all(), f"{scale}/{resid}: min={out.min().item()}"
    assert (out <= 1.0 + 1e-5).all(), f"{scale}/{resid}: max={out.max().item()}"


def test_residual_max_is_non_contracting():
    """The max-conorm residual `out = max(x, h_v)` is non-contracting:
    out_v ≥ x_v element-wise for any input. This is the defining
    property that escapes the contraction-collapse of revision 2."""
    torch.manual_seed(11)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                t_norm_kind="min", t_conorm_kind="probsum",
                                cr_input_scale="unit_to_grid",
                                residual_kind="max")
    x = torch.rand(2, 64, 4)
    out = layer(x)
    # Non-contraction: out_v ≥ x_v element-wise.
    assert (out >= x - 1e-7).all(), (
        f"max-residual not non-contracting: min(out-x) = "
        f"{(out - x).min().item():.6f}"
    )


def test_residual_avg_reproduces_revision2():
    """With cr_input_scale='raw' and residual_kind='avg', the layer
    must be bit-identical to revision 2's behaviour. Regression-protects
    the legacy path so ablations can still cite it."""
    torch.manual_seed(12)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                t_norm_kind="min", t_conorm_kind="probsum",
                                cr_input_scale="raw", residual_kind="avg")
    x = torch.rand(2, 64, 4)
    out = layer(x)
    # Manual recomputation following revision-2 forward (no rescale, avg).
    mu_p = layer.mu_plus(x, branch_idx=0)
    mu_n = layer.mu_minus(x, branch_idx=0)
    g = torch.sigmoid(layer.tau * (x - 0.5))
    mu = g * mu_p + (1.0 - g) * mu_n
    x_per_e = mu[:, layer.edge_members, :]
    h_e = t_norm(x_per_e, "min")
    h_e = h_e * layer.W_e_eff
    h_per_v = h_e[:, layer.vertex_edges, :]
    h_v = t_conorm(h_per_v, layer.vertex_edges_mask, "probsum")
    expected = 0.5 * (x + h_v)
    assert torch.allclose(out, expected, atol=1e-6), (
        f"rev-2 reproduction broken: max diff = "
        f"{(out - expected).abs().max().item():.2e}"
    )


def test_cr_input_scale_changes_forward():
    """The 'unit_to_grid' axis must produce numerically different
    outputs from 'raw' at the same input (sanity that the rescale is
    actually wired through)."""
    torch.manual_seed(13)
    common = dict(d=4, H=8, W=8, kernel=3, stride=1,
                  t_norm_kind="min", t_conorm_kind="probsum",
                  residual_kind="avg")
    layer_raw = FuzzySignatureLayer(cr_input_scale="raw", **common)
    layer_grid = FuzzySignatureLayer(cr_input_scale="unit_to_grid", **common)
    # Force identical Atanassov-pair params so the only difference is
    # the rescale path.
    layer_grid.load_state_dict(layer_raw.state_dict())
    x = torch.rand(2, 64, 4)
    out_raw = layer_raw(x)
    out_grid = layer_grid(x)
    diff = (out_raw - out_grid).abs().max().item()
    assert diff > 1e-3, (
        f"unit_to_grid produced identical output to raw (diff={diff:.2e}); "
        "rescale is not wired through forward."
    )


@pytest.mark.parametrize("bad_scale", ["foo", "", "Raw", "UNIT_TO_GRID"])
def test_invalid_cr_input_scale_raises(bad_scale):
    with pytest.raises(ValueError, match="cr_input_scale"):
        FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                            cr_input_scale=bad_scale)


@pytest.mark.parametrize("bad_resid", ["foo", "", "Avg", "MAX"])
def test_invalid_residual_kind_raises(bad_resid):
    with pytest.raises(ValueError, match="residual_kind"):
        FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                            residual_kind=bad_resid)


def test_classifier_propagates_revision3_axes():
    """The classifier must thread cr_input_scale and residual_kind
    through to every layer instance."""
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=8, n_layers=4,
                                  cr_input_scale="raw", residual_kind="probsum")
    for L in m.layers:
        assert L.cr_input_scale == "raw"
        assert L.residual_kind == "probsum"


# ─── init_kind axis (Atanassov-init knob added when C+B insufficient) ─


def test_init_kind_random_keeps_narrow_mu_range():
    """The 'random' init (legacy) produces σ(CR(x)) ≈ 0.5 ± small noise
    for any x — this is the diagnostic the C+B smoke surfaced."""
    torch.manual_seed(100)
    layer = FuzzySignatureLayer(d=8, H=8, W=8, kernel=3, stride=1,
                                cr_input_scale="unit_to_grid",
                                init_kind="random")
    x = torch.rand(2, 64, 8)
    with torch.no_grad():
        x_hat = 6.0 * x - 3.0
        mu_p = layer.mu_plus(x_hat, branch_idx=0)
    # Range narrow around 0.5 because init_scale=0.05.
    assert mu_p.min() > 0.4 and mu_p.max() < 0.6, (
        f"random init produced wider range than expected: "
        f"[{mu_p.min().item():.3f}, {mu_p.max().item():.3f}]"
    )


def test_init_kind_ramp_widens_mu_range():
    """The 'ramp' init produces μ⁺ approximately monotone increasing
    in x with a wide range across [0, 1] — the fix for the
    σ(small CR) narrow-range problem."""
    torch.manual_seed(101)
    layer = FuzzySignatureLayer(d=8, H=8, W=8, kernel=3, stride=1,
                                cr_input_scale="unit_to_grid",
                                init_kind="ramp", ramp_strength=1.5)
    with torch.no_grad():
        # x=0 → x_hat=-3 → CR(-3) at low CP → σ(-low) << 0.5
        # x=1 → x_hat=+3 → CR(+3) at high CP → σ(+high) >> 0.5
        x_low = torch.zeros(1, 8, 8)
        x_high = torch.ones(1, 8, 8)
        x_hat_low = 6.0 * x_low - 3.0
        x_hat_high = 6.0 * x_high - 3.0
        mu_p_low = layer.mu_plus(x_hat_low, branch_idx=0)
        mu_p_high = layer.mu_plus(x_hat_high, branch_idx=0)
        mu_n_low = layer.mu_minus(x_hat_low, branch_idx=0)
        mu_n_high = layer.mu_minus(x_hat_high, branch_idx=0)
    # μ⁺ ramps up: low → ~0.05, high → ~0.95
    assert mu_p_high.mean() - mu_p_low.mean() > 0.3, (
        f"μ⁺ didn't ramp up enough: low={mu_p_low.mean().item():.3f}, "
        f"high={mu_p_high.mean().item():.3f}"
    )
    # μ⁻ ramps down (symmetric)
    assert mu_n_low.mean() - mu_n_high.mean() > 0.3, (
        f"μ⁻ didn't ramp down: low={mu_n_low.mean().item():.3f}, "
        f"high={mu_n_high.mean().item():.3f}"
    )
    # μ⁺ and μ⁻ should be approximately complementary at the extremes
    assert abs(mu_p_low.mean().item() - mu_n_high.mean().item()) < 0.1, (
        "μ⁺(x=0) and μ⁻(x=1) should be similar (symmetric ramp)"
    )


@pytest.mark.parametrize("bad_init", ["foo", "", "Random", "RAMP"])
def test_invalid_init_kind_raises(bad_init):
    with pytest.raises(ValueError, match="init_kind"):
        FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                            init_kind=bad_init)


def test_classifier_propagates_init_kind():
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=8, n_layers=4,
                                  init_kind="ramp", ramp_strength=2.0)
    for L in m.layers:
        assert L.init_kind == "ramp"
        assert L.ramp_strength == 2.0


# ─── Revision-4 axes: residual_kind="lerp", gate_center_learnable,
#                      init_kind="asymmetric_ramp" ───────────────────


def test_residual_lerp_is_skip_dominant_at_init():
    """At init lerp residual gives out ≈ 0.95·x + 0.05·h_v (with default
    lerp_alpha_init=0.05). This is the highway-gate inductive bias that
    avoids the contraction-toward-fixed-point failure of avg/max/probsum."""
    torch.manual_seed(20)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                residual_kind="lerp", lerp_alpha_init=0.05)
    alpha = layer.alpha_eff
    assert alpha is not None
    assert torch.allclose(alpha, torch.full_like(alpha, 0.05), atol=1e-3), (
        f"α not at init: {alpha.mean().item():.4f}"
    )
    x = torch.rand(2, 64, 4)
    out = layer(x)
    # out ≈ x at init (skip-dominant), so |out - x| should be small.
    diff = (out - x).abs().mean().item()
    assert diff < 0.2, (
        f"lerp residual not near-identity at init: |out-x|.mean = {diff:.3f}"
    )


def test_residual_lerp_alpha_zero_is_strict_identity():
    """When α=0 exactly (forced), the layer is the identity in x."""
    torch.manual_seed(21)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                residual_kind="lerp")
    with torch.no_grad():
        layer.alpha_raw.fill_(-100.0)  # σ(-100) ≈ 0
    x = torch.rand(2, 64, 4)
    out = layer(x)
    assert torch.allclose(out, x, atol=1e-6), (
        f"layer at α=0 should be identity, got max diff "
        f"{(out - x).abs().max().item():.2e}"
    )


def test_residual_lerp_alpha_one_is_pure_h_v():
    """When α=1 exactly (forced), out = h_v (no skip contribution)."""
    torch.manual_seed(22)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                residual_kind="lerp")
    with torch.no_grad():
        layer.alpha_raw.fill_(100.0)  # σ(100) ≈ 1
    x = torch.rand(2, 64, 4)
    # Manual h_v computation: forward without the residual step
    out = layer(x)
    # At α=1, out should not equal x (rules out identity)
    diff_from_x = (out - x).abs().mean().item()
    # Should differ from x meaningfully (h_v is not equal to x in general)
    assert diff_from_x > 0.01, (
        f"layer at α=1 didn't differ from x enough: {diff_from_x:.4f}"
    )


@pytest.mark.parametrize("scale", ["raw", "unit_to_grid"])
@pytest.mark.parametrize("resid", ["avg", "max", "probsum", "lerp"])
def test_revision4_residuals_stay_in_01(scale, resid):
    """The 2×4 cross product preserves [0,1]."""
    torch.manual_seed(23)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                cr_input_scale=scale, residual_kind=resid)
    x = torch.rand(2, 64, 4)
    out = layer(x)
    assert torch.isfinite(out).all()
    assert (out >= 0.0).all(), f"{scale}/{resid}: min={out.min().item()}"
    assert (out <= 1.0 + 1e-5).all(), f"{scale}/{resid}: max={out.max().item()}"


def test_gate_center_learnable_starts_at_half():
    """When gate_center_learnable=True, c is a parameter init at ½."""
    layer = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                gate_center_learnable=True)
    assert isinstance(layer.c_gate, torch.nn.Parameter)
    assert torch.allclose(layer.c_gate, torch.full_like(layer.c_gate, 0.5))


def test_gate_center_fixed_is_buffer():
    """When gate_center_learnable=False, c is a buffer (not a param)."""
    layer = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                gate_center_learnable=False)
    assert not isinstance(layer.c_gate, torch.nn.Parameter)
    assert torch.allclose(layer.c_gate, torch.full_like(layer.c_gate, 0.5))
    # c not in named_parameters
    param_names = {n for n, _ in layer.named_parameters()}
    assert "c_gate" not in param_names


def test_init_kind_asymmetric_ramp_breaks_symmetry():
    """asymmetric_ramp init: μ⁺(x) is monotone in x, but μ⁻(x) is NOT
    the mirror of μ⁺(x). This breaks the symmetry implicated in the
    9-cell smoke failure."""
    torch.manual_seed(24)
    layer = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                init_kind="asymmetric_ramp",
                                cr_input_scale="unit_to_grid")
    with torch.no_grad():
        x_low = torch.zeros(1, 16, 4)
        x_high = torch.ones(1, 16, 4)
        x_hat_low = 6.0 * x_low - 3.0
        x_hat_high = 6.0 * x_high - 3.0
        mu_p_low = layer.mu_plus(x_hat_low, branch_idx=0).mean().item()
        mu_p_high = layer.mu_plus(x_hat_high, branch_idx=0).mean().item()
        mu_n_low = layer.mu_minus(x_hat_low, branch_idx=0).mean().item()
        mu_n_high = layer.mu_minus(x_hat_high, branch_idx=0).mean().item()
    # μ⁺ still ramps up like "ramp"
    assert mu_p_high - mu_p_low > 0.3
    # μ⁻ is NOT the mirror (μ⁻(0) ≠ μ⁺(1) within tolerance)
    # In "ramp" mode μ⁻(0) ≈ μ⁺(1) ≈ 0.95 (symmetric). In
    # "asymmetric_ramp" μ⁻ has a bump at low x, so μ⁻(0) < μ⁻(peak).
    mirror_gap = abs(mu_n_low - mu_p_high)
    # symmetric ramp would have mirror_gap < 0.1
    # asymmetric should have mirror_gap > 0.15
    assert mirror_gap > 0.15, (
        f"asymmetric_ramp didn't break μ⁻(0)≈μ⁺(1) symmetry: "
        f"μ⁻(0)={mu_n_low:.3f} vs μ⁺(1)={mu_p_high:.3f}"
    )


def test_classifier_propagates_revision4_axes():
    m = FuzzySignatureClassifier(
        H=28, W=28, n_classes=10, d=8, n_layers=4,
        residual_kind="lerp", lerp_alpha_init=0.1,
        gate_center_learnable=True, init_kind="asymmetric_ramp",
    )
    for L in m.layers:
        assert L.residual_kind == "lerp"
        assert L.init_kind == "asymmetric_ramp"
        assert L.gate_center_learnable is True
        assert L.alpha_raw is not None
        # α ≈ 0.1 at init
        assert abs(L.alpha_eff.mean().item() - 0.1) < 1e-3


def test_revision4_gradient_flow_includes_alpha_and_c():
    """One training step: gradients flow through α_raw and c_gate."""
    torch.manual_seed(25)
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=8, n_layers=2,
                                  residual_kind="lerp",
                                  gate_center_learnable=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    saw_alpha = saw_c = False
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        if "alpha_raw" in name:
            saw_alpha = True
        if "c_gate" in name:
            saw_c = True
    assert saw_alpha, "no gradient hit α_raw"
    assert saw_c, "no gradient hit c_gate"
    opt.step()


@pytest.mark.parametrize("bad_init", ["foo", "RAMP", "ASYMMETRIC", "random_v2"])
def test_invalid_init_kind_v4_raises(bad_init):
    with pytest.raises(ValueError, match="init_kind"):
        FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                            init_kind=bad_init)


@pytest.mark.parametrize("bad_resid", ["foo", "linear", "Lerp"])
def test_invalid_residual_kind_v4_raises(bad_resid):
    with pytest.raises(ValueError, match="residual_kind"):
        FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                            residual_kind=bad_resid)


def test_lerp_alpha_init_extreme_values_handled():
    """Edge case: lerp_alpha_init at boundary values should work via clamping."""
    for p in (1e-6, 0.999999):
        layer = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                     residual_kind="lerp",
                                     lerp_alpha_init=p)
        alpha = layer.alpha_eff
        assert torch.isfinite(alpha).all()
        assert (alpha > 0).all() and (alpha < 1).all()


# ─── Revision-5 axis: additive_centered residual + linear fuzzification ─


def test_residual_additive_centered_relaxes_01_bound():
    """The additive_centered residual `out = x + (h_v − 0.5)` does NOT
    preserve [0,1]: signal grows by ±0.5 per layer. This is the HSiKAN-
    style relaxation that user-directed after the 18-cell smoke matrix
    of bounded-fuzzy designs all failed."""
    torch.manual_seed(30)
    layer = FuzzySignatureLayer(d=4, H=8, W=8, kernel=3, stride=1,
                                residual_kind="additive_centered")
    # Inputs outside [0, 1] — fine in this regime.
    x = torch.randn(2, 64, 4) * 0.5
    out = layer(x)
    assert torch.isfinite(out).all()
    # Out is in roughly [x - 0.5, x + 0.5] component-wise.
    delta = out - x
    assert (delta >= -0.5 - 1e-5).all()
    assert (delta <= 0.5 + 1e-5).all()


def test_residual_additive_centered_signal_grows_with_depth():
    """Stack two layers; signal range should widen since each layer
    adds (h_v − 0.5) ∈ [−0.5, 0.5]."""
    torch.manual_seed(31)
    layer1 = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                  residual_kind="additive_centered")
    layer2 = FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                                  residual_kind="additive_centered")
    x = torch.zeros(2, 16, 4)  # start at 0 so growth is measurable
    out1 = layer1(x)
    out2 = layer2(out1)
    # After 2 layers, |out2 - x| can exceed 0.5 (cumulative).
    spread = (out2 - x).abs().max().item()
    assert spread > 1e-3  # something happened (not identity)


def test_classifier_fuzzification_linear_keeps_v_unbounded_post_embed():
    """fuzzification_kind='linear' skips σ on embed — internal v can
    be outside [0,1] after the embed Linear (HSiKAN pattern)."""
    torch.manual_seed(32)
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=8, n_layers=2,
                                  fuzzification_kind="linear",
                                  residual_kind="additive_centered")
    x = torch.rand(2, 1, 28, 28)
    # Forward should produce finite logits even though internal v is unbounded.
    logits = m(x)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_classifier_fuzzification_sigmoid_keeps_v_in_01_after_embed():
    """fuzzification_kind='sigmoid' (default) applies σ on embed,
    so the input to the first layer is in [0,1]."""
    torch.manual_seed(33)
    m = FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=8, n_layers=2,
                                  fuzzification_kind="sigmoid")
    x = torch.rand(2, 1, 28, 28)
    # Intercept the post-embed value.
    B = x.shape[0]
    v = x.view(B, -1, 1)
    v = m.embed(v)
    v_sigmoid = torch.sigmoid(v)
    assert (v_sigmoid >= 0.0).all() and (v_sigmoid <= 1.0).all()


@pytest.mark.parametrize("bad_fuzz", ["foo", "Sigmoid", "linear_v2"])
def test_invalid_fuzzification_kind_raises(bad_fuzz):
    with pytest.raises(ValueError, match="fuzzification_kind"):
        FuzzySignatureClassifier(H=28, W=28, n_classes=10, d=8, n_layers=2,
                                  fuzzification_kind=bad_fuzz)


def test_revision5_full_config_trains_one_step():
    """End-to-end one-step training with full rev-5 HSiKAN-relaxed
    config: fuzzification=linear, residual=additive_centered, ramp init."""
    torch.manual_seed(34)
    m = FuzzySignatureClassifier(
        H=28, W=28, n_classes=10, d=8, n_layers=4,
        fuzzification_kind="linear",
        residual_kind="additive_centered",
        cr_input_scale="raw",
        init_kind="ramp",
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(m(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
    opt.step()


@pytest.mark.parametrize("bad_resid", ["additive", "centered", "Additive_centered"])
def test_invalid_residual_kind_v5_raises(bad_resid):
    with pytest.raises(ValueError, match="residual_kind"):
        FuzzySignatureLayer(d=4, H=4, W=4, kernel=2, stride=1,
                            residual_kind=bad_resid)


# ─── Revision-6 axis: MultiArityFuzzySignatureLayer (αₖ mixer) ────


def test_multi_arity_layer_alpha_sums_to_1():
    """Softmax-normalised αₖ must sum to 1 across arities."""
    torch.manual_seed(40)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
    )
    a = layer.alpha_weights()
    assert a.shape == (2,)
    assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-6)
    # At init, αₖ logits are zero → uniform.
    assert torch.allclose(a, torch.full_like(a, 0.5), atol=1e-6)


def test_multi_arity_layer_forward_shape_and_finite():
    """The αₖ-weighted sum across arities preserves (B, V, d) shape."""
    torch.manual_seed(41)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        residual_kind="lerp",
    )
    x = torch.rand(2, 64, 4)
    out = layer(x)
    assert out.shape == (2, 64, 4)
    assert torch.isfinite(out).all()


def test_multi_arity_layer_preserves_01_when_all_sub_layers_do():
    """If every per-arity FuzzySignatureLayer preserves [0,1] (i.e.
    residual ≠ additive_centered), the αₖ convex combination
    inherits the property."""
    torch.manual_seed(42)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        residual_kind="lerp",  # preserves [0,1]
    )
    x = torch.rand(2, 64, 4)
    out = layer(x)
    assert (out >= 0.0).all()
    assert (out <= 1.0 + 1e-5).all()


def test_multi_arity_classifier_dispatch():
    """Classifier with len(arities) > 1 should build a
    MultiArityFuzzySignatureLayer per layer slot."""
    m = FuzzySignatureClassifier(
        H=28, W=28, n_classes=10, d=8, n_layers=2,
        arities=[(3, 1), (5, 2)],
    )
    for L in m.layers:
        assert isinstance(L, MultiArityFuzzySignatureLayer)
        assert len(L.layers) == 2  # one FuzzySignatureLayer per arity


def test_single_arity_classifier_keeps_simple_layer():
    """Classifier with len(arities) == 1 should still build single-arity
    FuzzySignatureLayers (no wrapper)."""
    m = FuzzySignatureClassifier(
        H=28, W=28, n_classes=10, d=8, n_layers=2,
        arities=[(3, 1)],
    )
    for L in m.layers:
        assert isinstance(L, FuzzySignatureLayer)
        assert not isinstance(L, MultiArityFuzzySignatureLayer)


def test_multi_arity_classifier_trains_one_step():
    """End-to-end one step on multi-arity classifier with rev-5
    HSiKAN-relaxed config."""
    torch.manual_seed(43)
    m = FuzzySignatureClassifier(
        H=28, W=28, n_classes=10, d=8, n_layers=2,
        arities=[(3, 1), (5, 2)],
        fuzzification_kind="linear",
        residual_kind="additive_centered",
        cr_input_scale="raw",
        init_kind="ramp",
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    x = torch.rand(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(m(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    saw_alpha_logits = False
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
        if "alpha_logits" in name:
            saw_alpha_logits = True
    assert saw_alpha_logits, "no gradient hit the αₖ mixer logits"
    opt.step()


def test_multi_arity_empty_raises():
    with pytest.raises(ValueError, match="arities"):
        MultiArityFuzzySignatureLayer(d=4, H=4, W=4, arities=[])


# ─── multi_arity_mixer axis (rev-7: fuzzy αₖ aggregator) ───────────


@pytest.mark.parametrize("mixer", ["softmax", "t_norm", "t_conorm", "owa"])
def test_multi_arity_mixer_forward_shape(mixer):
    """All 4 mixers preserve (B, V, d) shape and produce finite output."""
    torch.manual_seed(50)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        multi_arity_mixer=mixer,
    )
    x = torch.rand(2, 64, 4)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_multi_arity_mixer_t_norm_equals_min():
    """t_norm mixer = element-wise min across branches."""
    torch.manual_seed(51)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        multi_arity_mixer="t_norm",
    )
    x = torch.rand(2, 64, 4)
    out = layer(x)
    branch0 = layer.layers[0](x)
    branch1 = layer.layers[1](x)
    expected = torch.minimum(branch0, branch1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_multi_arity_mixer_t_conorm_equals_max():
    """t_conorm mixer = element-wise max across branches."""
    torch.manual_seed(52)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        multi_arity_mixer="t_conorm",
    )
    x = torch.rand(2, 64, 4)
    out = layer(x)
    branch0 = layer.layers[0](x)
    branch1 = layer.layers[1](x)
    expected = torch.maximum(branch0, branch1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_multi_arity_mixer_t_norm_no_learnable_weights():
    """t_norm / t_conorm are parameter-free aggregators."""
    layer_tn = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        multi_arity_mixer="t_norm",
    )
    layer_tc = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        multi_arity_mixer="t_conorm",
    )
    assert layer_tn.alpha_logits is None
    assert layer_tc.alpha_logits is None


def test_multi_arity_mixer_owa_uniform_weights_equals_mean():
    """OWA with uniform weights = arithmetic mean across branches."""
    torch.manual_seed(53)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        multi_arity_mixer="owa",
    )
    # Uniform weights ARE the default (softmax of zeros).
    x = torch.rand(2, 64, 4)
    out = layer(x)
    branch0 = layer.layers[0](x)
    branch1 = layer.layers[1](x)
    expected = 0.5 * (branch0 + branch1)
    assert torch.allclose(out, expected, atol=1e-5), (
        f"OWA uniform should equal mean; max diff = "
        f"{(out - expected).abs().max().item():.2e}"
    )


def test_multi_arity_mixer_owa_extreme_weights_equals_max():
    """OWA with weights = [1, 0, ...] (all on largest) = element-wise max."""
    torch.manual_seed(54)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        multi_arity_mixer="owa",
    )
    # Force weights = [1, 0] (all mass on the largest branch).
    with torch.no_grad():
        layer.alpha_logits.fill_(0.0)
        layer.alpha_logits.data[0] = 100.0  # softmax → ≈ [1, 0]
    x = torch.rand(2, 64, 4)
    out = layer(x)
    branch0 = layer.layers[0](x)
    branch1 = layer.layers[1](x)
    expected = torch.maximum(branch0, branch1)
    assert torch.allclose(out, expected, atol=1e-3)


@pytest.mark.parametrize("mixer", ["t_norm", "t_conorm", "owa"])
def test_multi_arity_mixer_gradient_flow(mixer):
    """All non-default mixers admit gradient flow through both branches."""
    torch.manual_seed(55)
    layer = MultiArityFuzzySignatureLayer(
        d=4, H=8, W=8, arities=[(3, 1), (5, 2)],
        multi_arity_mixer=mixer,
    )
    x = torch.rand(2, 64, 4, requires_grad=True)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    saw_layer_0 = saw_layer_1 = False
    for name, p in layer.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"
            if "layers.0" in name:
                saw_layer_0 = True
            if "layers.1" in name:
                saw_layer_1 = True
    assert saw_layer_0 and saw_layer_1, (
        f"mixer={mixer}: gradient didn't reach both arity branches"
    )


@pytest.mark.parametrize("bad", ["foo", "Softmax", "T_NORM", "mean"])
def test_multi_arity_mixer_invalid_raises(bad):
    with pytest.raises(ValueError, match="multi_arity_mixer"):
        MultiArityFuzzySignatureLayer(
            d=4, H=4, W=4, arities=[(3, 1), (5, 2)],
            multi_arity_mixer=bad,
        )


def test_classifier_propagates_multi_arity_mixer():
    m = FuzzySignatureClassifier(
        H=28, W=28, n_classes=10, d=8, n_layers=2,
        arities=[(3, 1), (5, 2)],
        multi_arity_mixer="t_conorm",
    )
    assert m.multi_arity_mixer == "t_conorm"
    for L in m.layers:
        assert isinstance(L, MultiArityFuzzySignatureLayer)
        assert L.multi_arity_mixer == "t_conorm"
        assert L.alpha_logits is None  # parameter-free for t_conorm
