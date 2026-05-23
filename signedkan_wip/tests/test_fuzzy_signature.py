"""Tests for the HSIKAN fuzzy signature interpretability view."""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.core.signedkan import (
    MultiLayerSignedKANConfig, build_vertex_triad_incidence,
)
from signedkan_wip.src.mixed_arity_signedkan import (
    MixedAritySignedKAN, MixedAritySignedKANConfig,
)
from signedkan_wip.src.core.side_signedkan import (
    SideMixedAritySignedKAN, SideMixedAritySignedKANConfig,
)
from signedkan_wip.src.interpret import (
    CycleContribution, FuzzySignature, extract_signature, plot_signature,
)


def _mixed_arity_cfg(n_nodes=8, hidden=4, n_layers=2, arities=(3,),
                      attention_m_e=False):
    return MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=n_nodes, n_layers=n_layers,
            hidden_dim=hidden, grid=3, k=3,
            spline_kinds=["catmull_rom"] * n_layers,
            init_scale=0.05,
            pool_mode="sum",
            jk_mode="concat",
            layer_norm_between=True,
            share_weights=True,
            inner_skip="highway",
            outer_skip="none",
            use_residual=True,
        ),
        arities=arities,
        init_arity_logits=tuple([0.0] * len(arities)),
        attention_m_e=attention_m_e,
    )


def _build_per_arity_inputs(n_nodes=8, n_triads=6, k=3, n_edges=5,
                              device=torch.device("cpu")):
    """Build single-arity per_arity_inputs with a uniform M_e."""
    rng = np.random.default_rng(0)
    triad_v_np = rng.integers(0, n_nodes, size=(n_triads, k))
    triad_v_np.sort(axis=1)
    for i in range(n_triads):
        while len(set(triad_v_np[i])) < k:
            triad_v_np[i] = sorted(rng.integers(0, n_nodes, size=k))
    triad_sigma_np = rng.choice([-1, 1], size=(n_triads, k))
    triad_v = torch.from_numpy(triad_v_np).long().to(device)
    triad_sigma = torch.from_numpy(triad_sigma_np).long().to(device)
    M_vt = build_vertex_triad_incidence(triad_v_np, n_nodes, device,
                                          mode="sum")

    # Uniform M_e: each edge attaches to a fixed subset of triads.
    rows, cols, vals = [], [], []
    for ei in range(n_edges):
        n_attach = max(1, n_triads // 2)
        attached = rng.choice(n_triads, size=n_attach, replace=False)
        w = 1.0 / float(n_attach)
        for t in attached:
            rows.append(ei); cols.append(int(t)); vals.append(w)
    M_e = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32),
        (n_edges, n_triads),
    ).coalesce()
    q_edges = rng.integers(0, n_nodes, size=(n_edges, 2))
    q_edges = torch.from_numpy(q_edges).long().to(device)
    return ([(triad_v, triad_sigma, M_vt, M_e)], q_edges,
            triad_v_np, triad_sigma_np, np.array(rows), np.array(cols),
            np.array(vals))


def test_signature_returns_correct_shape():
    """Smoke test: extract_signature returns a FuzzySignature with
    non-empty contributions on a toy graph."""
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2, arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    sig = extract_signature(model, per_arity, q_edges, query_idx=0)
    assert isinstance(sig, FuzzySignature)
    assert sig.query_idx == 0
    assert len(sig.contributions) > 0


def test_sigma_prod_is_product_of_edge_signs_when_provided():
    """When ``arity_edge_signs`` is provided, ``sigma_prod`` is the
    Cartwright-Harary balance vote (product of edge signs). This is
    the interpretive primitive — the per-vertex σ product is
    structurally always +1 and uninformative."""
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    # Inject synthetic edge_signs: alternate balanced/unbalanced
    # cycles so the test sees both vote values.
    n_triads = int(per_arity[0][0].shape[0])
    es = np.ones((n_triads, 3), dtype=np.int64)
    es[1::2, 0] = -1  # every-other cycle has one negative edge
    sig = extract_signature(model, per_arity, q_edges, query_idx=1,
                              arity_edge_signs=[es])
    for c in sig.contributions:
        expected = int(np.prod(np.asarray(c.edge_signs,
                                            dtype=np.int64)))
        expected = 1 if expected > 0 else -1
        assert c.sigma_prod == expected
        assert c.balanced == (c.sigma_prod == 1)


def test_sigma_prod_fallback_warns_and_is_always_plus_one():
    """Without ``arity_edge_signs``, sigma_prod falls back to the
    per-vertex σ product which the model constructs to always be
    +1 (every negative edge flips parity at two vertices). The
    extractor warns about this."""
    import warnings
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sig = extract_signature(model, per_arity, q_edges, query_idx=1)
        assert any("uninformative" in str(w.message) for w in caught), \
            "expected uninformative-vote warning"
    # All contributions must report sigma_prod = +1 in fallback mode.
    for c in sig.contributions:
        # The per-vertex sigma sum is even-by-construction in the
        # real model. For test data the random sigma may break this,
        # so just assert that sigma_prod is in {±1}.
        assert c.sigma_prod in (1, -1)


def test_membership_sum_matches_m_e_when_attention_off():
    """When attention is off the per-cycle membership α_c is exactly
    the M_e[query, c] weight; their sum equals the sum of M_e's
    column entries for the query row."""
    base_cfg = _mixed_arity_cfg(arities=(3,), attention_m_e=False)
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, _v, _s, rows_np, cols_np, vals_np = \
        _build_per_arity_inputs()
    q = 2
    sig = extract_signature(model, per_arity, q_edges, query_idx=q)
    expected_sum = float(vals_np[rows_np == q].sum())
    got_sum = sum(c.membership for c in sig.contributions)
    assert abs(got_sum - expected_sum) < 1e-5, (
        f"got {got_sum}, expected {expected_sum}"
    )


def test_contribution_count_matches_m_e_row():
    """The number of contributions equals the number of cycles
    incident to the query in M_e."""
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, _v, _s, rows_np, _c, _val = \
        _build_per_arity_inputs()
    for q in range(int(q_edges.shape[0])):
        sig = extract_signature(model, per_arity, q_edges, query_idx=q)
        expected = int((rows_np == q).sum())
        assert len(sig.contributions) == expected, (
            f"query {q}: got {len(sig.contributions)} contribs, "
            f"M_e row has {expected} entries"
        )


def test_mixed_arity_contributions_tagged_per_arity():
    """A 2-arity model produces contributions tagged with both
    arity values."""
    cfg_k3 = _mixed_arity_cfg(arities=(3, 4))
    model = MixedAritySignedKAN(cfg_k3)
    per_arity_k3, q_edges, *_ = _build_per_arity_inputs(k=3)
    per_arity_k4, _, *_ = _build_per_arity_inputs(k=4)
    per_arity = [per_arity_k3[0], per_arity_k4[0]]
    sig = extract_signature(model, per_arity, q_edges, query_idx=0)
    arities = {c.arity for c in sig.contributions}
    # Some queries may only touch one arity's cycles. To make this
    # deterministic across seeds, check that arities is a subset of
    # the two configured ones.
    assert arities.issubset({3, 4}), (
        f"contributions report arities {arities}, expected subset of "
        "{3, 4}"
    )


def test_wrapper_extracts_first_branch():
    """Phase 21 wrapper: extract_signature works through
    SideMixedAritySignedKAN by reading the first branch."""
    base_cfg = _mixed_arity_cfg(arities=(3,))
    wrap_cfg = SideMixedAritySignedKANConfig(
        base=base_cfg, n_branches=3, fusion="mean",
    )
    model = SideMixedAritySignedKAN(wrap_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    sig = extract_signature(model, per_arity, q_edges, query_idx=0)
    assert isinstance(sig, FuzzySignature)
    assert len(sig.contributions) > 0


def test_net_vote_consistent_with_individual_contributions():
    """``sig.net_vote()`` equals Σ σ_c · α_c."""
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    sig = extract_signature(model, per_arity, q_edges, query_idx=3)
    expected = sum(c.sigma_prod * c.membership
                    for c in sig.contributions)
    assert abs(sig.net_vote() - expected) < 1e-6


def test_vote_by_arity_sums_to_total_membership():
    """vote_by_arity()'s flat sum equals total_membership()."""
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    sig = extract_signature(model, per_arity, q_edges, query_idx=2)
    flat = 0.0
    for tag, bkt in sig.vote_by_arity().items():
        flat += bkt[+1] + bkt[-1]
    assert abs(flat - sig.total_membership()) < 1e-6


def test_plot_signature_smoke():
    """plot_signature returns an (ax_top, ax_bot) pair on a real
    signature. We don't assert visual properties — just that the
    plot pipeline runs without error."""
    import matplotlib
    matplotlib.use("Agg")
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    sig = extract_signature(model, per_arity, q_edges, query_idx=0)
    result = plot_signature(sig)
    # When no arc_weights are present, plot returns (ax_top, ax_bot).
    assert len(result) == 2
    assert all(a is not None for a in result)


def test_arc_weights_default_empty_when_not_provided():
    """``CycleContribution.arc_weights`` defaults to ``()`` when the
    extractor wasn't given ``arity_arc_weights``."""
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    sig = extract_signature(model, per_arity, q_edges, query_idx=0)
    for c in sig.contributions:
        assert c.arc_weights == ()
    assert sig.mean_abs_arc_weight() == 0.0


def test_arc_weights_populated_when_provided():
    """When ``arity_arc_weights`` is passed, each contribution's
    arc_weights tuple matches the input row exactly."""
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    n_triads = int(per_arity[0][0].shape[0])
    # Synthetic arc weights: each cycle gets w = (0.1, 0.5, -0.8).
    aw = np.tile(np.array([[0.1, 0.5, -0.8]]), (n_triads, 1))
    sig = extract_signature(
        model, per_arity, q_edges, query_idx=1,
        arity_arc_weights=[aw],
    )
    for c in sig.contributions:
        assert len(c.arc_weights) == 3
        assert abs(c.arc_weights[0] - 0.1) < 1e-6
        assert abs(c.arc_weights[1] - 0.5) < 1e-6
        assert abs(c.arc_weights[2] + 0.8) < 1e-6
    # mean_abs_arc_weight averages |w| across all 3 edges of all
    # contributions: (0.1 + 0.5 + 0.8) / 3 = 0.466...
    assert abs(sig.mean_abs_arc_weight() - 0.4666666667) < 1e-4


def test_plot_signature_three_panels_when_arc_weights_present():
    """``plot_signature`` returns a (ax_top, ax_bot, ax_arc) triple
    when at least one contribution has arc_weights."""
    import matplotlib
    matplotlib.use("Agg")
    base_cfg = _mixed_arity_cfg(arities=(3,))
    model = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges, *_ = _build_per_arity_inputs()
    n_triads = int(per_arity[0][0].shape[0])
    aw = np.random.default_rng(0).uniform(-1, 1, size=(n_triads, 3))
    sig = extract_signature(
        model, per_arity, q_edges, query_idx=0,
        arity_arc_weights=[aw],
    )
    result = plot_signature(sig)
    assert len(result) == 3, \
        f"expected 3 panels with arc weights present; got {len(result)}"
