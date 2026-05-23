"""Unit tests for the Phase-21 side-stacked mixed-arity HSIKAN.

Pins the parallel-branch wrapper's behaviour over MixedAritySignedKAN:
forward / backward, N=1 parity at fixed seed, parameter count scaling,
alpha-shape, classifier API, and shape preservation under
``fusion in {"mean", "sum"}``.
"""
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


def _build_arity_inputs(n_nodes=8, n_triads=6, k=3, n_edges=5,
                         device=torch.device("cpu")):
    """Build a single-arity per_arity_inputs tuple for testing."""
    rng = np.random.default_rng(0)
    triad_v_np = rng.integers(0, n_nodes, size=(n_triads, k))
    triad_v_np.sort(axis=1)
    # Ensure unique vertices per triad.
    for i in range(n_triads):
        while len(set(triad_v_np[i])) < k:
            triad_v_np[i] = sorted(rng.integers(0, n_nodes, size=k))
    triad_sigma_np = rng.choice([-1, 1], size=(n_triads, k))
    triad_v = torch.from_numpy(triad_v_np).long().to(device)
    triad_sigma = torch.from_numpy(triad_sigma_np).long().to(device)
    M_vt = build_vertex_triad_incidence(triad_v_np, n_nodes, device,
                                          mode="sum")
    # Build a simple uniform edge-incidence M_e (E x T): each query
    # edge connects to a fixed subset of triads.
    rows, cols, vals = [], [], []
    for ei in range(n_edges):
        # Each edge attaches to ~half the triads, uniform weight.
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
    # Build random query edges (u, v).
    q_edges = rng.integers(0, n_nodes, size=(n_edges, 2))
    q_edges = torch.from_numpy(q_edges).long().to(device)
    return [(triad_v, triad_sigma, M_vt, M_e)], q_edges


def _mixed_arity_cfg(n_nodes=8, hidden=4, n_layers=2, arities=(3,)):
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
    )


def test_forward_shape_mean_fusion():
    """Mean fusion preserves the bare ``(E, d_jk)`` shape."""
    per_arity, q_edges = _build_arity_inputs()
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    cfg = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=3,
                                          fusion="mean")
    model = SideMixedAritySignedKAN(cfg)
    edge_emb = model.encode_edges(per_arity, query_edges=q_edges)
    d_jk = 4 * 2  # hidden_dim * n_layers under jk_mode='concat'
    assert edge_emb.shape == (5, d_jk), \
        f"got {edge_emb.shape}, expected (5, {d_jk})"


def test_forward_shape_sum_fusion():
    """Sum fusion also preserves the bare ``(E, d_jk)`` shape."""
    per_arity, q_edges = _build_arity_inputs()
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    cfg = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=3,
                                          fusion="sum")
    model = SideMixedAritySignedKAN(cfg)
    edge_emb = model.encode_edges(per_arity, query_edges=q_edges)
    assert edge_emb.shape == (5, 8)


def test_classifier_input_dim_matches_d_jk():
    """The wrapper-level classifier accepts the fused d_jk vector."""
    per_arity, q_edges = _build_arity_inputs()
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    cfg = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=3,
                                          fusion="mean")
    model = SideMixedAritySignedKAN(cfg)
    edge_emb = model.encode_edges(per_arity, query_edges=q_edges)
    logits = model.classifier(edge_emb).squeeze(-1)
    assert logits.shape == (5,)


def test_invalid_fusion_raises():
    base_cfg = _mixed_arity_cfg(hidden=4)
    bad = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=2,
                                          fusion="concat")
    try:
        SideMixedAritySignedKAN(bad)
    except ValueError as e:
        assert "fusion" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for fusion='concat'")


def test_param_count_scales_with_n_branches():
    """N=4 branches has ~4× the inner-mixed-arity parameter budget."""
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    m1 = SideMixedAritySignedKAN(SideMixedAritySignedKANConfig(
        base=base_cfg, n_branches=1, fusion="mean",
    ))
    m4 = SideMixedAritySignedKAN(SideMixedAritySignedKANConfig(
        base=base_cfg, n_branches=4, fusion="mean",
    ))
    inner = MixedAritySignedKAN(base_cfg).num_parameters()
    classifier = 1 * (4 * 2 + 1)   # Linear(d_jk → 1) bias + weight
    expected_m1 = inner + classifier
    expected_m4 = 4 * inner + classifier
    # Allow small slack from internal init: param counts must be exact.
    assert m1.num_parameters() == expected_m1, (
        f"N=1 param count {m1.num_parameters()} != "
        f"inner({inner}) + clf({classifier}) = {expected_m1}"
    )
    assert m4.num_parameters() == expected_m4, (
        f"N=4 param count {m4.num_parameters()} != "
        f"4·inner({inner}) + clf({classifier}) = {expected_m4}"
    )


def test_backward_passes_through_branches():
    """Gradients reach every branch's mixed-arity parameters."""
    per_arity, q_edges = _build_arity_inputs()
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    cfg = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=3,
                                          fusion="mean")
    model = SideMixedAritySignedKAN(cfg)
    edge_emb = model.encode_edges(per_arity, query_edges=q_edges)
    loss = edge_emb.sum()
    loss.backward()
    for i, branch in enumerate(model.branches):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in branch.parameters()
        )
        assert has_grad, f"branch {i} got no gradient"


def test_arity_logits_get_gradient():
    """Each branch's ``arity_logits`` learns its own αₖ. With two
    arities and a backward pass, gradients must reach all per-branch
    arity_logits."""
    per_arity_k3, q_edges = _build_arity_inputs(k=3)
    per_arity_k4, _ = _build_arity_inputs(k=4)
    per_arity = [per_arity_k3[0], per_arity_k4[0]]
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2, arities=(3, 4))
    cfg = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=3,
                                          fusion="mean")
    model = SideMixedAritySignedKAN(cfg)
    edge_emb = model.encode_edges(per_arity, query_edges=q_edges)
    loss = edge_emb.sum()
    loss.backward()
    for i, branch in enumerate(model.branches):
        assert branch.arity_logits.grad is not None, \
            f"branch {i} arity_logits got no gradient"
        assert branch.arity_logits.grad.abs().sum().item() > 0, \
            f"branch {i} arity_logits gradient is identically zero"


def test_alpha_method_returns_branch_mean():
    """``model.alpha()`` is the mean of each branch's softmaxed α."""
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2, arities=(3, 4))
    cfg = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=3,
                                          fusion="mean")
    model = SideMixedAritySignedKAN(cfg)
    a = model.alpha()
    # Shape matches a single branch's α.
    assert a.shape == (2,), f"alpha shape {a.shape} != (2,)"
    # Should sum to ≈ 1 (mean of softmax outputs).
    assert abs(a.sum().item() - 1.0) < 1e-5


def test_n_branches_1_matches_bare_mixed_arity_at_same_init():
    """N=1 + fusion='mean' (or 'sum') must be functionally equivalent
    to a bare ``MixedAritySignedKAN`` at the same init."""
    torch.manual_seed(42)
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    bare = MixedAritySignedKAN(base_cfg)
    torch.manual_seed(42)
    cfg = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=1,
                                          fusion="mean")
    side = SideMixedAritySignedKAN(cfg)
    per_arity, q_edges = _build_arity_inputs()
    out_bare = bare.encode_edges(per_arity, query_edges=q_edges)
    out_side = side.encode_edges(per_arity, query_edges=q_edges)
    diff = (out_bare - out_side).abs().max().item()
    assert diff < 1e-5, (
        f"N=1 mean-fusion should reproduce bare MixedArity at same init; "
        f"max diff = {diff:.2e}"
    )


def test_node_embed_passthrough():
    """``model.node_embed`` proxies to the first branch's node embed."""
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    cfg = SideMixedAritySignedKANConfig(base=base_cfg, n_branches=2,
                                          fusion="mean")
    model = SideMixedAritySignedKAN(cfg)
    assert model.node_embed is model.branches[0].node_embed
    # Sanity: weight is a tensor.
    assert model.node_embed.weight.shape[0] > 0


def test_collect_attn_entropy_false_skips_collection():
    """When ``collect_attn_entropy=False`` is passed to a bare
    ``MixedAritySignedKAN.encode_edges`` the per-branch entropy
    accumulator is set to ``None`` (skipping the autograd graph that
    backs each entropy scalar)."""
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    bare = MixedAritySignedKAN(base_cfg)
    per_arity, q_edges = _build_arity_inputs()
    _ = bare.encode_edges(per_arity, query_edges=q_edges,
                            collect_attn_entropy=False)
    assert bare._attn_entropy_terms is None
    # And on the default path (True), the list is populated.
    _ = bare.encode_edges(per_arity, query_edges=q_edges,
                            collect_attn_entropy=True)
    assert bare._attn_entropy_terms is not None


def test_outer_grad_checkpoint_numerical_parity():
    """``outer_grad_checkpoint=True`` must produce the same output as
    the bare path at fixed init, modulo recomputation noise. We test
    at N=2 in training mode (where the gate is live)."""
    torch.manual_seed(7)
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    cfg_off = SideMixedAritySignedKANConfig(
        base=base_cfg, n_branches=2, fusion="mean",
        use_grad_checkpoint=False, outer_grad_checkpoint=False,
    )
    torch.manual_seed(99)
    side_off = SideMixedAritySignedKAN(cfg_off)
    side_off.train()

    torch.manual_seed(7)
    cfg_on = SideMixedAritySignedKANConfig(
        base=base_cfg, n_branches=2, fusion="mean",
        use_grad_checkpoint=False, outer_grad_checkpoint=True,
    )
    torch.manual_seed(99)
    side_on = SideMixedAritySignedKAN(cfg_on)
    side_on.train()

    per_arity, q_edges = _build_arity_inputs()
    out_off = side_off.encode_edges(per_arity, query_edges=q_edges)
    out_on = side_on.encode_edges(per_arity, query_edges=q_edges)
    diff = (out_off - out_on).abs().max().item()
    assert diff < 1e-4, (
        f"outer_grad_checkpoint should be numerically equivalent; "
        f"max diff = {diff:.2e}"
    )


def test_outer_grad_checkpoint_backward_reaches_branches():
    """With ``outer_grad_checkpoint=True`` backward must still
    populate every branch's parameter gradients (the whole point of
    the recompute path)."""
    base_cfg = _mixed_arity_cfg(hidden=4, n_layers=2)
    cfg = SideMixedAritySignedKANConfig(
        base=base_cfg, n_branches=3, fusion="mean",
        use_grad_checkpoint=False, outer_grad_checkpoint=True,
    )
    model = SideMixedAritySignedKAN(cfg)
    model.train()
    per_arity, q_edges = _build_arity_inputs()
    edge_emb = model.encode_edges(per_arity, query_edges=q_edges)
    loss = edge_emb.sum()
    loss.backward()
    for i, branch in enumerate(model.branches):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in branch.parameters()
        )
        assert has_grad, f"branch {i} got no gradient under outer ckpt"
