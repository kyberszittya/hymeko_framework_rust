"""Unit tests for ``GombWithOuterHSIKAN`` — the outer HSIKAN
backbone variant that feeds Gömb's Clifford-FIR layer.

Pins:
- Construction requires ``outer_hsikan_n_layers >= 1``.
- Forward returns ``(E,)`` logits.
- Backward reaches both the outer HSIKAN's parameters AND every
  Gömb shell's parameters (the gradient must flow through the
  whole chain).
- The outer HSIKAN's output dim matches the FIR layer's input
  expectation (jk_mode='last' → d_embed; 'concat' → L·d_embed).
- Param count increases with outer-HSIKAN depth.
- The outer HSIKAN's ``node_embed`` is exposed for caller
  regularisers via ``model.node_embed``.
"""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.hymeko_gomb.cascade import (
    GombConfig, GombWithOuterHSIKAN, HymeKoGomb,
)


def _toy_cfg(n_nodes=12, outer_hsikan_n_layers=2,
              outer_hsikan_jk_mode="last"):
    return GombConfig(
        n_nodes=n_nodes,
        d_embed=8,
        d_outer=4, M_outer=2,
        d_middle=4,
        d_core=4, n_tiers=2,
        cycle_k=3, middle_grid=5,
        outer_hsikan_n_layers=outer_hsikan_n_layers,
        outer_hsikan_inner_skip="highway",
        outer_hsikan_jk_mode=outer_hsikan_jk_mode,
        outer_hsikan_share_weights=False,
    )


def _toy_inputs(n_nodes=12, n_cycles=8, cycle_k=3, n_query=4,
                  device=torch.device("cpu")):
    rng = np.random.default_rng(0)
    cycles_np = rng.integers(0, n_nodes, size=(n_cycles, cycle_k))
    cycles_np.sort(axis=1)
    for i in range(n_cycles):
        while len(set(cycles_np[i])) < cycle_k:
            cycles_np[i] = sorted(rng.integers(0, n_nodes, size=cycle_k))
    signs_np = rng.choice([-1, 1], size=(n_cycles, cycle_k))
    queries = [(int(cycles_np[i, 0]), int(cycles_np[i, 1]))
                for i in range(min(n_query, n_cycles))]
    while len(queries) < n_query:
        queries.append((int(cycles_np[0, 0]), int(cycles_np[0, 1])))
    cycles = torch.from_numpy(cycles_np).long().to(device)
    signs = torch.from_numpy(signs_np).float().to(device)
    tier_of = torch.zeros(n_nodes, dtype=torch.long, device=device)
    edges_to_score = torch.tensor(queries, dtype=torch.long,
                                    device=device)
    return cycles, signs, tier_of, edges_to_score


def test_construction_requires_outer_hsikan():
    """``outer_hsikan_n_layers=0`` is the marker for "use plain
    HymeKoGomb"; constructing GombWithOuterHSIKAN with 0 layers
    must raise."""
    cfg = _toy_cfg(outer_hsikan_n_layers=0)
    try:
        GombWithOuterHSIKAN(cfg)
    except ValueError as e:
        assert "outer_hsikan_n_layers" in str(e)
    else:
        raise AssertionError(
            "expected ValueError on outer_hsikan_n_layers=0"
        )


def test_forward_shape():
    """Forward returns (E,) edge sign logits."""
    cfg = _toy_cfg(outer_hsikan_n_layers=2)
    model = GombWithOuterHSIKAN(cfg)
    cycles, signs, tier_of, edges = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    assert out.shape == (edges.shape[0],)


def test_depth_4_runs():
    """``outer_hsikan_n_layers=4`` runs end-to-end."""
    cfg = _toy_cfg(outer_hsikan_n_layers=4)
    model = GombWithOuterHSIKAN(cfg)
    cycles, signs, tier_of, edges = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    assert out.shape == (edges.shape[0],)


def test_fir_input_dim_matches_d_embed():
    """The Clifford-FIR shell sees ``d_embed`` channels regardless
    of ``outer_hsikan_jk_mode`` (the stack's ``return_h_v`` always
    yields the last-layer per-vertex embedding at d_embed)."""
    for jk in ("last", "sum", "concat"):
        cfg = _toy_cfg(outer_hsikan_n_layers=3,
                         outer_hsikan_jk_mode=jk)
        model = GombWithOuterHSIKAN(cfg)
        assert model.outer.d_in == cfg.d_embed
        cycles, signs, tier_of, edges = _toy_inputs()
        out = model(cycles, signs, tier_of, edges)
        assert out.shape == (edges.shape[0],), \
            f"forward failed for jk_mode={jk}"


def test_backward_reaches_outer_hsikan_and_gomb_shells():
    """Backward flows to BOTH the outer HSIKAN's parameters AND
    every Gömb shell's parameters. This is the central property:
    the gradient through Clifford-FIR must drive both ends of the
    composite architecture."""
    torch.manual_seed(0)
    cfg = _toy_cfg(outer_hsikan_n_layers=2)
    model = GombWithOuterHSIKAN(cfg)
    cycles, signs, tier_of, edges = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    out.sum().backward()
    # Outer HSIKAN params get grad.
    hsikan_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.outer_hsikan.parameters()
    )
    assert hsikan_has_grad, "outer HSIKAN got no gradient"
    # FIR layer params get grad.
    fir_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.outer.parameters()
    )
    assert fir_has_grad, "Clifford-FIR shell got no gradient"
    # Middle shell params get grad.
    mid_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.middle.parameters()
    )
    assert mid_has_grad, "middle shell got no gradient"
    # Core params get grad.
    core_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.core.parameters()
    )
    assert core_has_grad, "inner CPML core got no gradient"


def test_param_count_grows_with_depth():
    """A deeper outer HSIKAN has more parameters."""
    cfg_d1 = _toy_cfg(outer_hsikan_n_layers=1)
    cfg_d4 = _toy_cfg(outer_hsikan_n_layers=4)
    m1 = GombWithOuterHSIKAN(cfg_d1)
    m4 = GombWithOuterHSIKAN(cfg_d4)
    assert m4.n_params() > m1.n_params(), (
        f"depth=4 ({m4.n_params()}) should have more params than "
        f"depth=1 ({m1.n_params()})"
    )


def test_node_embed_exposes_base_embedding():
    """Under the highway-gated residual composition,
    ``model.node_embed`` exposes the BASE embedding (the one
    Gömb's cascade actually consumes). The outer HSIKAN has its
    own internal embedding accessible via
    ``model.outer_hsikan.node_embed``."""
    cfg = _toy_cfg(outer_hsikan_n_layers=2)
    model = GombWithOuterHSIKAN(cfg)
    assert model.node_embed is model.base_node_embed
    # Sanity: outer HSIKAN owns a separate embedding too.
    assert model.outer_hsikan.node_embed is not model.base_node_embed


def test_gate_starts_low_so_plain_gomb_dominates_at_init():
    """The highway gate is biased low (sigmoid(-3) ≈ 0.05) at
    init so the model starts effectively as plain Gömb. As
    training progresses the gate can lift per-channel to use
    HSIKAN's refinement."""
    cfg = _toy_cfg(outer_hsikan_n_layers=2)
    model = GombWithOuterHSIKAN(cfg)
    g = torch.sigmoid(model.hsikan_gate_logit)
    # All channels should start near 0.05 (sigmoid(-3)).
    assert g.max().item() < 0.1, (
        f"gate init too high; max g = {g.max().item():.3f}"
    )
    assert g.min().item() > 0.01


def test_grad_checkpoint_forward_matches_no_checkpoint():
    """With ``outer_hsikan_grad_checkpoint=True`` the forward output
    should match the no-checkpoint forward bit-for-bit (the
    checkpoint API recomputes deterministically). Tested in eval
    mode so the checkpoint path bypass is symmetric."""
    torch.manual_seed(0)
    cfg = _toy_cfg(outer_hsikan_n_layers=3)
    cfg.outer_hsikan_grad_checkpoint = False
    m_off = GombWithOuterHSIKAN(cfg)
    torch.manual_seed(0)
    cfg.outer_hsikan_grad_checkpoint = True
    m_on = GombWithOuterHSIKAN(cfg)
    cycles, signs, tier_of, edges = _toy_inputs()
    m_off.eval(); m_on.eval()
    with torch.no_grad():
        out_off = m_off(cycles, signs, tier_of, edges)
        out_on = m_on(cycles, signs, tier_of, edges)
    diff = (out_off - out_on).abs().max().item()
    assert diff < 1e-5, f"checkpoint-on != off at eval; diff={diff:.2e}"


def test_grad_checkpoint_backward_works():
    """With grad-checkpoint enabled, backward must still flow to
    every parameter (the whole point of the recompute path)."""
    torch.manual_seed(0)
    cfg = _toy_cfg(outer_hsikan_n_layers=3)
    cfg.outer_hsikan_grad_checkpoint = True
    model = GombWithOuterHSIKAN(cfg)
    cycles, signs, tier_of, edges = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    out.sum().backward()
    # Outer HSIKAN node_embed.
    assert model.outer_hsikan.node_embed.weight.grad is not None
    assert model.outer_hsikan.node_embed.weight.grad.abs().sum() > 0
    # Highway gate.
    assert model.hsikan_gate_logit.grad is not None
    assert model.hsikan_gate_logit.grad.abs().sum() > 0
    # Base embedding (downstream Gömb path).
    assert model.base_node_embed.weight.grad is not None
    assert model.base_node_embed.weight.grad.abs().sum() > 0


def test_backward_reaches_base_embed_and_gate():
    """The gradient must flow to BOTH the base node embedding
    AND the highway gate (else the residual composition would
    be degenerate)."""
    torch.manual_seed(0)
    cfg = _toy_cfg(outer_hsikan_n_layers=2)
    model = GombWithOuterHSIKAN(cfg)
    cycles, signs, tier_of, edges = _toy_inputs()
    out = model(cycles, signs, tier_of, edges)
    out.sum().backward()
    assert model.base_node_embed.weight.grad is not None
    assert model.base_node_embed.weight.grad.abs().sum().item() > 0, \
        "base_node_embed got no gradient"
    assert model.hsikan_gate_logit.grad is not None
    assert model.hsikan_gate_logit.grad.abs().sum().item() > 0, \
        "hsikan_gate_logit got no gradient"


def test_vs_hymekogomb_has_more_params():
    """At the same base config, GombWithOuterHSIKAN has more
    parameters than HymeKoGomb (the outer HSIKAN backbone is
    additive overhead). Sanity that the new class actually adds
    capacity."""
    cfg_no = GombConfig(
        n_nodes=12, d_embed=8, d_outer=4, M_outer=2,
        d_middle=4, d_core=4, n_tiers=2,
        cycle_k=3, middle_grid=5,
        outer_hsikan_n_layers=0,
    )
    cfg_yes = GombConfig(
        n_nodes=12, d_embed=8, d_outer=4, M_outer=2,
        d_middle=4, d_core=4, n_tiers=2,
        cycle_k=3, middle_grid=5,
        outer_hsikan_n_layers=2,
    )
    g_no = HymeKoGomb(cfg_no)
    g_yes = GombWithOuterHSIKAN(cfg_yes)
    assert g_yes.n_params() > g_no.n_params()
