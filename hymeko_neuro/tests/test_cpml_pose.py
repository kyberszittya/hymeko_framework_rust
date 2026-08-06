"""Tests for CPMLPose — CPML adapted for per-vertex 3D pose regression.

Verifies (Shape A of the 2026-05-29 scoping note):
  1. CPMLPose constructs and yields a valid module.
  2. Forward on a tiny synthetic (N, M_cycles, k) input returns shape (N, 3).
  3. Trains one step without NaN; the pos_head has finite gradient.
  4. The CPML refactor is non-invasive — CPML itself still produces
     edge logits as before (no regression).
"""
from __future__ import annotations

import torch


def _tiny_pose_input(device="cpu", n_nodes=8, n_cycles=6, arity=4):
    """Build a fake per_arity_input tuple matching ``_build_input``'s
    contract closely enough for CPMLPose's forward."""
    triad_v = torch.randint(0, n_nodes, (n_cycles, arity), device=device,
                            dtype=torch.long)
    triad_sigma = (torch.randint(0, 2, (n_cycles, arity), device=device,
                                  dtype=torch.long) * 2 - 1)
    # Sparse M_vt: (N, M_cycles). Not directly used by CPMLPose (it derives
    # tier from triad_v); included for API parity with PositionRegHSiKAN.
    rows = triad_v.view(-1)
    cols = torch.arange(n_cycles, device=device).repeat_interleave(arity)
    vals = torch.ones(rows.numel(), device=device)
    M_vt = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), vals, (n_nodes, n_cycles)
    )
    M_e = torch.zeros(0, n_cycles, device=device)  # unused by CPMLPose
    return [(triad_v, triad_sigma, M_vt, M_e)]


def test_cpml_pose_construct_and_forward_shape():
    from hymeko_neuro.experiments.runs.run_phase12_position_regression import (
        CPMLPose,
    )
    torch.manual_seed(0)
    n_nodes = 8
    model = CPMLPose(n_nodes_max=n_nodes, arity=4, hidden=8, tier_l=2)
    inp = _tiny_pose_input(n_nodes=n_nodes, arity=4)
    out = model(inp)
    assert out.shape == (n_nodes, 3), (
        f"expected ({n_nodes}, 3); got {tuple(out.shape)}"
    )
    assert torch.isfinite(out).all()


def test_cpml_pose_trains_one_step():
    from hymeko_neuro.experiments.runs.run_phase12_position_regression import (
        CPMLPose,
    )
    torch.manual_seed(1)
    n_nodes = 8
    model = CPMLPose(n_nodes_max=n_nodes, arity=4, hidden=8, tier_l=2)
    inp = _tiny_pose_input(n_nodes=n_nodes, arity=4)
    target = torch.randn(n_nodes, 3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    pred = model(inp)
    loss = (pred - target).pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    opt.step()
    # The pos_head_pose lives inside model.cpml (the subclassed CPML
    # backbone). It must have received a finite gradient.
    assert model.cpml.pos_head_pose.weight.grad is not None
    assert torch.isfinite(model.cpml.pos_head_pose.weight.grad).all()
    # Node embedding should also have received a gradient (CPML chain
    # is differentiable end-to-end).
    assert model.node_embed.weight.grad is not None
    assert torch.isfinite(model.node_embed.weight.grad).all()


def test_cpml_pose_tier_organizations():
    """Both 'structural' and 'capsule_soft' tier organizations should
    construct + forward without error."""
    from hymeko_neuro.experiments.runs.run_phase12_position_regression import (
        CPMLPose,
    )
    for org in ("structural", "capsule_soft"):
        torch.manual_seed(2)
        model = CPMLPose(n_nodes_max=8, arity=4, hidden=8, tier_l=2,
                         tier_organization=org)
        inp = _tiny_pose_input(n_nodes=8, arity=4)
        out = model(inp)
        assert out.shape == (8, 3)


def test_cpml_still_emits_edge_logits():
    """Regression: instantiating CPMLPose must not affect CPML's normal
    per-edge output (the subclass override only applies to its own
    instance)."""
    from hymeko_neuro.experiments.runs.run_phase12_position_regression import (
        CPMLPose,
    )
    from hymeko_neuro.hyperedge.cpml import CPML, CPMLConfig, TierSpec

    # Build CPMLPose first to ensure no global side effects.
    _ = CPMLPose(n_nodes_max=8, arity=4, hidden=8, tier_l=2)

    # Now build a plain CPML and verify edge-logits shape.
    torch.manual_seed(3)
    cfg = CPMLConfig(tier_spec=TierSpec(cuts=(0.0, 0.5, 1.0)), d_in=8,
                     d_layer=8, cycle_k=4)
    cpml = CPML(cfg)
    n, m = 8, 6
    node_features = torch.randn(n, 8)
    cycles = torch.randint(0, n, (m, 4), dtype=torch.long)
    cycle_signs = torch.randint(0, 2, (m, 4), dtype=torch.long) * 2 - 1
    tier_of = torch.randint(0, 2, (n,), dtype=torch.long)
    edges = torch.randint(0, n, (5, 2), dtype=torch.long)
    out = cpml(node_features, cycles, cycle_signs, tier_of, edges)
    # CPML's regular forward returns per-edge scores (E,), not per-vertex.
    assert out.shape == (5,), (
        f"CPML regression: expected (5,); got {tuple(out.shape)}"
    )
