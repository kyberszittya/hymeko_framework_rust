"""Unit tests for the Phase-17 side-stacked HSIKAN.

Pins the parallel-branch wrapper's behaviour: forward + backward
on every fusion mode, N=1 equivalence to bare SignedKAN at same
init, parameter count scales with N, and concat output dim is
N × hidden_dim while other fusions keep hidden_dim.
"""
from __future__ import annotations

import numpy as np
import torch

from signedkan_wip.src.core.signedkan import SignedKAN, SignedKANConfig
from signedkan_wip.src.core.side_signedkan import (
    MembraneSignedKAN,
    MembraneSignedKANConfig,
    SideSignedKAN,
    SideSignedKANConfig,
)


def _toy_triads(n_nodes: int = 6, n_triads: int = 5):
    rng = np.random.default_rng(0)
    triad_v_np = rng.integers(0, n_nodes, size=(n_triads, 3))
    triad_v_np.sort(axis=1)
    for i in range(n_triads):
        while len(set(triad_v_np[i])) < 3:
            triad_v_np[i] = sorted(rng.integers(0, n_nodes, size=3))
    triad_sigma = torch.tensor(rng.choice([-1, 1], size=(n_triads, 3)),
                               dtype=torch.long)
    return torch.from_numpy(triad_v_np).long(), triad_sigma


def test_forward_shape_mean_fusion():
    cfg = SideSignedKANConfig(n_nodes=6, n_branches=4, hidden_dim=8,
                              fusion="mean")
    model = SideSignedKAN(cfg)
    triad_v, triad_sigma = _toy_triads()
    h_t = model.encode_triads(triad_v, triad_sigma)
    assert h_t.shape == (triad_v.shape[0], 8)


def test_concat_fusion_widens_output():
    cfg = SideSignedKANConfig(n_nodes=6, n_branches=3, hidden_dim=8,
                              fusion="concat")
    model = SideSignedKAN(cfg)
    triad_v, triad_sigma = _toy_triads()
    h_t = model.encode_triads(triad_v, triad_sigma)
    assert h_t.shape == (triad_v.shape[0], 24)  # N=3 × hidden=8


def test_all_fusion_modes_run():
    """Every fusion mode forward-passes without error."""
    triad_v, triad_sigma = _toy_triads()
    for fusion in ("sum", "mean", "concat", "learned_alpha", "attention"):
        cfg = SideSignedKANConfig(n_nodes=6, n_branches=2, hidden_dim=4,
                                  fusion=fusion)
        model = SideSignedKAN(cfg)
        h_t = model.encode_triads(triad_v, triad_sigma)
        assert h_t.numel() > 0, f"fusion={fusion} produced empty output"


def test_backward_passes_through_branches():
    """Gradients reach every branch's parameters."""
    cfg = SideSignedKANConfig(n_nodes=6, n_branches=3, hidden_dim=4,
                              fusion="mean")
    model = SideSignedKAN(cfg)
    triad_v, triad_sigma = _toy_triads()
    h_t = model.encode_triads(triad_v, triad_sigma)
    loss = h_t.sum()
    loss.backward()
    # Every branch should have at least one non-zero grad.
    for i, branch in enumerate(model.branches):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in branch.parameters()
        )
        assert has_grad, f"branch {i} got no gradient"


def test_learned_alpha_param_count():
    """`fusion='learned_alpha'` adds exactly `n_branches` parameters
    beyond what sum/mean would have."""
    base = SideSignedKAN(SideSignedKANConfig(
        n_nodes=6, n_branches=4, hidden_dim=8, fusion="mean",
    ))
    learned = SideSignedKAN(SideSignedKANConfig(
        n_nodes=6, n_branches=4, hidden_dim=8, fusion="learned_alpha",
    ))
    diff = learned.num_parameters() - base.num_parameters()
    assert diff == 4, f"learned_alpha overhead {diff} != n_branches=4"


def test_param_count_scales_with_n_branches():
    """N=4 branches has ~4× the per-branch parameter budget of N=1."""
    m1 = SideSignedKAN(SideSignedKANConfig(
        n_nodes=6, n_branches=1, hidden_dim=8, fusion="mean",
    ))
    m4 = SideSignedKAN(SideSignedKANConfig(
        n_nodes=6, n_branches=4, hidden_dim=8, fusion="mean",
    ))
    # 4× is the exact ratio for mean fusion (zero fusion params).
    ratio = m4.num_parameters() / m1.num_parameters()
    assert 3.9 < ratio < 4.1, f"ratio = {ratio} should be ≈ 4.0"


# ─── Phase 18: MembraneSignedKAN tests ──────────────────────────────


def test_membrane_forward_shape_mean_fusion():
    cfg = MembraneSignedKANConfig(n_nodes=6, n_branches=4, hidden_dim=8,
                                   fusion="mean")
    model = MembraneSignedKAN(cfg)
    triad_v, triad_sigma = _toy_triads()
    h_t = model.encode_triads(triad_v, triad_sigma)
    assert h_t.shape == (triad_v.shape[0], 8)


def test_membrane_starts_close_to_side_at_init():
    """`MembraneSignedKAN` initialises read gates near zero, so at
    init time it should produce a similar embedding to a
    `SideSignedKAN` with the same branch init.

    Confirms the read-gate-near-zero design: the model starts
    behaving like its plain side-stacked cousin and learns the
    membrane coupling over training."""
    torch.manual_seed(123)
    triad_v, triad_sigma = _toy_triads()
    side_cfg = SideSignedKANConfig(n_nodes=6, n_branches=4, hidden_dim=8,
                                     fusion="mean")
    torch.manual_seed(99)
    side = SideSignedKAN(side_cfg)
    torch.manual_seed(99)
    mem_cfg = MembraneSignedKANConfig(n_nodes=6, n_branches=4, hidden_dim=8,
                                         fusion="mean",
                                         read_gate_init=0.0)
    mem = MembraneSignedKAN(mem_cfg)
    h_side = side.encode_triads(triad_v, triad_sigma)
    h_mem = mem.encode_triads(triad_v, triad_sigma)
    # Branches were re-seeded; values won't be byte-identical but
    # the L2 distance should be small because read_gate weights
    # init at std=0.01.
    rel_diff = (h_side - h_mem).abs().mean() / (h_side.abs().mean() + 1e-9)
    assert rel_diff.item() < 0.20, (
        f"membrane at near-zero gate init should approximate side; "
        f"rel_diff = {rel_diff.item():.4f}"
    )


def test_membrane_backward_passes_through_branches_and_gates():
    cfg = MembraneSignedKANConfig(n_nodes=6, n_branches=3, hidden_dim=4,
                                    fusion="mean")
    model = MembraneSignedKAN(cfg)
    triad_v, triad_sigma = _toy_triads()
    h_t = model.encode_triads(triad_v, triad_sigma)
    loss = h_t.sum()
    loss.backward()
    # Every branch parameter group should have a gradient.
    for i, branch in enumerate(model.branches):
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in branch.parameters()
        ), f"branch {i} got no gradient"
    # Every read gate should also have a gradient.
    for i, gate in enumerate(model.read_gates):
        assert gate.weight.grad is not None
        # bias init is 0 + small linear → grad must be non-zero
        # somewhere in the gate.
        has_grad = (gate.weight.grad.abs().sum().item() > 0
                    or gate.bias.grad.abs().sum().item() > 0)
        assert has_grad, f"read_gate {i} got no gradient"


def test_membrane_aggregator_choices():
    """Every membrane_aggregator option forward-passes."""
    triad_v, triad_sigma = _toy_triads()
    for agg in ("mean", "max", "sum"):
        cfg = MembraneSignedKANConfig(n_nodes=6, n_branches=2, hidden_dim=4,
                                         fusion="mean",
                                         membrane_aggregator=agg)
        model = MembraneSignedKAN(cfg)
        h_t = model.encode_triads(triad_v, triad_sigma)
        assert h_t.shape == (triad_v.shape[0], 4), \
            f"aggregator={agg} produced wrong shape"


def test_membrane_param_count_vs_side():
    """Membrane adds N read gates (N × d × (d+1) params) over a
    side-stacked baseline at the same N."""
    side = SideSignedKAN(SideSignedKANConfig(
        n_nodes=6, n_branches=4, hidden_dim=8, fusion="mean",
    ))
    mem = MembraneSignedKAN(MembraneSignedKANConfig(
        n_nodes=6, n_branches=4, hidden_dim=8, fusion="mean",
    ))
    # Each read gate is Linear(8, 8) → 8*8 + 8 = 72 params; 4 gates → 288.
    diff = mem.num_parameters() - side.num_parameters()
    expected = 4 * (8 * 8 + 8)
    assert diff == expected, \
        f"membrane overhead = {diff}, expected {expected}"


def test_n_branches_1_matches_bare_signedkan_at_same_init():
    """N=1 with fusion='mean' (or 'sum') should be functionally
    equivalent to a bare SignedKAN at the same init."""
    torch.manual_seed(42)
    bare = SignedKAN(SignedKANConfig(
        n_nodes=6, hidden_dim=4, grid=5, k=3,
        spline_kind="bspline",
    ))
    torch.manual_seed(42)
    cfg = SideSignedKANConfig(n_nodes=6, n_branches=1, hidden_dim=4,
                              fusion="mean")
    side = SideSignedKAN(cfg)
    triad_v, triad_sigma = _toy_triads()
    out_bare = bare.encode_triads(triad_v, triad_sigma)
    out_side = side.encode_triads(triad_v, triad_sigma)
    assert torch.allclose(out_bare, out_side, atol=1e-5), \
        f"N=1 side != bare at same init: max diff = {(out_bare - out_side).abs().max().item()}"
