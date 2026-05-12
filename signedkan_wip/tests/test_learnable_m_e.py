"""Tests for Path B: LearnableMe per-cycle importance weighting.

Validates:
  * identity init: w_c=1.0 reproduces the baseline M_e exactly
  * each parameterization (scalar, exp, sigmoid2) produces correct
    initial weights
  * gradients flow through the cycle-weight parameters to the loss
  * a tiny synthetic problem learns non-uniform cycle weights when
    cycles have different predictive value
  * regularization term is correct
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signedkan_wip.src.learnable_m_e import LearnableMe


# ─── Identity-init smoke ────────────────────────────────────────────


@pytest.mark.parametrize("param_kind", ["scalar", "exp", "sigmoid2"])
def test_init_at_unity(param_kind):
    """At init, w_c == init_value (==1.0 by default) for every cycle."""
    n_cycles = 50
    lm = LearnableMe(n_cycles=n_cycles, param_kind=param_kind, init_value=1.0)
    w = lm.weights()
    assert w.shape == (n_cycles,)
    assert torch.allclose(w, torch.ones(n_cycles), atol=1e-6), \
        f"{param_kind} init produced {w[:5]} not all 1.0"


def test_apply_at_unity_preserves_m_e():
    """At identity init, apply() returns the same M_e values."""
    indices = torch.tensor([[0, 1, 2, 1], [0, 0, 1, 2]], dtype=torch.long)
    values = torch.tensor([1.0, -1.0, 1.0, -1.0])
    M_e = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()
    cycle_idx_of_nz = M_e.indices()[1]
    lm = LearnableMe(n_cycles=3, init_value=1.0)
    M_e_weighted = lm.apply(M_e, cycle_idx_of_nz)
    assert torch.allclose(M_e.values(), M_e_weighted.values(), atol=1e-6)
    assert torch.equal(M_e.indices(), M_e_weighted.indices())


# ─── Gradient flow ──────────────────────────────────────────────────


def test_gradient_flows_to_cycle_weights():
    """∂L/∂θ_c is non-zero when cycle c contributes to the loss."""
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    values = torch.tensor([1.0, -1.0, 1.0])
    M_e = torch.sparse_coo_tensor(indices, values, (3, 3)).coalesce()
    cycle_idx = M_e.indices()[1]
    lm = LearnableMe(n_cycles=3, init_value=1.0)
    M_e_w = lm.apply(M_e, cycle_idx)
    # Synthetic loss: sum of M_e_w values (so each w_c sees gradient).
    loss = M_e_w.values().sum()
    loss.backward()
    assert lm.theta.grad is not None
    assert not torch.isnan(lm.theta.grad).any()
    # Each w_c contributes M_e[*, c] to the loss; gradient is the
    # sum of M_e values at that column (here 1.0 / -1.0 / 1.0).
    assert torch.allclose(
        lm.theta.grad, torch.tensor([1.0, -1.0, 1.0]), atol=1e-5,
    )


# ─── Learning non-uniform weights on a synthetic problem ────────────


def test_synth_problem_learns_to_downweight_noise_cycles():
    """Synthetic setup: 4 cycles, 2 of them carry signal, 2 are noise.
    Train edge sign prediction on edge 0; verify the noise cycles'
    weights drift below the signal cycles' weights."""
    torch.manual_seed(0)
    n_edges = 1
    n_cycles = 4
    # All 4 cycles incident to edge 0 in M_e (init value = 1.0
    # uniformly); the model must learn to down-weight noise cycles.
    indices = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=torch.long)
    values = torch.ones(4)
    M_e = torch.sparse_coo_tensor(indices, values, (n_edges, n_cycles)).coalesce()
    cycle_idx = M_e.indices()[1]

    target_h_signal = torch.tensor([[1.0], [1.0]])
    target_y = torch.tensor([1.0])

    lm = LearnableMe(n_cycles=n_cycles, param_kind="scalar", init_value=1.0)
    opt = torch.optim.Adam([lm.theta], lr=0.2)

    for step in range(500):
        h_noise = torch.randn(2, 1) * 1.0
        h_cycle = torch.cat([target_h_signal, h_noise], dim=0)
        M_e_w = lm.apply(M_e, cycle_idx)
        edge_pool = torch.sparse.mm(M_e_w, h_cycle)  # (1, 1)
        loss = F.binary_cross_entropy_with_logits(
            edge_pool.squeeze(-1), target_y,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    w_final = lm.weights().detach()
    signal_mean = w_final[:2].mean().item()
    noise_mean = w_final[2:].mean().item()
    assert signal_mean > noise_mean, \
        f"signal weights {w_final[:2].tolist()} should exceed noise " \
        f"weights {w_final[2:].tolist()}"
    # Reasonably interpretable gap
    assert signal_mean - noise_mean > 0.1, \
        f"separation only {signal_mean - noise_mean:.3f}"


# ─── Regularization ─────────────────────────────────────────────────


def test_regularization_is_zero_at_init():
    """At init (w_c = 1.0 ∀ c), reg = lam · sum(0²) = 0."""
    lm = LearnableMe(n_cycles=10, init_value=1.0)
    reg = lm.regularization(lam=0.01)
    assert reg.item() == pytest.approx(0.0, abs=1e-6)


def test_regularization_grows_when_weights_drift():
    """Set θ to a known non-unity value, verify reg(lam, w) matches."""
    lm = LearnableMe(n_cycles=5, init_value=1.0)
    with torch.no_grad():
        lm.theta[0] = 3.0    # w[0] = 3.0; (3-1)^2 = 4
        lm.theta[1] = -1.0   # w[1] = -1.0; (-1-1)^2 = 4
    reg = lm.regularization(lam=0.5)
    # 0.5 · (4 + 4 + 0 + 0 + 0) = 4.0
    assert reg.item() == pytest.approx(4.0, abs=1e-4)


# ─── Parameter count ─────────────────────────────────────────────────


def test_n_params_matches_cycle_count():
    """LearnableMe should add exactly n_cycles parameters."""
    lm = LearnableMe(n_cycles=12345)
    assert lm.n_params() == 12345
    assert sum(p.numel() for p in lm.parameters()) == 12345
