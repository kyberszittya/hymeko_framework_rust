"""Tests for signed slerp/nlerp rotor propagation (manifold-aware neighbourhood
input). Plan: docs/plans/2026-06-17-signed-rotor-slerp-propagation/.

Run: pytest -p no:randomly signedkan_wip/tests/test_signed_rotor_propagation.py
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.embeddings.cayley_rotor import cayley_to_unit_quat
from signedkan_wip.src.embeddings.signed_rotor_propagation import SignedRotorPropagation


def _unit_q(n, b, seed=0):
    g = torch.Generator().manual_seed(seed)
    return cayley_to_unit_quat(torch.randn(n, b, 3, generator=g))


EDGES = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)
SIGNS = torch.tensor([1.0, -1.0, 1.0])


def test_output_stays_on_the_sphere():
    """nlerp invariant: every propagated rotor is unit-norm (on S³)."""
    q = _unit_q(4, 3, 0)
    out = SignedRotorPropagation(rounds=2)(q, EDGES, SIGNS)
    assert out.shape == q.shape
    assert torch.allclose(out.norm(dim=-1), torch.ones(4, 3), atol=1e-5)


def test_rounds_and_self_weight_validation():
    with pytest.raises(ValueError):
        SignedRotorPropagation(rounds=0)
    with pytest.raises(ValueError):
        SignedRotorPropagation(self_weight=-1.0)


def test_isolated_node_unchanged():
    """A node with no incident edge is normalise(self_weight·q) = q (unchanged)."""
    q = _unit_q(4, 2, 1)                       # node 3 appears in no edge
    out = SignedRotorPropagation(rounds=1)(q, EDGES, SIGNS)
    assert torch.allclose(out[3], q[3], atol=1e-6)


def test_hemisphere_alignment_invariance():
    """q and −q are the same rotation; negating a neighbour's input rotor must not
    change a node's aggregate (the double-cover fix)."""
    q = _unit_q(3, 2, 2)
    q_flip = q.clone()
    q_flip[1] = -q_flip[1]                     # negate node 1's rotor
    prop = SignedRotorPropagation(rounds=1)
    out = prop(q, EDGES, SIGNS)
    out_flip = prop(q_flip, EDGES, SIGNS)
    # node 0's only moving neighbour-input is node 1 (edge 0-1, 0-2); node 2 also
    # flips, but check node 0 whose edges touch node 1: alignment cancels the flip.
    assert torch.allclose(out[0], out_flip[0], atol=1e-5)


def test_edge_sign_changes_the_aggregate():
    """Signed propagation: flipping an edge sign changes the result."""
    q = _unit_q(3, 2, 3)
    prop = SignedRotorPropagation(rounds=1)
    out_pos = prop(q, EDGES, torch.tensor([1.0, 1.0, 1.0]))
    out_mix = prop(q, EDGES, torch.tensor([1.0, -1.0, 1.0]))
    assert not torch.allclose(out_pos, out_mix, atol=1e-4)


def test_gradient_flows_through_propagation():
    q = _unit_q(4, 3, 4).requires_grad_(True)
    SignedRotorPropagation(rounds=2)(q, EDGES, SIGNS).pow(2).sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_bad_q_shape_raises():
    with pytest.raises(ValueError):
        SignedRotorPropagation(rounds=1)(torch.randn(4, 3), EDGES, SIGNS)


# --- Phase 1: learnable per-block self-retention --------------------------

def test_learnable_init_reproduces_fixed_self_weight():
    """Parity: an untrained learnable retention == the fixed sw=4 propagation."""
    q = _unit_q(4, 3, 5)
    fixed = SignedRotorPropagation(rounds=2, self_weight=4.0)
    learn = SignedRotorPropagation(rounds=2, self_weight=4.0,
                                   learnable=True, n_blocks=3)
    assert torch.allclose(learn(q, EDGES, SIGNS), fixed(q, EDGES, SIGNS), atol=1e-5)
    # The per-block retention starts at exactly self_weight.
    assert torch.allclose(learn.self_retention(),
                          torch.full((3,), 4.0), atol=1e-5)


def test_learnable_output_stays_on_the_sphere():
    q = _unit_q(4, 3, 6)
    out = SignedRotorPropagation(rounds=2, self_weight=4.0,
                                 learnable=True, n_blocks=3)(q, EDGES, SIGNS)
    assert torch.allclose(out.norm(dim=-1), torch.ones(4, 3), atol=1e-5)


def test_learnable_gradient_reaches_the_retention_param():
    q = _unit_q(4, 3, 7)
    prop = SignedRotorPropagation(rounds=2, self_weight=4.0,
                                  learnable=True, n_blocks=3)
    prop(q, EDGES, SIGNS).pow(2).sum().backward()
    g = prop.log_self_weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_learnable_requires_n_blocks():
    with pytest.raises(ValueError):
        SignedRotorPropagation(rounds=1, learnable=True)


def test_learnable_block_count_mismatch_raises():
    prop = SignedRotorPropagation(rounds=1, learnable=True, n_blocks=2)
    with pytest.raises(ValueError):
        prop(_unit_q(4, 3, 8), EDGES, SIGNS)  # q has 3 blocks, built for 2


def test_retention_floor_bounds_and_preserves_parity():
    """floor>0 keeps α_b ≥ floor (anti-collapse) yet still inits at self_weight."""
    prop = SignedRotorPropagation(rounds=1, self_weight=4.0, learnable=True,
                                  n_blocks=3, retention_floor=0.5)
    alpha = prop.self_retention()
    assert torch.all(alpha >= 0.5)
    assert torch.allclose(alpha, torch.full((3,), 4.0), atol=1e-5)  # init parity
    with pytest.raises(ValueError):
        SignedRotorPropagation(rounds=1, self_weight=-1.0)  # guard still fires
