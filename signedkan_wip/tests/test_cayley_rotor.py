"""Tests for the Cayley-rotor embedding (signedkan_wip/src/embeddings/cayley_rotor.py).

Run: pytest -p no:randomly signedkan_wip/tests/test_cayley_rotor.py
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.embeddings.cayley_rotor import (
    CayleyRotorEmbedding,
    cayley_to_unit_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate,
)


# ── rotor algebra ─────────────────────────────────────────────────────

def test_cayley_map_is_unit_quaternion() -> None:
    torch.manual_seed(0)
    b = torch.randn(64, 5, 3) * 3.0          # large b stresses the normalisation
    q = cayley_to_unit_quat(b)
    norms = q.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert (q[..., 0] >= 0).all()            # scalar part from the "1" is non-negative


def test_zero_bivector_is_identity_rotor() -> None:
    b = torch.zeros(3, 4, 3)
    q = cayley_to_unit_quat(b)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(q, ident.expand_as(q), atol=1e-6)


def test_rotation_preserves_norm() -> None:
    torch.manual_seed(1)
    b = torch.randn(32, 3)
    q = cayley_to_unit_quat(b)
    v = torch.randn(32, 3)
    rv = quat_rotate(q, v)
    assert torch.allclose(rv.norm(dim=-1), v.norm(dim=-1), atol=1e-5)


def test_quat_rotate_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        quat_rotate(torch.zeros(2, 4), torch.zeros(2, 4))


# ── embedding module ──────────────────────────────────────────────────

def test_transductive_identity_at_init_equals_reference() -> None:
    # bivectors init to 0 → identity rotor → embedding == tiled reference.
    m = CayleyRotorEmbedding(n_blocks=4, n_items=10)
    e = m(torch.arange(10))
    ref = m.reference.detach().reshape(-1)
    assert e.shape == (10, 12)
    assert torch.allclose(e[0], ref, atol=1e-6)


def test_inductive_output_shape_and_sphere() -> None:
    m = CayleyRotorEmbedding(n_blocks=6, in_features=16)    # n_refs=1
    x = torch.randn(8, 16)
    e = m(x).reshape(8, 6, 1, 3)
    assert e.shape == (8, 6, 1, 3)
    # every (block, ref) 3-block sits on its reference sphere (rotation isometry)
    ref_norm = m.reference.detach().norm(dim=-1)            # (6, 1)
    assert torch.allclose(e.norm(dim=-1), ref_norm.expand(8, 6, 1), atol=1e-5)


# ── quaternion product / conjugate (rotor-relative head algebra) ──────
def test_quat_mul_matches_reference_hamilton() -> None:
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    # (1+2i+3j+4k)(5+6i+7j+8k) by hand:
    assert torch.allclose(quat_mul(a, b), torch.tensor([[-60.0, 12.0, 30.0, 24.0]]),
                          atol=1e-4)


def test_quat_mul_identity_and_associativity() -> None:
    torch.manual_seed(3)
    a, b, c = torch.randn(10, 4), torch.randn(10, 4), torch.randn(10, 4)
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(a)
    assert torch.allclose(quat_mul(ident, a), a, atol=1e-6)
    assert torch.allclose(quat_mul(a, ident), a, atol=1e-6)
    assert torch.allclose(quat_mul(quat_mul(a, b), c), quat_mul(a, quat_mul(b, c)),
                          atol=1e-4)


def test_conjugate_inverts_unit_rotor() -> None:
    torch.manual_seed(2)
    q = cayley_to_unit_quat(torch.randn(20, 5, 3))
    r = quat_mul(quat_conjugate(q), q)                      # should be identity
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(r)
    assert torch.allclose(r, ident, atol=1e-5)


# ── rotor accessor + multi-reference (less-lossy) embedding ───────────
def test_rotors_accessor_is_unit_and_shaped() -> None:
    m = CayleyRotorEmbedding(n_blocks=4, in_features=8)
    q = m.rotors(torch.randn(5, 8))
    assert q.shape == (5, 4, 4)
    assert torch.allclose(q.norm(dim=-1), torch.ones(5, 4), atol=1e-5)


def test_n_refs_scales_dim_and_default_unchanged() -> None:
    assert CayleyRotorEmbedding(n_blocks=4, in_features=8).embedding_dim == 12
    m2 = CayleyRotorEmbedding(n_blocks=4, in_features=8, n_refs=2)
    assert m2.embedding_dim == 24
    assert m2(torch.randn(3, 8)).shape == (3, 24)


def test_n_refs_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        CayleyRotorEmbedding(n_blocks=4, in_features=8, n_refs=0)


def test_embed_rotors_factorisation_parity() -> None:
    """forward == embed_rotors(rotors(x)) — the refactor that lets a propagation
    step transform q before embedding must not change the default output."""
    m = CayleyRotorEmbedding(n_blocks=5, in_features=8, n_refs=2)
    x = torch.randn(7, 8)
    assert torch.allclose(m(x), m.embed_rotors(m.rotors(x)), atol=1e-6)
    with pytest.raises(ValueError):
        m.embed_rotors(torch.randn(7, 5, 3))      # wrong last dim (not a quaternion)


def test_gradient_flows_both_modes() -> None:
    for kw in (dict(n_items=5), dict(in_features=7)):
        m = CayleyRotorEmbedding(n_blocks=3, **kw)
        x = torch.arange(5) if "n_items" in kw else torch.randn(5, 7)
        loss = m(x).pow(2).sum()
        loss.backward()
        trained = m.bivectors if m.bivectors is not None else m.proj.weight
        assert trained.grad is not None and torch.isfinite(trained.grad).all()
        assert m.reference.grad is not None


def test_inductive_param_count_is_item_count_free() -> None:
    # The whole point: inductive params do NOT grow with the item count, so they
    # beat a dense nn.Embedding(n_items, d) table for large vocabularies.
    small = CayleyRotorEmbedding(n_blocks=8, in_features=32).n_parameters()
    # a transductive table for 100k items at the same width:
    dense_table = 100_000 * (3 * 8)
    assert small < dense_table
    # and inductive count is independent of any item count (no item axis at all)
    assert CayleyRotorEmbedding(n_blocks=8, in_features=32).n_parameters() == small


def test_rejects_both_or_neither_mode() -> None:
    with pytest.raises(ValueError):
        CayleyRotorEmbedding(n_blocks=4)                        # neither
    with pytest.raises(ValueError):
        CayleyRotorEmbedding(n_blocks=4, n_items=3, in_features=3)  # both
