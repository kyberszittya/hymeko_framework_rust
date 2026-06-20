"""Tests for the rotor-relative edge head (core/rotor_relative_head.py).

The head's defining property: it projects the *relative* rotation
``conj(q_u) ⊗ q_v`` of an edge's endpoints — so it is invariant to a global
rotation of both endpoints and sensitive to rotating just one. That is the DoF
the lossy ``e = R v0`` embedding (and hence the bilinear head) cannot see.

Run: pytest -p no:randomly signedkan_wip/tests/test_rotor_relative_head.py
"""
from __future__ import annotations

import pytest
import torch

from signedkan_wip.src.core.rotor_relative_head import (
    RotorGeodesicHead,
    RotorRelativeHead,
)
from signedkan_wip.src.embeddings.cayley_rotor import cayley_to_unit_quat, quat_mul


def _rotors(e: int, n_blocks: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return cayley_to_unit_quat(torch.randn(e, n_blocks, 3, generator=g))


def test_output_shape_and_dead_start() -> None:
    head = RotorRelativeHead(n_blocks=4)
    out = head(_rotors(7, 4, 0), _rotors(7, 4, 1))
    assert out.shape == (7,)
    assert out.abs().max() < 0.05          # gamma_init=1e-3 → effectively off


def test_rejects_mismatched_or_nonquaternion_shape() -> None:
    head = RotorRelativeHead(n_blocks=3)
    with pytest.raises(ValueError):
        head(torch.zeros(5, 3, 4), torch.zeros(5, 2, 4))   # block mismatch


def test_rejects_nonpositive_blocks() -> None:
    with pytest.raises(ValueError):
        RotorRelativeHead(n_blocks=0)


def test_relative_rotation_semantics() -> None:
    """Invariant to a global rotation of both endpoints; changes when only one
    endpoint rotates — i.e. it reads the *relative* rotor, the lost DoF."""
    head = RotorRelativeHead(n_blocks=4)
    with torch.no_grad():
        head.gamma.fill_(1.0)              # activate to observe the effect
    q_u, q_v, g = _rotors(6, 4, 2), _rotors(6, 4, 3), _rotors(6, 4, 4)
    base = head(q_u, q_v)
    # conj(g⊗q_u)⊗(g⊗q_v) = conj(q_u)⊗q_v  (global rotation cancels)
    assert torch.allclose(head(quat_mul(g, q_u), quat_mul(g, q_v)), base, atol=1e-5)
    # rotating only q_v changes the relative rotor → changes the output
    assert not torch.allclose(head(q_u, quat_mul(g, q_v)), base, atol=1e-4)


def test_gradient_flows() -> None:
    head = RotorRelativeHead(n_blocks=3)
    head(_rotors(5, 3, 5), _rotors(5, 3, 6)).pow(2).sum().backward()
    assert head.proj.weight.grad is not None
    assert head.gamma.grad is not None


# ── geodesic (RotatE-style) head ──────────────────────────────────────
def test_geodesic_output_shape_and_dead_start() -> None:
    head = RotorGeodesicHead(n_blocks=4)
    out = head(_rotors(7, 4, 0), _rotors(7, 4, 1))
    assert out.shape == (7,)
    assert out.abs().max() < 0.05            # dead-start gamma


def test_geodesic_invariant_to_global_rotation() -> None:
    """Reads only the relative-rotation alignment cos(θ/2)=⟨q_u,q_v⟩, so a global
    rotation of both endpoints leaves the score unchanged."""
    head = RotorGeodesicHead(n_blocks=4)
    with torch.no_grad():
        head.gamma.fill_(1.0)
    q_u, q_v, g = _rotors(6, 4, 2), _rotors(6, 4, 3), _rotors(6, 4, 4)
    base = head(q_u, q_v)
    assert torch.allclose(head(quat_mul(g, q_u), quat_mul(g, q_v)), base, atol=1e-5)
    assert not torch.allclose(head(q_u, quat_mul(g, q_v)), base, atol=1e-4)


def test_geodesic_gradient_flows() -> None:
    head = RotorGeodesicHead(n_blocks=3)
    head(_rotors(5, 3, 5), _rotors(5, 3, 6)).pow(2).sum().backward()
    assert head.proj.weight.grad is not None and head.gamma.grad is not None
