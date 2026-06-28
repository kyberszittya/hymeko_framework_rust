"""Tests for the rotor / holonomy toy (hymeko_rl/rotor_probe.py).

Pins the rotation matrix, the holonomy-transport target, both transport models, and the headline result —
the signed (Z2 scalar) model is stuck at sin²Φ while the rotor (SO(2)) fits the continuous holonomy.
"""
from __future__ import annotations

import math

import torch

from hymeko_rl.rotor_probe import (
    RotorTransport, SignedTransport, make_holonomy_data, rot_matrix, run_rotor_probe,
)


def test_rot_matrix_is_a_rotation() -> None:
    r = rot_matrix(0.7)
    assert torch.allclose(r.T @ r, torch.eye(2), atol=1e-6)        # orthogonal
    assert abs(float(torch.det(r)) - 1.0) < 1e-6                   # proper (det +1)
    # R(π/2) maps (1,0) -> (0,1)
    assert torch.allclose(rot_matrix(math.pi / 2) @ torch.tensor([1.0, 0.0]),
                          torch.tensor([0.0, 1.0]), atol=1e-6)


def test_holonomy_data_is_rotated_source() -> None:
    x, y = make_holonomy_data(0.9, 64, seed=0)
    assert x.shape == (64, 2) and y.shape == (64, 2)
    assert torch.allclose(y, x @ rot_matrix(0.9).T, atol=1e-6)     # y[i] = R(phi) x[i]


def test_both_transports_forward_finite() -> None:
    x, _ = make_holonomy_data(0.5, 8, seed=1)
    for model in (RotorTransport(), SignedTransport()):
        out = model(x)
        assert out.shape == (8, 2) and torch.isfinite(out).all()


def test_signed_cannot_rotate_but_rotor_can() -> None:
    """The headline: at Φ=π/2 (a pure 90° holonomy) the signed scalar model is stuck near sin²(π/2)=1 while
    the rotor fits it (≈0). This fails for any scalar-only ('signed') connection — it cannot rotate."""
    report = run_rotor_probe(phis=[math.pi / 2], n_train=256, n_test=512, seeds=2, epochs=300)
    row = report["rows"][0]
    assert row["rotor_mse"] < 0.05, f"rotor should fit the holonomy, got {row['rotor_mse']}"
    assert row["signed_mse"] > 0.5, f"signed (scalar) cannot rotate, got {row['signed_mse']}"
    assert abs(row["signed_mse"] - 1.0) < 0.25                     # tracks sin²(π/2) = 1
    assert report["signed_over_rotor_at_halfpi"] > 10


def test_signed_matches_z2_points() -> None:
    """At Φ=π (holonomy = a sign flip, a Z2 point) the signed model CAN represent it (c=-1) → low MSE,
    confirming the signed link captures exactly the Z2 quotient of the holonomy."""
    report = run_rotor_probe(phis=[math.pi], n_train=256, n_test=512, seeds=2, epochs=300)
    row = report["rows"][0]
    assert row["signed_mse"] < 0.1, f"signed should fit the Z2 point Φ=π, got {row['signed_mse']}"
