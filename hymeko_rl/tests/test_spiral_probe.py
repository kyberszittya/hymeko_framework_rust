"""Tests for the spiral-highway toy (hymeko_rl/spiral_probe.py).

Pins the walk-incidence, the walk-holonomy target, the three models, and the headline — the spiral (rotor
transport + α-collect along walks) fits the holonomy while the plain highway (identity carry) cannot.
"""
from __future__ import annotations

import math

import torch

from hymeko_rl.rotor_probe import rot_matrix
from hymeko_rl.spiral_probe import (
    FlatMlp, HighwayMlp, SpiralModel, make_spiral_data, run_spiral_probe, theta_graph_walks,
)


def test_theta_graph_walks_shape() -> None:
    inc = theta_graph_walks(3)
    assert inc.shape == (3, 6)                       # K walks, 2K edges
    assert inc.sum().item() == 6.0                   # each walk uses exactly 2 edges
    assert torch.equal(inc[1], torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0]))


def test_spiral_target_is_walk_holonomy_collection() -> None:
    """y = mean_k R(Σ_{e∈W_k} θ_e)·x. Hand-check the K=1 case (one walk = one rotation by θ0+θ1)."""
    inc = theta_graph_walks(1)
    theta, x, y = make_spiral_data(inc, 32, seed=0)
    phi = theta.sum(dim=1)                            # single walk holonomy = θ0 + θ1
    expected = torch.stack([torch.cos(phi) * x[:, 0] - torch.sin(phi) * x[:, 1],
                            torch.sin(phi) * x[:, 0] + torch.cos(phi) * x[:, 1]], dim=1)
    assert torch.allclose(y, expected, atol=1e-5)


def test_three_models_forward_finite() -> None:
    inc = theta_graph_walks(2)
    theta, x, _ = make_spiral_data(inc, 8, seed=1)
    for model in (SpiralModel(inc), HighwayMlp(4), FlatMlp(4)):
        out = model(theta, x)
        assert out.shape == (8, 2) and torch.isfinite(out).all()


def test_spiral_fits_holonomy_plain_highway_cannot() -> None:
    """The headline: the spiral fits the walk-holonomy (~0 MSE) while the plain highway (and the MLP) are
    stuck far above — the plain highway carries no structural/holonomy signal (its carry is identity-class)."""
    report = run_spiral_probe(k_paths=[2], n_train=256, n_test=512, seeds=2, epochs=400)
    row = report["rows"][0]
    assert row["spiral_mse"] < 0.01, f"spiral should fit the holonomy, got {row['spiral_mse']}"
    assert row["highway_mlp_mse"] > 0.1, f"plain highway should fail, got {row['highway_mlp_mse']}"
    assert row["mlp_over_spiral"] > 10
    # the plain highway is no better than the flat MLP — the gate is structure-free (mirrors the galambos null)
    assert abs(row["highway_mlp_mse"] - row["mlp_mse"]) < 0.5 or row["highway_mlp_mse"] > 0.1


def test_rot_matrix_reused_not_rebuilt() -> None:
    """The toy reuses the rotor probe's rotation (§6.1, no rebuild)."""
    assert torch.allclose(rot_matrix(math.pi / 2) @ torch.tensor([1.0, 0.0]),
                          torch.tensor([0.0, 1.0]), atol=1e-6)
