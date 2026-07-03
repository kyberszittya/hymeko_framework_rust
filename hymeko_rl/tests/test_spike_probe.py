"""Tests for the spikes toy (hymeko_rl/spike_probe.py).

Pins the SO(3) rotations (proper, non-commuting), the order-selected target, an oracle check (the spike-gated
model with the true angles reproduces the target exactly — guards the order convention), and the headline:
once the connection is non-abelian, the order-blind model fails while the spike-gated one fits.
"""
from __future__ import annotations

import math

import torch

from hymeko_rl.experiments.spike_probe import (
    OrderBlind, SpikeGated, SpikeMlp, WalkMlp, WalkSpikeGated, make_spike_data, make_walk_data, rot_x,
    rot_y, run_spike_probe, run_spike_walks,
)


def test_so3_rotations_are_proper_and_noncommuting() -> None:
    a, b = rot_x(torch.tensor(0.9)), rot_y(torch.tensor(1.3))
    for r in (a, b):
        assert torch.allclose(r.T @ r, torch.eye(3), atol=1e-5)        # orthonormal
        assert abs(float(torch.det(r)) - 1.0) < 1e-5                   # proper (det +1)
    assert not torch.allclose(a @ b, b @ a, atol=1e-3)                 # non-abelian
    assert torch.allclose(rot_x(torch.tensor(0.0)) @ rot_y(torch.tensor(0.0)),
                          rot_y(torch.tensor(0.0)) @ rot_x(torch.tensor(0.0)), atol=1e-6)  # commute at 0


def test_target_is_order_selected_transport() -> None:
    v, s, y = make_spike_data(0.7, 64, seed=0)
    th = torch.tensor(0.7)
    m1, m0 = rot_x(th) @ rot_y(th), rot_y(th) @ rot_x(th)
    expected = torch.where(s.unsqueeze(1) > 0.5, v @ m1.T, v @ m0.T)
    assert torch.allclose(y, expected, atol=1e-5)


def test_spike_gated_oracle_reproduces_target() -> None:
    """A SpikeGated with the TRUE angles reproduces the target exactly — proving the model's order convention
    matches the data's, so spike_gated genuinely CAN fit (no hidden order bug)."""
    v, s, y = make_spike_data(0.7, 128, seed=1)
    model = SpikeGated()
    with torch.no_grad():
        model.theta_a.fill_(0.7)
        model.theta_b.fill_(0.7)
        out = model(v, s)
    assert torch.allclose(out, y, atol=1e-5)


def test_three_models_forward_finite() -> None:
    v, s, _ = make_spike_data(0.5, 8, seed=2)
    for model in (SpikeGated(), OrderBlind(), SpikeMlp()):
        out = model(v, s)
        assert out.shape == (8, 3) and torch.isfinite(out).all()


def test_spike_needed_when_nonabelian() -> None:
    """At θ=0 (commuting) the order-blind model fits; at θ=1.2 (non-abelian) it fails while spike_gated fits —
    the spike is required to select walk order once the connection is non-abelian."""
    report = run_spike_probe(thetas=[0.0, 1.2], n_train=256, n_test=512, seeds=2, epochs=400)
    at0 = next(r for r in report["rows"] if r["theta"] == 0.0)
    at12 = next(r for r in report["rows"] if r["theta"] == 1.2)
    assert at0["order_blind_mse"] < 0.05, "at θ=0 the rotations commute → order-blind fits"
    assert at12["spike_gated_mse"] < 0.05, "spike_gated fits the order-selected target"
    assert at12["order_blind_mse"] > 0.2, "order-blind cannot select order when non-abelian"
    assert at12["blind_over_gated"] > 5


def test_walk_spike_gated_oracle_reproduces_target() -> None:
    """The stronger toy's correctness guard: WalkSpikeGated with the default unit gains reproduces the
    walk-holonomy target exactly across walk lengths (the batched SO(3) composition + walk-selection are right)."""
    for length in (1, 2, 3, 4):
        th, s, v, y = make_walk_data(length, 128, seed=3)
        model = WalkSpikeGated(length)               # g initialised to ones = the true gain
        with torch.no_grad():
            out = model(th, s, v)
        assert torch.allclose(out, y, atol=1e-5), f"oracle mismatch at L={length}"


def test_walk_models_forward_finite() -> None:
    th, s, v, _ = make_walk_data(3, 8, seed=0)
    for model in (WalkSpikeGated(3), WalkMlp(3)):
        out = model(th, s, v)
        assert out.shape == (8, 3) and torch.isfinite(out).all()


def test_walk_generalization_spike_gated_beats_mlp() -> None:
    """At a longer walk the structured spike-gated model generalizes (low MSE) while the flat MLP degrades —
    a generalization win, not mere representability."""
    report = run_spike_walks(walk_lens=[4], n_train=256, n_test=512, seeds=2, epochs=300)
    row = report["rows"][0]
    assert row["spike_gated_mse"] < 0.02, f"spike_gated should generalize, got {row['spike_gated_mse']}"
    assert row["mlp_over_gated"] > 3, f"spike_gated should beat the MLP, ratio {row['mlp_over_gated']}"


def test_rot_matrix_link_to_so2_probe_unused() -> None:
    """Sanity: a half-turn about x sends (0,1,0) -> (0,-1,0)."""
    assert torch.allclose(rot_x(torch.tensor(math.pi)) @ torch.tensor([0.0, 1.0, 0.0]),
                          torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)
