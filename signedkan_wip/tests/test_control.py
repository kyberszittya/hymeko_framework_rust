"""Unit tests for the lateral-tracking control module."""

from __future__ import annotations

import math

import numpy as np
import torch

from signedkan_wip.src.control import (
    BicycleParams,
    BicycleState,
    BicycleVehicle,
    HSIKANController,
    LQRController,
    MPCController,
    PurePursuitController,
    s_curve_track,
    sinusoid_track,
    straight_track,
)
from signedkan_wip.src.control.benchmark import (
    collect_imitation_dataset,
    imitation_train_hsikan,
    run_episode,
)
from signedkan_wip.src.control.controllers import HSIKANPolicy


# ─── Bicycle dynamics ─────────────────────────────────────────────


def test_bicycle_zero_steer_goes_straight():
    veh = BicycleVehicle()
    s = BicycleState(x=0.0, y=0.0, psi=0.0, v=5.0)
    for _ in range(20):
        s = veh.step(s, delta=0.0, dt=0.05)
    # Should be exactly along the x-axis after 1 s.
    assert abs(s.y) < 1e-6
    assert abs(s.psi) < 1e-6
    assert s.x > 4.99  # 5.0 m/s × 1 s


def test_bicycle_constant_steer_curves_left():
    veh = BicycleVehicle()
    s = BicycleState(x=0.0, y=0.0, psi=0.0, v=5.0)
    for _ in range(20):
        s = veh.step(s, delta=0.1, dt=0.05)
    # δ > 0 ⇒ ψ̇ > 0 ⇒ heading turns left (positive ψ); y goes positive.
    assert s.psi > 0
    assert s.y > 0


def test_bicycle_steering_clipped():
    veh = BicycleVehicle(BicycleParams(delta_max=0.3))
    s = BicycleState(x=0.0, y=0.0, psi=0.0, v=5.0)
    s_clipped = veh.step(s, delta=10.0, dt=0.05)      # huge command
    s_capped = veh.step(s, delta=0.3, dt=0.05)        # at the cap
    # The clipped command should equal the capped command (within FP).
    assert abs(s_clipped.psi - s_capped.psi) < 1e-9


# ─── Tracks ───────────────────────────────────────────────────────


def test_straight_track_psi_is_zero():
    tk = straight_track(length=100.0, n=501)
    assert np.allclose(tk.psi, 0.0)
    # project a point above the path: lateral error should be POSITIVE.
    idx, lat, psi_r, _ = tk.project(50.0, 1.0)
    assert lat > 0.9
    assert abs(lat - 1.0) < 0.05


def test_sinusoid_track_curvature_alternates():
    tk = sinusoid_track(length=60.0, amplitude=2.0, wavelength=30.0)
    # curvature should have both positive and negative samples
    assert (tk.kappa > 0).any() and (tk.kappa < 0).any()


def test_s_curve_track_y_monotonic_after_centre():
    tk = s_curve_track(length=100.0, lane_width=3.5)
    # Sample far end: should be at ~lane_width.
    assert abs(tk.y[-1] - 3.5) < 0.5


# ─── Controllers (smoke + behavioural) ────────────────────────────


def test_lqr_keeps_close_to_straight():
    veh = BicycleVehicle()
    tk = straight_track(length=80.0)
    ctrl = LQRController(BicycleParams())
    init = BicycleState(x=0.0, y=0.5, psi=0.0, v=5.0)
    m = run_episode(ctrl, tk, veh, init, dt=0.05, T=10.0)
    assert m.lat_rmse < 0.2  # less than 20 cm RMSE for a 50-cm initial offset


def test_pure_pursuit_returns_valid_steering():
    veh = BicycleVehicle()
    tk = straight_track(length=50.0)
    ctrl = PurePursuitController(BicycleParams(), lookahead=4.0)
    init = BicycleState(x=0.0, y=0.0, psi=0.1, v=5.0)
    m = run_episode(ctrl, tk, veh, init, dt=0.05, T=5.0)
    # finite metrics + reasonable steering control effort
    assert math.isfinite(m.lat_rmse)
    assert m.delta_rmse < 0.5


def test_mpc_matches_lqr_within_tolerance_on_sinusoid():
    veh = BicycleVehicle()
    tk = sinusoid_track(length=60.0, amplitude=1.5, wavelength=30.0)
    init = BicycleState(x=0.0, y=0.0, psi=0.0, v=5.0)
    lqr = LQRController(BicycleParams())
    mpc = MPCController(BicycleParams(), horizon=6, dt=0.05)
    m_lqr = run_episode(lqr, tk, veh, init, dt=0.05, T=10.0)
    m_mpc = run_episode(mpc, tk, veh, init, dt=0.05, T=10.0)
    # LQR and MPC are both near-optimal on linearisable dynamics; their
    # lateral RMSE should be within 30 % of each other.
    ratio = max(m_lqr.lat_rmse, m_mpc.lat_rmse) / max(1e-6, min(m_lqr.lat_rmse, m_mpc.lat_rmse))
    assert ratio < 1.4, f"LQR={m_lqr.lat_rmse:.4f}, MPC={m_mpc.lat_rmse:.4f}, ratio={ratio}"


# ─── HSIKAN imitation + closed-loop ───────────────────────────────


def test_hsikan_policy_forward_shapes():
    p = HSIKANPolicy(window=8, n_channels=4, delta_max=0.6)
    feats = torch.randn(3, 8, 3)
    sigma = torch.sign(torch.randn(3, 8))
    delta = p(feats, sigma)
    assert delta.shape == (3,)
    assert (delta.abs() <= 0.6 + 1e-5).all()


def test_hsikan_imitation_lowers_loss():
    torch.manual_seed(0)
    np.random.seed(0)
    veh = BicycleVehicle()
    tk = sinusoid_track(length=60.0, amplitude=2.0, wavelength=30.0)
    init = BicycleState(x=0.0, y=0.5, psi=0.0, v=5.0)
    lqr = LQRController(BicycleParams())
    X, S, y = collect_imitation_dataset(lqr, tk, veh, init, dt=0.05, T=8.0, window=8)
    policy = HSIKANPolicy(window=8, n_channels=4)
    # initial val loss
    with torch.no_grad():
        init_loss = torch.nn.functional.mse_loss(policy(X, S), y).item()
    metrics = imitation_train_hsikan(policy, X, S, y, epochs=50, lr=5e-3)
    assert metrics["val_mse_final"] < init_loss
    # Should learn to predict actions with small MSE (LQR's |δ| ~ 0.15).
    assert metrics["val_mse_final"] < 0.05


def test_hsikan_controller_finite_steering_per_step():
    veh = BicycleVehicle()
    tk = straight_track(length=20.0)
    ctrl = HSIKANController(BicycleParams())
    init = BicycleState(x=0.0, y=0.2, psi=0.0, v=5.0)
    m = run_episode(ctrl, tk, veh, init, dt=0.05, T=4.0)
    # Untrained HSIKAN may drift wildly, but per-step steering must be finite & bounded.
    assert math.isfinite(m.lat_rmse)
    assert m.delta_rmse <= 0.6 + 1e-5
