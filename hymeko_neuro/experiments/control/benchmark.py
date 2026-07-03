"""Benchmark harness for the lateral-tracking comparison.

Simulates a controller against a track for T seconds at fixed dt and
returns per-run metrics: lateral error RMSE, max lateral error,
final-position error, control effort (RMSE of δ), wall-clock time per
step.

Also exposes :func:`imitation_train_hsikan` — a small training loop
that fits the HSIKAN policy on (state-window, expert action) pairs
collected from a teacher controller (default LQR) on a training
track.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .bicycle import BicycleParams, BicycleState, BicycleVehicle
from .controllers import HSIKANController, HSIKANPolicy
from .tracks import Track


@dataclass
class RunMetrics:
    lat_rmse: float
    lat_max: float
    final_pos_err: float
    delta_rmse: float
    wall_per_step_ms: float
    n_steps: int


def run_episode(
    controller, track: Track, vehicle: BicycleVehicle,
    initial_state: BicycleState, dt: float = 0.05, T: float = 20.0,
) -> RunMetrics:
    state = initial_state
    controller.reset(track, state)
    n = int(T / dt)
    lat_errs: List[float] = []
    deltas: List[float] = []
    t0 = time.perf_counter()
    for _ in range(n):
        delta = controller.step(state, track, dt)
        deltas.append(delta)
        state = vehicle.step(state, delta, dt, a=0.0)
        _, lat, _, _ = track.project(state.x, state.y)
        lat_errs.append(lat)
    wall = time.perf_counter() - t0

    lat_arr = np.array(lat_errs, dtype=np.float64)
    delta_arr = np.array(deltas, dtype=np.float64)
    final_pos_err = np.hypot(state.x - track.x[-1], state.y - track.y[-1])

    return RunMetrics(
        lat_rmse=float(np.sqrt(np.mean(lat_arr ** 2))),
        lat_max=float(np.max(np.abs(lat_arr))),
        final_pos_err=float(final_pos_err),
        delta_rmse=float(np.sqrt(np.mean(delta_arr ** 2))),
        wall_per_step_ms=float(1000.0 * wall / n),
        n_steps=n,
    )


def collect_imitation_dataset(
    teacher, track: Track, vehicle: BicycleVehicle,
    initial_state: BicycleState, dt: float = 0.05, T: float = 20.0,
    window: int = 8,
):
    """Run ``teacher`` and record (feature-window, expert-action, sigma-window)."""
    state = initial_state
    teacher.reset(track, state)
    feats_hist: List[List[List[float]]] = []
    sigma_hist: List[List[float]] = []
    actions: List[float] = []
    window_buf: List[List[float]] = [[0.0, 0.0, 0.0] for _ in range(window)]
    sigma_buf: List[float] = [0.0 for _ in range(window)]
    n = int(T / dt)
    for _ in range(n):
        idx, e_y, psi_r, kappa_r = track.project(state.x, state.y)
        e_psi = ((state.psi - psi_r + np.pi) % (2.0 * np.pi)) - np.pi
        window_buf.append([e_y, e_psi, kappa_r])
        window_buf.pop(0)
        sigma_buf.append(float(np.sign(e_y)))
        sigma_buf.pop(0)
        delta = teacher.step(state, track, dt)
        feats_hist.append([list(row) for row in window_buf])
        sigma_hist.append(list(sigma_buf))
        actions.append(delta)
        state = vehicle.step(state, delta, dt, a=0.0)
    X = torch.tensor(feats_hist, dtype=torch.float32)
    S = torch.tensor(sigma_hist, dtype=torch.float32)
    y = torch.tensor(actions, dtype=torch.float32)
    return X, S, y


def imitation_train_hsikan(
    policy: HSIKANPolicy, X: torch.Tensor, S: torch.Tensor, y: torch.Tensor,
    epochs: int = 200, lr: float = 5e-3, batch_size: int = 128,
    val_frac: float = 0.2,
) -> Dict[str, float]:
    """Fit ``policy`` to predict teacher actions from windowed state.

    Returns ``{"train_mse_final", "val_mse_final"}``.
    """
    n = X.shape[0]
    n_val = max(1, int(n * val_frac))
    perm = torch.randperm(n)
    idx_tr, idx_va = perm[: n - n_val], perm[n - n_val :]
    X_tr, S_tr, y_tr = X[idx_tr], S[idx_tr], y[idx_tr]
    X_va, S_va, y_va = X[idx_va], S[idx_va], y[idx_va]

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_tr = X_tr.shape[0]
    final = {"train_mse_final": float("nan"), "val_mse_final": float("nan")}
    for ep in range(epochs):
        policy.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = perm[i : i + batch_size]
            pred = policy(X_tr[idx], S_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        policy.eval()
        with torch.no_grad():
            tr_mse = float(loss_fn(policy(X_tr, S_tr), y_tr))
            va_mse = float(loss_fn(policy(X_va, S_va), y_va))
        final["train_mse_final"] = tr_mse
        final["val_mse_final"] = va_mse
    return final


__all__ = ["RunMetrics", "run_episode", "collect_imitation_dataset",
            "imitation_train_hsikan"]
