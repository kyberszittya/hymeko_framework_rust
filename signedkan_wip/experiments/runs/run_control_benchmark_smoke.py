"""HSIKAN vs LQR vs Pure Pursuit vs MPC — lateral-tracking smoke.

Trains HSIKAN by imitation of LQR on a training track (sinusoid),
then evaluates all four controllers on a held-out test track
(S-curve lane change).  Reports per-controller metrics.

Usage::

    python -m signedkan_wip.experiments.runs.run_control_benchmark_smoke
    python -m signedkan_wip.experiments.runs.run_control_benchmark_smoke \\
        --train-epochs 300 --window 12 --jsonl-out /tmp/ctrl.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--train-epochs", type=int, default=200)
    ap.add_argument("--train-lr", type=float, default=5e-3)
    ap.add_argument("--jsonl-out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    params = BicycleParams()
    vehicle = BicycleVehicle(params)

    # Initial pose: start a bit off the path so all controllers actually
    # *need* to correct.
    init = BicycleState(x=0.0, y=0.5, psi=0.0, v=params.v_target)

    print(f"=== Lateral-tracking benchmark (seed={args.seed}, dt={args.dt}, T={args.T}s) ===")

    # ─── Build controllers ────────────────────────────────────────────
    lqr     = LQRController(params, q_y=1.0, q_psi=1.0, r=0.5)
    pp      = PurePursuitController(params, lookahead=6.0)
    mpc     = MPCController(params, horizon=8, dt=args.dt)
    policy  = HSIKANPolicy(window=args.window, delta_max=params.delta_max)
    hsikan  = HSIKANController(params, policy=policy, window=args.window)

    # ─── Imitation train HSIKAN — multi-init to mitigate distribution
    #     shift (Pomerleau's behaviour-cloning failure mode). ─────────
    print(f"\n--- training HSIKAN by imitation of LQR (multi-init) ---")
    train_tracks = {
        "sinusoid": sinusoid_track(length=100.0, amplitude=2.0, wavelength=30.0),
        "straight": straight_track(length=100.0),
    }
    # 16 perturbed starts spread across (y_off, psi_off).
    inits = [
        BicycleState(x=0.0, y=dy, psi=dpsi, v=params.v_target)
        for dy in (-1.0, -0.5, 0.0, 0.5, 1.0)
        for dpsi in (-0.15, 0.0, 0.15)
    ]
    Xs, Ss, ys = [], [], []
    for tk_label, tk in train_tracks.items():
        for st0 in inits:
            Xi, Si, yi = collect_imitation_dataset(
                teacher=LQRController(params, q_y=1.0, q_psi=1.0, r=0.5),
                track=tk, vehicle=vehicle,
                initial_state=st0, dt=args.dt, T=args.T,
                window=args.window,
            )
            Xs.append(Xi); Ss.append(Si); ys.append(yi)
    X = torch.cat(Xs, dim=0); S = torch.cat(Ss, dim=0); y = torch.cat(ys, dim=0)
    print(f"  dataset: feats={tuple(X.shape)}  sigma={tuple(S.shape)}  actions={tuple(y.shape)}")
    train_metrics = imitation_train_hsikan(
        policy, X, S, y, epochs=args.train_epochs, lr=args.train_lr,
    )
    print(f"  imitation MSE: train={train_metrics['train_mse_final']:.5f}  "
          f"val={train_metrics['val_mse_final']:.5f}")

    # ─── Evaluate all four on three tracks ───────────────────────────
    tracks = {
        "straight":  straight_track(length=100.0),
        "sinusoid":  sinusoid_track(length=100.0, amplitude=2.0, wavelength=30.0),
        "s_curve":   s_curve_track(length=100.0, lane_width=3.5),
    }
    controllers = {
        "lqr":           lqr,
        "pure_pursuit":  pp,
        "mpc":           mpc,
        "hsikan":        hsikan,
    }

    rows = []
    for tk_name, tk in tracks.items():
        print(f"\n--- track: {tk_name} ---")
        for ctrl_name, ctrl in controllers.items():
            m = run_episode(ctrl, tk, vehicle, init, dt=args.dt, T=args.T)
            row = {
                "track": tk_name,
                "controller": ctrl_name,
                "lat_rmse": m.lat_rmse,
                "lat_max": m.lat_max,
                "final_pos_err": m.final_pos_err,
                "delta_rmse": m.delta_rmse,
                "wall_per_step_ms": m.wall_per_step_ms,
                "n_steps": m.n_steps,
            }
            rows.append(row)
            print(f"  {ctrl_name:14s}  RMSE={m.lat_rmse:.4f} m  "
                  f"max={m.lat_max:.4f} m  "
                  f"δ_rmse={m.delta_rmse:.4f} rad  "
                  f"wall={m.wall_per_step_ms:.2f} ms/step")

    if args.jsonl_out:
        out = Path(args.jsonl_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(rows)} rows to {out}")

    # ─── Summary table ───────────────────────────────────────────────
    print()
    print(f"=== SUMMARY (lateral RMSE in m; lower = better) ===")
    ctrl_names = list(controllers.keys())
    print(f"{'track':14s} " + " ".join(f"{c:>14s}" for c in ctrl_names))
    for tk_name in tracks:
        line = f"{tk_name:14s} "
        for c in ctrl_names:
            hit = next((r for r in rows if r["track"] == tk_name and r["controller"] == c), None)
            line += f" {hit['lat_rmse']:>14.4f}"
        print(line)


if __name__ == "__main__":
    main()
