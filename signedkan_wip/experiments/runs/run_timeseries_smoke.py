"""Smoke benchmark for time-series forecasting.

Trains four models — LinearAR, MLP, GRU, HSIKANSeq — on four
basic-dataset cells (sine, noisy_sine, mackey_glass, lorenz_x).
Prints per-model val MSE + param count, and (optional) writes a
jsonl summary.

Usage::

    python -m signedkan_wip.experiments.runs.run_timeseries_smoke
    python -m signedkan_wip.experiments.runs.run_timeseries_smoke \\
        --datasets mackey_glass --epochs 50 --jsonl-out /tmp/ts.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from signedkan_wip.src.timeseries import (
    DATASETS,
    MODELS,
    TSConfig,
    load,
    windowed,
)


def _split_train_val(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2):
    n = X.shape[0]
    n_val = max(1, int(n * val_frac))
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]


def _train_one(model: nn.Module, X_tr: torch.Tensor, y_tr: torch.Tensor,
                X_va: torch.Tensor, y_va: torch.Tensor,
                epochs: int, lr: float, batch_size: int,
                device: torch.device) -> Dict[str, float]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n_tr = X_tr.shape[0]
    losses: List[float] = []
    best_val = float("inf")
    for ep in range(epochs):
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch_size):
            idx = perm[i : i + batch_size]
            x_b = X_tr[idx]
            y_b = y_tr[idx]
            opt.zero_grad()
            pred = model(x_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_va = model(X_va)
            val_mse = float(loss_fn(pred_va, y_va).detach())
        model.train()
        losses.append(val_mse)
        best_val = min(best_val, val_mse)
    return {
        "val_mse_final": losses[-1],
        "val_mse_best": best_val,
        "val_mse_start": losses[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    ap.add_argument("--models", nargs="*", default=list(MODELS.keys()))
    ap.add_argument("--n-steps", type=int, default=4096)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--noise-std", type=float, default=0.05)
    ap.add_argument("--jsonl-out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    rows: List[dict] = []
    print(f"=== HSIKAN time-series smoke ===")
    print(f"  datasets={args.datasets}  models={args.models}")
    print(f"  window={args.window}  horizon={args.horizon}  "
          f"epochs={args.epochs}  device={device}")

    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            raise KeyError(f"unknown dataset {ds_name!r}")
        cfg = TSConfig(n_steps=args.n_steps, seed=args.seed,
                        noise_std=args.noise_std)
        series, dt = load(ds_name, cfg)
        X_np, y_np = windowed(series, args.window, horizon=args.horizon)
        X_tr_np, y_tr_np, X_va_np, y_va_np = _split_train_val(X_np, y_np)
        X_tr = torch.from_numpy(X_tr_np).to(device)
        y_tr = torch.from_numpy(y_tr_np).to(device)
        X_va = torch.from_numpy(X_va_np).to(device)
        y_va = torch.from_numpy(y_va_np).to(device)

        print(f"\n--- dataset={ds_name}  n_train={X_tr.shape[0]}  "
              f"n_val={X_va.shape[0]}  dt={dt:g} ---")

        for model_name in args.models:
            if model_name not in MODELS:
                raise KeyError(f"unknown model {model_name!r}")
            torch.manual_seed(args.seed)
            if model_name in ("linear_ar", "hsikan_seq"):
                model = MODELS[model_name](window=args.window)
            elif model_name == "mlp":
                model = MODELS[model_name](window=args.window, hidden=32)
            elif model_name == "gru":
                model = MODELS[model_name](hidden=16)
            else:
                model = MODELS[model_name]()
            n_params = sum(p.numel() for p in model.parameters())
            t0 = time.perf_counter()
            metrics = _train_one(model, X_tr, y_tr, X_va, y_va,
                                  args.epochs, args.lr, args.batch_size, device)
            wall = time.perf_counter() - t0
            row = {
                "dataset": ds_name,
                "model": model_name,
                "n_params": n_params,
                "wall_s": wall,
                "window": args.window,
                "horizon": args.horizon,
                "epochs": args.epochs,
                "seed": args.seed,
                **metrics,
            }
            rows.append(row)
            print(f"  {model_name:12s}  params={n_params:>6d}  "
                  f"val_mse_best={metrics['val_mse_best']:.4f}  "
                  f"val_mse_final={metrics['val_mse_final']:.4f}  "
                  f"wall={wall:5.1f}s")

    if args.jsonl_out:
        out = Path(args.jsonl_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(rows)} rows to {out}")

    # Tabular summary
    print()
    print(f"=== SUMMARY (val_mse_best lower=better) ===")
    print(f"{'dataset':14s} " + " ".join(f"{m:>14s}" for m in args.models))
    for ds_name in args.datasets:
        line = f"{ds_name:14s} "
        for model_name in args.models:
            hit = next((r for r in rows if r["dataset"] == ds_name and r["model"] == model_name), None)
            if hit:
                line += f" {hit['val_mse_best']:>14.4f}"
            else:
                line += f" {'-':>14s}"
        print(line)


if __name__ == "__main__":
    main()
