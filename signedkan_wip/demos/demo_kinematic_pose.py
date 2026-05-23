"""Fast demo: HSiKAN per-vertex pose regression on a kinematic
mechanism graph.

Trains briefly on synthetic Stewart / 4-bar / delta_3rrr fixtures,
then predicts XYZ coordinates of every link in a fresh test
mechanism.  Total runtime < 30 s on cuda, < 60 s on cpu.

Usage:
    HSIKAN_TORCH_COMPILE=1 python -m signedkan_wip.demos.demo_kinematic_pose

Optional flags:
    --arity {4,6}   — 4-bar (k=4) or Stewart/delta (k=6)
    --n-train 80    — training mechanism count
    --epochs 80     — training epochs
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from signedkan_wip.experiments.runs.run_phase11_kinematic_tasks import detect_dominant_arity
from signedkan_wip.experiments.runs.run_phase12_position_regression import (
    build_mechanism_with_positions, PositionRegHSiKAN, _build_input,
)


def fmt_xyz(v):
    return f"({v[0]:+.3f}, {v[1]:+.3f}, {v[2]:+.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arity", type=int, default=6, choices=[4, 6])
    ap.add_argument("--n-train", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-vertices", type=int, default=8,
                    help="how many vertices to dump in the per-vertex table")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== HSiKAN pose-regression demo ===")
    print(f"  device: {device}  arity: k={args.arity}  hidden: {args.hidden}")
    print(f"  n_train: {args.n_train}  n_test: {args.n_test}  "
          f"epochs: {args.epochs}\n")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # ---- Generate train + test mechanisms ----
    t0 = time.perf_counter()
    train_raw = [build_mechanism_with_positions(rng) for _ in range(args.n_train)]
    test_raw  = [build_mechanism_with_positions(rng) for _ in range(args.n_test)]
    cands_tr = [(g, p, fam) for g, p, fam in train_raw
                 if detect_dominant_arity(g) == args.arity]
    cands_te = [(g, p, fam) for g, p, fam in test_raw
                 if detect_dominant_arity(g) == args.arity]
    if not cands_tr or not cands_te:
        print(f"  no mechanisms with dominant arity k={args.arity}; "
              f"try --arity 4")
        return
    n_nodes_max = max(g.n_nodes for g, _, _ in cands_tr + cands_te)
    print(f"  matched k={args.arity} mechanisms: "
          f"{len(cands_tr)} train, {len(cands_te)} test  "
          f"(n_nodes_max={n_nodes_max})")

    def pack(g, pos):
        inp = _build_input(g, args.arity, 30000, device, args.seed,
                              n_nodes_max)
        if inp is None: return None
        pos_pad = np.zeros((n_nodes_max, 3), dtype=np.float32)
        pos_pad[:pos.shape[0]] = pos
        mask = np.zeros(n_nodes_max, dtype=np.float32)
        mask[:g.n_nodes] = 1.0
        return (inp,
                  torch.from_numpy(pos_pad).to(device),
                  torch.from_numpy(mask).to(device),
                  g.n_nodes)

    train_inputs = [pack(g, p) for g, p, _ in cands_tr]
    train_inputs = [x for x in train_inputs if x is not None]
    test_inputs = [pack(g, p) for g, p, _ in cands_te]
    test_inputs = [x for x in test_inputs if x is not None]

    print(f"  data prep: {(time.perf_counter() - t0):.2f}s\n")

    # ---- Train ----
    model = PositionRegHSiKAN(n_nodes_max=n_nodes_max,
                                  arity=args.arity,
                                  hidden=args.hidden,
                                  n_layers=2, grid=3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print(f"  model: PositionRegHSiKAN h={args.hidden} L=2 grid=3  "
          f"({n_params:,} params)")
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_inputs))
        ep_loss = 0.0
        for i in perm:
            inp, pos, mask, _ = train_inputs[i]
            pred = model(inp)
            err = (pred - pos) * mask.unsqueeze(-1)
            loss = err.pow(2).sum() / max(mask.sum().item(), 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % 20 == 0:
            print(f"    epoch {ep+1:>3d}  train_loss={ep_loss/len(train_inputs):.4f}")
    train_time = time.perf_counter() - t0
    print(f"  train: {train_time:.2f}s\n")

    # ---- Eval (aggregate MAE) ----
    model.eval()
    all_mae = []
    with torch.no_grad():
        for inp, pos, mask, _ in test_inputs:
            pred = model(inp)
            err = (pred - pos) * mask.unsqueeze(-1)
            mae = (err.abs().sum() / max(mask.sum().item(), 1.0) / 3.0).item()
            all_mae.append(mae)
    mae_mean = float(np.mean(all_mae))
    mae_std = float(np.std(all_mae))
    print(f"  test MAE: {mae_mean:.4f} ± {mae_std:.4f}  "
          f"(n_test={len(test_inputs)})")

    # ---- Pick one mechanism, time inference, dump per-vertex ----
    inp_one, pos_one, mask_one, n_nodes_real = test_inputs[0]

    # Warmup + per-call latency
    for _ in range(10):
        with torch.no_grad(): _ = model(inp_one)
    if device.type == "cuda": torch.cuda.synchronize()
    samples = []
    for _ in range(30):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(): _ = model(inp_one)
        if device.type == "cuda": torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    import statistics
    lat_ms = statistics.median(samples) * 1000
    print(f"  inference latency (median, single mechanism): {lat_ms:.2f} ms\n")

    with torch.no_grad():
        pred_one = model(inp_one).cpu().numpy()
    true_one = pos_one.cpu().numpy()
    err_one = np.abs(pred_one - true_one)

    print(f"  per-vertex prediction (first {min(args.show_vertices, n_nodes_real)} "
          f"of {n_nodes_real} real vertices):")
    print(f"  {'vid':>4}  {'predicted XYZ (m)':<26}  "
          f"{'true XYZ (m)':<26}  {'L2 err (m)':>10}")
    print("  " + "-" * 76)
    for i in range(min(args.show_vertices, n_nodes_real)):
        l2 = float(np.linalg.norm(err_one[i]))
        print(f"  {i:>4d}  {fmt_xyz(pred_one[i]):<26}  "
              f"{fmt_xyz(true_one[i]):<26}  {l2:>10.4f}")
    avg_l2 = float(np.mean(np.linalg.norm(err_one[:n_nodes_real], axis=1)))
    print(f"  {'':>4}  {'':>26}  {'mean L2 over all':>26}  {avg_l2:>10.4f}\n")
    print(f"  total demo wall-clock (training + eval + dump): "
          f"{train_time + 0.5:.1f}s")


if __name__ == "__main__":
    main()
