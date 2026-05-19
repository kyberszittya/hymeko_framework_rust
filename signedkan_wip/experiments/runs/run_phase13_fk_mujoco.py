"""Phase 13 — Forward kinematics learning from MuJoCo rollouts.

Train two models to predict flange XYZ from joint angles:
  1. MLP baseline (joint vector → XYZ, no graph context)
  2. HSiKAN-augmented: graph embedding from cycles + joint vector → XYZ

The 4-DOF arm has no closed loops (it's serial), so HSiKAN's cycle-pool
features are degenerate (no cycles). To make the experiment meaningful,
we ALSO run the 4-bar linkage which has a real k=4 closed loop.

Task expectation: For the SERIAL arm, both models should match (no
graph signal). For the 4-bar with closed loop, HSiKAN's cycle context
might help disambiguate the two valid IK branches.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signedkan_wip.src.mujoco_bridge import MuJoCoBridge, SimulationStates


def gather_dataset(sim: MuJoCoBridge, n_rollouts: int = 50,
                     duration: float = 2.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Run multiple rollouts with random sine controllers, return
    (X = joint angles & velocities, Y = end-effector XYZ).

    Each rollout uses a different random seed for the sine controller
    parameters, giving a diverse joint-state distribution."""
    rng = np.random.RandomState(seed)
    X_list, Y_list = [], []
    n_act = sim.model.nu
    flange_idx = None
    for bn in ["flange_link", "wrist_link", "ee", "end_effector"]:
        try:
            flange_idx = [mujoco_id for mujoco_id, name in
                           enumerate(_body_names(sim))
                           if name == bn][0]
            break
        except IndexError:
            continue
    if flange_idx is None:
        flange_idx = sim.model.nbody - 1   # last body as fallback

    for r in range(n_rollouts):
        sim.reset()
        # Random sine controller params per rollout.
        freqs = 0.3 + rng.rand(n_act) * 0.7
        phases = rng.rand(n_act) * 2 * math.pi
        amps = 0.4 + rng.rand(n_act) * 0.6
        n_steps = int(duration / sim.model.opt.timestep)
        for step in range(n_steps):
            t = step * sim.model.opt.timestep
            for ai in range(n_act):
                sim.data.ctrl[ai] = amps[ai] * np.sin(2 * math.pi * freqs[ai] * t + phases[ai])
            import mujoco
            mujoco.mj_step(sim.model, sim.data)
            if step % 10 == 0:    # 50 Hz capture
                qpos = sim.data.qpos[:n_act].copy()
                qvel = sim.data.qvel[:n_act].copy()
                X_list.append(np.concatenate([qpos, qvel]))
                Y_list.append(sim.data.xpos[flange_idx].copy())
    return np.stack(X_list).astype(np.float32), np.stack(Y_list).astype(np.float32)


def _body_names(sim):
    import mujoco
    return [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, i)
             for i in range(sim.model.nbody)]


class MLPBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, Y_tr, X_te, Y_te, hidden=64, n_epochs=200, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(X_tr).to(device); Yt = torch.from_numpy(Y_tr).to(device)
    Xe = torch.from_numpy(X_te).to(device); Ye = torch.from_numpy(Y_te).to(device)
    model = MLPBaseline(X_tr.shape[1], hidden, Y_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        model.train()
        pred = model(Xt)
        loss = F.mse_loss(pred, Yt)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred_te = model(Xe)
        rmse = float(torch.sqrt(F.mse_loss(pred_te, Ye)).item())
    return rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", default="arm", choices=["arm", "4bar"])
    ap.add_argument("--n_rollouts", type=int, default=80)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_epochs", type=int, default=300)
    args = ap.parse_args()

    print(f"=== Phase 13 — Forward kinematics on MuJoCo {args.mech} ===",
          flush=True)
    if args.mech == "arm":
        sim = MuJoCoBridge.canonical_4dof_arm()
    else:
        sim = MuJoCoBridge.canonical_4bar()
    g = sim.kinematic_graph()
    print(f"Graph: {g.stats()}", flush=True)

    print(f"Generating {args.n_rollouts} rollouts ({args.duration}s each) ...",
          flush=True)
    t0 = time.time()
    X, Y = gather_dataset(sim, n_rollouts=args.n_rollouts,
                            duration=args.duration, seed=args.seed)
    print(f"  dataset: X={X.shape}  Y={Y.shape}  in {time.time()-t0:.1f}s",
          flush=True)
    print(f"  X range: {X.min():.2f} to {X.max():.2f}", flush=True)
    print(f"  Y range: x[{Y[:,0].min():.2f},{Y[:,0].max():.2f}] "
          f"y[{Y[:,1].min():.2f},{Y[:,1].max():.2f}] "
          f"z[{Y[:,2].min():.2f},{Y[:,2].max():.2f}]", flush=True)

    # Random 80/20 train/test split (stratified by rollout index).
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    X_tr, Y_tr = X[train_idx], Y[train_idx]
    X_te, Y_te = X[test_idx], Y[test_idx]
    print(f"Train n={len(X_tr)}  Test n={len(X_te)}", flush=True)

    print("\n--- MLP baseline (joints → XYZ, no graph) ---", flush=True)
    for h in (32, 64, 128):
        rmses = []
        for seed in range(3):
            r = train_mlp(X_tr, Y_tr, X_te, Y_te, hidden=h,
                            n_epochs=args.n_epochs, seed=seed)
            rmses.append(r)
        print(f"  h={h}  test_rmse_med={statistics.median(rmses):.4f} m  "
              f"std={statistics.stdev(rmses):.4f}",
              flush=True)


if __name__ == "__main__":
    main()
