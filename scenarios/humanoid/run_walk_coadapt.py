"""Co-adaptation loop for the viability-gated walk — the fix for the one-shot gating's distribution shift.

Learn the viability boundary from the CURRENT policy, retrain the gated policy, re-learn the boundary,
repeat, so the certificate tracks the policy's state distribution instead of lagging it
(reports/2026-08-06-humanoid-viability-gate.md). Run on the PLAIN torque walker (the most sustained
baseline) so the gate's effect is not drowned out by the periodic+toe stack.

Each iteration: train a gated SAC (gated by the previous iteration's boundary) → measure survival →
learn the boundary from THIS policy for the next iteration. Iteration 0 is ungated (the baseline).

Usage:  PYTHONPATH=. python -m scenarios.humanoid.run_walk_coadapt --iters 3 --steps 80000 --out <dir>
SIMULATION. Checkpointed per iteration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.train.sac import build_sac

from .balance_env import BalanceConfig, HumanoidBalanceEnv
from .run_humanoid_walk_sac import train_walk_sac
from .viability_gate import learn_from_policy


def _load_actor(cfg: BalanceConfig, path):
    env = HumanoidBalanceEnv(cfg, seed=0)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    actor, _ = build_sac("mlp", obs_dim=od, flat_dim=od, action_dim=ad, action_scale=1.0, hidden=128)
    actor.load_state_dict(torch.load(path))
    return env, actor


def _policy_fn(actor):
    def pf(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return pf


def _measure(cfg: BalanceConfig, path, seeds, cap: int = 4000) -> "tuple[float, float]":
    """Mean (survival seconds, forward distance m) of the greedy policy — the gate is reward-only, so the
    trajectory is unaffected by it; measure on the ungated env dynamics."""
    env, actor = _load_actor(cfg, path)
    pf = _policy_fn(actor)
    dt = float(env.model.opt.timestep)
    times, dists = [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        x0 = float(env.data.xpos[env._pelvis, 0])
        k = 0
        for k in range(cap):
            obs, _r, term, trunc, _i = env.step(pf(obs))
            if term or trunc:
                break
        times.append((k + 1) * dt)
        dists.append(float(env.data.xpos[env._pelvis, 0]) - x0)
    return float(np.mean(times)), float(np.mean(dists))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--w_velocity", type=float, default=50.0)
    ap.add_argument("--horizon", type=int, default=200)     # viability fall-labelling horizon (steps)
    ap.add_argument("--out", type=str, default="experiments/humanoid_coadapt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # PLAIN torque config (no toe / periodic / healthy) — the most sustained baseline.
    base = dict(perturb_lo=0.0, perturb_hi=0.0, torque_action=True, w_velocity=args.w_velocity,
                vel_cap=1.0, delta_scale=0.7, max_steps=600)
    plain = BalanceConfig(**base)                            # ungated env for measuring + boundary fitting
    fit_seeds = list(range(40, 62))
    test_seeds = list(range(3000, 3006))

    summary: "list[dict]" = []
    boundary_path = ""                                       # iteration 0 is ungated
    for it in range(args.iters):
        it_out = out / f"iter{it}"
        cfg = BalanceConfig(**base, viability_boundary=boundary_path)
        best_path, result = train_walk_sac(cfg, args.steps, it_out)
        surv_s, dist_m = _measure(plain, best_path, test_seeds)
        env, actor = _load_actor(plain, best_path)
        _b, acc = learn_from_policy(env, _policy_fn(actor), fit_seeds,
                                    horizon=args.horizon, out_path=it_out / "viab.npz")
        boundary_path = str(it_out / "viab.npz")             # next iteration is gated by THIS policy's boundary
        row = {"iter": it, "gated_by": "none" if it == 0 else f"iter{it - 1}",
               "survival_s": round(surv_s, 3), "distance_m": round(dist_m, 3),
               "fwd_test": result["sac_best_val_forward_test"], "boundary_acc": round(acc, 3)}
        summary.append(row)
        print("[coadapt]", json.dumps(row), flush=True)

    (out / "coadapt_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
