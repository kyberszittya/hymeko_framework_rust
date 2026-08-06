"""Direct-locomotion SAC — the STRONGER learner for forward walking (vs the CEM baselines).

Reuses the project SAC (`hymeko_rl.train.sac`) on `balance_env`'s position-servo action with the
forward-velocity reward (`w_velocity`). Where CEM over a linear policy barely advanced, an off-policy
SAC over an MLP actor is far more sample-efficient — this asks whether it walks forward faster / cleaner.
The checkpoint is selected by held-out **forward distance** (not a balance certificate). CPU torch.

Usage:  PYTHONPATH=. python -m scenarios.humanoid.run_humanoid_walk_sac --steps 80000 --out <dir>
SIMULATION. Live [sac] progress; checkpointed (never-run-blind, §3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.train.sac import AlphaMode, SACConfig, build_sac, train_sac

from .balance_env import BalanceConfig, HumanoidBalanceEnv


def _greedy(actor, obs) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return actor.action_mean(t).squeeze(0).cpu().numpy()


def _forward_distance(env: HumanoidBalanceEnv, act_fn, seeds) -> float:
    """Mean net forward pelvis displacement (m) of a greedy policy over `seeds`."""
    nets = []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        x0 = float(env.data.xpos[env._pelvis, 0])
        done = False
        while not done:
            obs, _r, term, trunc, _i = env.step(act_fn(obs))
            done = term or trunc
        nets.append(float(env.data.xpos[env._pelvis, 0]) - x0)
    return float(np.mean(nets))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--w_velocity", type=float, default=8.0)   # >1 makes forward the dominant reward term
    ap.add_argument("--vel_cap", type=float, default=0.6)      # forward-velocity reward cap (m/s)
    ap.add_argument("--delta_scale", type=float, default=0.4)  # joint-target action authority (rad)
    ap.add_argument("--torque", action="store_true",          # DIRECT torque action (sustains a gait) vs servo
                    help="direct gravity-compensated torque action instead of the position-servo")
    ap.add_argument("--w_stability", type=float, default=0.0)  # viability-band healthy hinge (keeps it upright)
    ap.add_argument("--model_src", type=str, default="humanoid.hymeko")  # "humanoid_toe.hymeko" = toe push-off
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--out", type=str, default="experiments/humanoid_walk_sac")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_cfg = BalanceConfig(perturb_lo=0.0, perturb_hi=0.0, w_velocity=args.w_velocity,
                              vel_cap=args.vel_cap, delta_scale=args.delta_scale,
                              torque_action=args.torque, w_stability=args.w_stability,
                              model_src=args.model_src, max_steps=args.max_steps)
    env = HumanoidBalanceEnv(cfg=train_cfg, seed=0)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    torch.manual_seed(0)
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                               action_dim=act_dim, action_scale=1.0, hidden=128)

    val_seeds = list(range(2000, 2004))
    test_seeds = list(range(3000, 3006))
    base_fwd = _forward_distance(env, lambda _o: np.zeros(act_dim), test_seeds)  # PD-hold scaffold (a=0)

    best_path = out / "walk_sac_best.pt"
    best = {"fwd": -1e9}

    def eval_fn(e, a) -> float:
        fwd = _forward_distance(e, lambda o: _greedy(a, o), val_seeds)  # selection = forward distance on VAL
        if fwd > best["fwd"]:
            best["fwd"] = fwd
            torch.save(a.state_dict(), best_path)
        return fwd

    cfg = SACConfig(total_steps=args.steps, start_steps=2_000, batch_size=256,
                    eval_every=10_000, log_every=5_000, seed=0,
                    alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn)

    if not best_path.exists():
        torch.save(actor.state_dict(), best_path)
    best_actor, _ = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                              action_dim=act_dim, action_scale=1.0, hidden=128)
    best_actor.load_state_dict(torch.load(best_path))
    best_fwd = _forward_distance(env, lambda o: _greedy(best_actor, o), test_seeds)
    result = {
        "pd_hold_forward_test": round(base_fwd, 4),
        "sac_best_val_forward_val": round(best["fwd"], 4),
        "sac_best_val_forward_test": round(best_fwd, 4),
        "forward_delta_test": round(best_fwd - base_fwd, 4),
        "eval_curve_forward_val": [round(c, 4) for c in curve],
        "total_steps": args.steps,
        "note": "SIMULATION. Direct-locomotion SAC over balance_env's position-servo action with the "
                "forward-velocity reward. Checkpoint selected by forward distance on VAL; TEST reported once. "
                "Baseline = PD-hold (a=0) on the identical TEST seeds.",
    }
    (out / "walk_sac_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
