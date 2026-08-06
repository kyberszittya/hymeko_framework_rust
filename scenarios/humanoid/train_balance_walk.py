"""Direct locomotion on the position-servo humanoid — a DIFFERENT approach than the WBC footstep env.

Instead of the whole-body-controller footstep machinery, this drives the humanoid straight through
``balance_env``'s position-servo action (a bounded joint-target offset), with a forward-velocity reward
(``w_velocity``). A CEM search over a small linear policy asks: can the humanoid move forward by direct
joint control alone? Checkpointed, deterministic per seed. Usage:
    PYTHONPATH=. python -m scenarios.humanoid.train_balance_walk --iters 50 --pop 48 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv
from scenarios.humanoid.cem import cem_optimize
from scenarios.humanoid.cem import linear_policy as policy
from scenarios.humanoid.cem import policy_dim as _dim


def walk_cfg(max_steps: int, w_velocity: float = 6.0) -> BalanceConfig:
    return BalanceConfig(max_steps=max_steps, w_velocity=w_velocity, perturb_lo=0.0, perturb_hi=0.0)


def rollout(theta: np.ndarray, cfg: BalanceConfig, seed: int = 0) -> "tuple[float, int, float]":
    """One episode; returns (return, steps survived, net forward pelvis displacement)."""
    env = HumanoidBalanceEnv(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    x0 = float(env.data.xpos[env._pelvis, 0])
    ret, steps = 0.0, 0
    for _ in range(cfg.max_steps):
        obs, r, fell, trunc, _i = env.step(policy(theta, obs, od, ad).astype(np.float32))
        ret += r
        steps += 1
        if fell or trunc:
            break
    return ret, steps, float(env.data.xpos[env._pelvis, 0]) - x0


def _eval(args: "tuple[np.ndarray, BalanceConfig]") -> "tuple[float, int, float]":
    return rollout(*args)


def train(iters: int, pop: int, elite: int, cfg: BalanceConfig, workers: int, out: Path) -> np.ndarray:
    env0 = HumanoidBalanceEnv(cfg, seed=0)
    dim = _dim(env0.observation_space.shape[0], env0.action_space.shape[0])
    best_theta, best = cem_optimize(_eval, cfg, dim, iters=iters, pop=pop, elite=elite, workers=workers,
                                    sigma0=0.3, out=out, label="bwalk",
                                    report=lambda b: f"best_fwd={b[2]:+.3f}")
    np.save(out / "best_policy.npy", best_theta)
    (out / "result.json").write_text(json.dumps({
        "best_ret": best[0], "best_steps": best[1], "best_fwd": best[2], "iters": iters, "pop": pop},
        indent=2))
    print(f"[bwalk] DONE best_fwd={best[2]:+.3f} steps={best[1]}", flush=True)
    return best_theta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=int(os.environ.get("HYMEKO_ITERS", "50")))
    ap.add_argument("--pop", type=int, default=int(os.environ.get("HYMEKO_POP", "48")))
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("HYMEKO_WORKERS", "1")))
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--w_velocity", type=float, default=6.0)
    ap.add_argument("--out", type=str, default=os.environ.get("HYMEKO_OUT", "experiments/humanoid_bwalk"))
    args = ap.parse_args()
    cfg = walk_cfg(args.max_steps, w_velocity=args.w_velocity)
    train(args.iters, args.pop, args.elite, cfg, args.workers, Path(args.out))


if __name__ == "__main__":
    main()
