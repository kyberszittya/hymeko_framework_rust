"""Train a TARGET-CONDITIONED footstep policy — one that steps WHERE commanded.

The fixed-gait forward walker (``train_footstep_walk``) walks forward but ignores a commanded foothold:
overriding its nominal breaks the gait (measured — it goes backward). This trains a policy over
RANDOMISED per-step forward targets, with the target in the observation and a foot-to-target reward, so
it learns to place the swing foot at the commanded stone — making the shared-A\* stepping-stone plan
(``stepping_stone_demo``) executable.

CPU-bound (MuJoCo + the WBC KKT solve per tick); parallel CEM over a small linear policy, checkpointed.
Deterministic per seed. Usage:
    PYTHONPATH=. python -m scenarios.humanoid.train_target_footstep --iters 40 --pop 24 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scenarios.humanoid.footstep_env import FootstepConfig, HumanoidFootstepEnv
from scenarios.humanoid.train_footstep_walk import _dim, policy

# The per-step forward target range (m ahead of the stance foot) the policy is trained to hit. Bounded
# to the model's dynamically-feasible stride band (measured ~0.005–0.025 forward per step); commanding
# beyond it is unreachable in one step, so the range stays within reach for clean target-following.
TARGET_LO, TARGET_HI = 0.008, 0.03


def target_cfg(max_footsteps: int, w_target: float = 6.0, residual_xy: float = 0.10) -> FootstepConfig:
    # forward_stride is irrelevant (the target overrides the nominal each step); reward = upright +
    # foot-to-target accuracy. residual_xy is wide enough that the action can reach a commanded target.
    return FootstepConfig(max_footsteps=max_footsteps, target_conditioned=True, w_target=w_target,
                          forward_stride=0.0, residual_xy=residual_xy, w_forward=0.0, fall_penalty=25.0,
                          swing_weight=140.0)


def rollout(theta: np.ndarray, cfg: FootstepConfig, seed: int = 0) -> "tuple[float, int, float]":
    """One episode of randomised forward targets; returns (return, steps, mean |foot − target|)."""
    env = HumanoidFootstepEnv(cfg, seed=seed)
    env.reset(seed=seed)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    rng = np.random.default_rng(seed + 9973)
    ret, steps, errs = 0.0, 0, []
    for _ in range(cfg.max_footsteps):
        stance_b = env._fl if env._stance == "L" else env._fr
        target = float(env.data.xpos[stance_b, 0]) + float(rng.uniform(TARGET_LO, TARGET_HI))
        env._plan_forward_x = target
        obs = env._obs()                                   # includes the target offset
        _o, r, done, trunc, info = env.step(policy(theta, obs, od, ad).astype(np.float32))
        ret += r
        steps = info["steps"]
        swung_b = env._fl if env._stance == "L" else env._fr
        errs.append(abs(float(env.data.xpos[swung_b, 0]) - target))
        if done or trunc:
            break
    return ret, steps, float(np.mean(errs)) if errs else 1.0


def _eval(args: "tuple[np.ndarray, FootstepConfig]") -> "tuple[float, int, float]":
    return rollout(*args)


def train(iters: int, pop: int, elite: int, cfg: FootstepConfig, workers: int, out: Path) -> np.ndarray:
    od = HumanoidFootstepEnv(cfg, seed=0).observation_space.shape[0]
    dim = _dim(od, 2)
    rng = np.random.default_rng(0)
    mu, sig = np.zeros(dim), np.ones(dim) * 0.5
    best = (-1e9, np.zeros(dim), 0, 1.0)                    # (ret, theta, steps, mean_err)
    out.mkdir(parents=True, exist_ok=True)
    journal = (out / "journal.jsonl").open("w")
    for it in range(iters):
        cand = mu + sig * rng.standard_normal((pop, dim))
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(_eval, [(c, cfg) for c in cand]))
        else:
            res = [rollout(c, cfg) for c in cand]
        scores = np.array([r[0] for r in res])
        idx = np.argsort(scores)[::-1][:elite]
        mu, sig = cand[idx].mean(0), cand[idx].std(0) + 0.04
        for c, (rr, ss, ee) in zip(cand, res):
            if rr > best[0]:
                best = (rr, c.copy(), ss, ee)
        row = {"iter": it, "elite_ret": float(scores[idx].mean()), "best_err": float(best[3])}
        journal.write(json.dumps(row) + "\n")
        journal.flush()
        print(f"[target] iter{it} elite_ret={row['elite_ret']:.1f} best_err={best[3]:.4f}", flush=True)
    journal.close()
    np.save(out / "best_policy.npy", best[1])
    (out / "result.json").write_text(json.dumps({
        "best_ret": best[0], "best_steps": best[2], "best_mean_target_err": best[3],
        "target_lo": TARGET_LO, "target_hi": TARGET_HI, "iters": iters, "pop": pop}, indent=2))
    print(f"[target] DONE best_err={best[3]:.4f} steps={best[2]}", flush=True)
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=int(os.environ.get("HYMEKO_ITERS", "40")))
    ap.add_argument("--pop", type=int, default=int(os.environ.get("HYMEKO_POP", "24")))
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("HYMEKO_WORKERS", "1")))
    ap.add_argument("--max_footsteps", type=int, default=30)
    ap.add_argument("--w_target", type=float, default=6.0)
    ap.add_argument("--out", type=str, default=os.environ.get("HYMEKO_OUT", "experiments/humanoid_target_footstep"))
    args = ap.parse_args()
    cfg = target_cfg(args.max_footsteps, w_target=args.w_target)
    train(args.iters, args.pop, args.elite, cfg, args.workers, Path(args.out))


if __name__ == "__main__":
    main()
