"""Multi-seed evaluation of a Galambos planar-grasp policy — a real success rate with a CI.

Loads a checkpoint, rolls the deterministic policy over N distinct held-out seeds, and reports the
goal / death / timeout tally + the goal-rate 95% Wilson confidence interval (so 5/8 becomes a rate
with error bars rather than an anecdote).

    uv run python -m hymeko_rl.eval_planar_grasp --checkpoint checkpoints/galambos/ppo_strategy.pt -n 100
"""
from __future__ import annotations

import argparse
import math

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.evaluate import evaluate
from hymeko_rl.render_planar_gifs import load_policy, policy_action_fn


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion ``k/n`` (robust at the extremes)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("-n", "--n-episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=2000, help="first eval seed (held out from training)")
    ap.add_argument("--max-steps", type=int, default=160)
    a = ap.parse_args(argv)

    env = PlanarGraspEnv.from_hymeko(max_steps=a.max_steps)
    ac = load_policy(a.checkpoint, env)
    stats = evaluate(env, policy_action_fn(ac), source="hsikan", n_episodes=a.n_episodes,
                     seed0=a.seed0)
    lo, hi = wilson_ci(stats.goals, stats.n_episodes)
    print(f"checkpoint : {a.checkpoint}")
    print(f"episodes   : {stats.n_episodes} (seeds {a.seed0}..{a.seed0 + a.n_episodes - 1})")
    print(f"goals      : {stats.goals}  deaths {stats.deaths}  timeouts {stats.timeouts}")
    print(f"goal rate  : {stats.success_rate:.1%}  (95% Wilson CI {lo:.1%}-{hi:.1%})")
    print(f"mean return: {stats.mean_return:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
