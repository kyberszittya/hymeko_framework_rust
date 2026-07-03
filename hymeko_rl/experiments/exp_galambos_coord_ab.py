"""Coordination-reward A/B on the collaborative off-policy galambos scenario.

Baseline (``galambos_task.hymeko``) vs Coordination (``galambos_task_coord.hymeko`` = baseline +
``both_approach 4.0``). Identical collab CTDE off-policy setup on both arms
(:func:`build_collaborative_offpolicy`, ``sa_hsikan``, TD3+BC, best-checkpoint on delivery).

Hypothesis (measured): the baseline peaks at delivery 0.40 with ``both_contact ≈ 0.019`` — the two arms
almost never grip the coin *simultaneously*, and the two-arm ``coin_frictionloss`` needs simultaneous
force. ``both_approach = -max(left,right)`` penalises the lagging arm, the coordination gradient the
compensable mean ``grasp_approach`` lacks. Discriminating question: does ``both_contact`` (→ delivery) rise?

    python -m hymeko_rl.experiments.exp_galambos_coord_ab            # full overnight (3 seeds × 200k × 2)
    python -m hymeko_rl.experiments.exp_galambos_coord_ab --smoke    # 1 seed × ~3k, path check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.agents.multichannel_ctde import build_collaborative_offpolicy
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.eval.evaluate import greedy_action_fn
from hymeko_rl.experiments.galambos_bc import collect_galambos_demos, eval_delivery
from hymeko_rl.train.campaign import Campaign, CampaignConfig

_COORD_HYMEKO = "data/robotics/galambos_task_coord.hymeko"
_MAX_STEPS = 300
_EVAL_SEED = 9_000


def make_env(*, coord: bool, difficulty: float) -> PlanarGraspEnv:
    """A planar grasping env; ``coord=True`` swaps in the coordination reward (defined in its own
    ``.hymeko``, per the reward-in-hymeko rule — no in-memory term surgery)."""
    env = PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=difficulty)
    if coord:
        env.reward_spec = RewardSpec.from_hymeko(_COORD_HYMEKO)
    return env


def _both_contact_rate(env: PlanarGraspEnv, actor: Any, n_episodes: int, seed: int) -> float:
    """Fraction of steps in which BOTH fingertips touch the coin — the coordination metric the A/B tests."""
    act = greedy_action_fn(actor)
    steps = both = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _r, term, trunc, _info = env.step(act(env, obs))
            m = getattr(env, "_planar_metrics", None)
            steps += 1
            both += int(m is not None and m.left_contact and m.right_contact)
            done = bool(term or trunc)
    return both / max(1, steps)


def measure_factory(*, coord: bool, difficulty: float, n_eval: int):
    """A Campaign ``measure(make_env, actor) -> {delivery, both_contact}`` closed over the variant."""
    def _measure(_make_env: Any, actor: Any) -> "dict[str, float]":
        deliv = eval_delivery(make_env(coord=coord, difficulty=difficulty), actor, n_eval, _EVAL_SEED)
        bc = _both_contact_rate(make_env(coord=coord, difficulty=difficulty), actor, min(n_eval, 12), _EVAL_SEED)
        return {"delivery": deliv, "both_contact": bc}
    return _measure


def run(*, difficulty: float, smoke: bool) -> "dict[str, Any]":
    seeds = (0,) if smoke else (0, 1, 2)
    total_steps = 3_000 if smoke else 200_000
    eval_every = 1_500 if smoke else 25_000
    n_demos = 12 if smoke else 200
    bc_epochs = 3 if smoke else 200
    n_eval = 3 if smoke else 50

    summary: dict[str, Any] = {"difficulty": difficulty, "smoke": smoke, "variants": {}}
    for name, coord in (("baseline", False), ("coord", True)):
        cfg = CampaignConfig(
            name=f"galambos_coord_ab_{name}", select="delivery", seeds=seeds,
            total_steps=total_steps, eval_every=eval_every, n_demos=n_demos, bc_epochs=bc_epochs, n_eval=n_eval)
        camp = Campaign(
            cfg,
            make_env=lambda coord=coord: make_env(coord=coord, difficulty=difficulty),
            build=lambda env: build_collaborative_offpolicy(env, kind="sa_hsikan", hidden=64),
            measure=measure_factory(coord=coord, difficulty=difficulty, n_eval=n_eval),
            demos=collect_galambos_demos,
            gif=not smoke,
        )
        summary["variants"][name] = camp.run()
    return summary


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--smoke", action="store_true", help="1 seed × ~3k steps for a path check")
    args = ap.parse_args(argv)
    summary = run(difficulty=args.difficulty, smoke=args.smoke)
    print("\n=== A/B SUMMARY ===")
    for name, res in summary["variants"].items():
        peak = res.get("peak_delivery_median", res.get("peak_median", "?"))
        print(f"  {name:9s}: peak delivery median = {peak}")
    print(json.dumps(summary, indent=2, default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
