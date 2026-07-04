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


def make_env(*, coord: bool, difficulty: float, treatment_hymeko: str = _COORD_HYMEKO) -> PlanarGraspEnv:
    """A planar grasping env; ``coord=True`` swaps in the treatment reward from ``treatment_hymeko`` (defined in
    its own ``.hymeko``, per the reward-in-hymeko rule — no in-memory term surgery)."""
    env = PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=difficulty)
    if coord:
        env.reward_spec = RewardSpec.from_hymeko(treatment_hymeko)
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


def measure_factory(*, coord: bool, difficulty: float, n_eval: int, treatment_hymeko: str = _COORD_HYMEKO):
    """A Campaign ``measure(make_env, actor) -> {delivery, both_contact}`` closed over the variant. Eval always
    uses the BASELINE env (coord=False) so delivery is scored on the true task, not the shaped training reward."""
    def _measure(_make_env: Any, actor: Any) -> "dict[str, float]":
        deliv = eval_delivery(make_env(coord=False, difficulty=difficulty), actor, n_eval, _EVAL_SEED)
        bc = _both_contact_rate(make_env(coord=False, difficulty=difficulty), actor, min(n_eval, 12), _EVAL_SEED)
        return {"delivery": deliv, "both_contact": bc}
    return _measure


def run(*, difficulty: float, smoke: bool, n_demos: "int | None" = None, total_steps: "int | None" = None,
        bc_epochs: "int | None" = None, seeds: "tuple[int, ...] | None" = None,
        variants: "tuple[str, ...]" = ("baseline", "coord"),
        treatment_hymeko: str = _COORD_HYMEKO, treatment_name: str = "coord") -> "dict[str, Any]":
    seeds = seeds if seeds is not None else ((0,) if smoke else (0, 1, 2))
    total_steps = total_steps if total_steps is not None else (3_000 if smoke else 200_000)
    n_demos = n_demos if n_demos is not None else (12 if smoke else 200)
    bc_epochs = bc_epochs if bc_epochs is not None else (3 if smoke else 200)
    eval_every = 1_500 if smoke else max(25_000, total_steps // 8)
    n_eval = 3 if smoke else 50

    _all = {"baseline": False, treatment_name: True}
    variants = tuple(treatment_name if v == "coord" else v for v in variants)
    # The scripted demonstrator ignores the reward and the training seed-init, so its demos are identical
    # across seeds AND across the baseline/coord variants — collect once, reuse (a closure cache, not a
    # module global, §6.5 #11). Saves re-rolling ~16k samples per seed.
    _demo_cache: "dict[int, tuple[np.ndarray, np.ndarray]]" = {}

    def cached_demos(env: Any, n: int, seed: int) -> "tuple[np.ndarray, np.ndarray]":
        if n not in _demo_cache:
            _demo_cache[n] = collect_galambos_demos(env, n, seed)
        return _demo_cache[n]

    summary: dict[str, Any] = {"difficulty": difficulty, "smoke": smoke,
                               "budget": {"n_demos": n_demos, "total_steps": total_steps, "bc_epochs": bc_epochs,
                                          "seeds": list(seeds)}, "variants": {}}
    for name in variants:
        coord = _all[name]
        cfg = CampaignConfig(
            name=f"galambos_coord_ab_{name}", select="delivery", seeds=seeds,
            total_steps=total_steps, eval_every=eval_every, n_demos=n_demos, bc_epochs=bc_epochs, n_eval=n_eval)
        camp = Campaign(
            cfg,
            make_env=lambda coord=coord: make_env(coord=coord, difficulty=difficulty,
                                                  treatment_hymeko=treatment_hymeko),
            build=lambda env: build_collaborative_offpolicy(env, kind="sa_hsikan", hidden=64),
            measure=measure_factory(coord=coord, difficulty=difficulty, n_eval=n_eval,
                                    treatment_hymeko=treatment_hymeko),
            demos=cached_demos,
            gif=not smoke,
        )
        summary["variants"][name] = camp.run()
    return summary


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--smoke", action="store_true", help="1 seed × ~3k steps for a path check")
    ap.add_argument("--demos", type=int, default=None, help="rollout episodes for BC (only ~30%% deliver & are kept)")
    ap.add_argument("--steps", type=int, default=None, help="off-policy training steps per seed")
    ap.add_argument("--bc-epochs", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--variant", choices=["baseline", "coord", "both"], default="both")
    ap.add_argument("--treatment-hymeko", type=str, default=_COORD_HYMEKO,
                    help="reward .hymeko for the treatment arm (e.g. data/robotics/galambos_task_deliver.hymeko)")
    ap.add_argument("--treatment-name", type=str, default="coord", help="label for the treatment arm")
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None
    variants = ("baseline", "coord") if args.variant == "both" else (args.variant,)
    summary = run(difficulty=args.difficulty, smoke=args.smoke, n_demos=args.demos, total_steps=args.steps,
                  bc_epochs=args.bc_epochs, seeds=seeds, variants=variants,
                  treatment_hymeko=args.treatment_hymeko, treatment_name=args.treatment_name)
    print("\n=== A/B SUMMARY ===")
    for name, res in summary["variants"].items():
        peak = res.get("peak_delivery_median", res.get("peak_median", "?"))
        print(f"  {name:9s}: peak delivery median = {peak}")
    print(json.dumps(summary, indent=2, default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
