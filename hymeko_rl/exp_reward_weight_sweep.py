"""Reward-weight sweep: does scaling one reward term improve coin delivery?

For each weight in the sweep, the Galambos reward's chosen ``--term`` (default ``in_zone``, the sparse
success bonus) is overridden to that value while every other term is held at the refined
``galambos_task.hymeko`` default; a single HSiKAN policy is BC warm-started from the scripted demonstrator
and then TD3-refined (stable now: gradient clip + reward normalisation). Each setting reports the BC and the
post-refine delivery rate and renders a GIF, so a reward magnitude is settled by data + watchable behaviour
rather than a guess. BC delivery is reward-independent (it clones actions) → it is the same across weights
and serves as the baseline the refine must beat; only the RL number responds to the weight.

    python -m hymeko_rl.exp_reward_weight_sweep --term in_zone --weights 10 50 200 --refine 15000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.bc import behaviour_clone
from hymeko_rl.ddpg import build_offpolicy, td3_config, train_offpolicy
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.galambos_bc import collect_galambos_demos, eval_delivery

_TASK = "data/robotics/galambos_task.hymeko"


def reward_with_term(term: str, weight: float) -> RewardSpec:
    """The refined Galambos reward with ``term``'s weight overridden to ``weight`` (others unchanged).

    # Preconditions ``term`` appears in the galambos reward spec. # Errors ``ValueError`` if it does not."""
    base = RewardSpec.from_hymeko(_TASK)
    if term not in {k for k, _ in base.terms}:
        raise ValueError(f"term {term!r} not in galambos reward; have {[k for k, _ in base.terms]}")
    return RewardSpec(tuple((k, weight if k == term else w) for k, w in base.terms))


def _env(spec: RewardSpec, difficulty: float) -> PlanarGraspEnv:
    return PlanarGraspEnv(robot=None, max_steps=300, difficulty=difficulty, reward_spec=spec)


def run(*, term: str, weights: list[float], seed: int, n_demos: int, bc_epochs: int, refine: int,
        difficulty: float, n_eval: int, gif_dir: str) -> dict[str, Any]:
    Path(gif_dir).mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for w in weights:
        torch.manual_seed(seed)
        np.random.seed(seed)
        spec = reward_with_term(term, w)
        env = _env(spec, difficulty)
        feat = int(env.observation_space.shape[1])   # type: ignore[index]
        scale = float(np.max(np.abs(env.action_space.high)))   # type: ignore[attr-defined]
        actor, critics = build_offpolicy("hsikan", obs_dim=feat, flat_dim=env.hg.n_vertices * feat,
                                         action_dim=env.n_actions, action_scale=scale, n_critics=2,
                                         hidden=64, hg_state=env.hg)
        obs, acts = collect_galambos_demos(env, n_demos, seed)
        behaviour_clone(actor, obs, acts, n_epochs=bc_epochs, seed=seed)   # type: ignore[arg-type]
        bc_deliv = eval_delivery(_env(spec, difficulty), actor, n_eval, 9000)
        cfg = td3_config(total_steps=refine, seed=seed, warm_start=True, critic_warmup=2000,
                         noise_scale=0.05)
        train_offpolicy(actor, critics, env, cfg)   # type: ignore[arg-type]  # env duck-typed (planar grasp)
        rl_deliv = eval_delivery(_env(spec, difficulty), actor, n_eval, 9000)
        results[str(int(w))] = {"bc_deliv": round(bc_deliv, 4), "rl_deliv": round(rl_deliv, 4)}
        print(f"  {term}={w:g}: BC {bc_deliv:.3f} -> TD3 {rl_deliv:.3f}", flush=True)
        try:
            from hymeko_rl.campaign_viz import render_actor_gif
            render_actor_gif(_env(spec, difficulty), actor, f"{gif_dir}/{term}_{int(w)}", seed=20_000)
        except Exception as exc:   # noqa: BLE001 — viz is best-effort; a GL/camera failure must not fail the sweep
            print(f"  [gif {term}_{int(w)} skipped: {type(exc).__name__}: {exc}]", flush=True)
    return {"term": term, "seed": seed, "refine": refine, "results": results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--term", default="in_zone", help="reward term whose weight to sweep")
    ap.add_argument("--weights", type=float, nargs="+", default=[10.0, 50.0, 200.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-demos", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--refine", type=int, default=15_000, help="TD3 refine steps per weight")
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--gif-dir", default="reports/gifs/inzone_sweep")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    a = ap.parse_args(argv)
    res = run(term=a.term, weights=a.weights, seed=a.seed, n_demos=a.n_demos, bc_epochs=a.bc_epochs,
              refine=a.refine, difficulty=a.difficulty, n_eval=a.n_eval, gif_dir=a.gif_dir)
    print(json.dumps(res, indent=2), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
