"""Functional sanity for the collaborative CTDE reframe.

Does the cooperative CTDE policy (per-arm action heads + centralized critic over a shared HSiKAN backbone)
train and deliver comparably to the single-agent HSiKAN on Galambos? Both are behaviour-cloned from the same
cooperative demonstrator and evaluated on delivery rate.

This checks only that the multi-agent reframe is **sound and drop-in** (the per-arm head decomposition does not
break BC/eval). The *structural* payoff of multi-agent training (per-agent exploration, decentralized credit)
needs a real CTDE trainer with per-agent rollouts — BC supervises the joint action and so cannot exercise it.

    python -m hymeko_rl.exp_collaborative --seed 0
    python -m hymeko_rl.exp_collaborative --smoke
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from hymeko_rl.bc import behaviour_clone
from hymeko_rl.collaborative import build_collaborative
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.galambos_bc import collect_galambos_demos, eval_delivery
from hymeko_rl.policy import build_policy
from hymeko_rl.ppo import PPOConfig, train_ppo


def _refine(policy: Any, env: PlanarGraspEnv, refine: int, seed: int, *, bc_coef: float = 0.0,
            bc_data: "tuple[Any, Any] | None" = None) -> None:
    """PPO-refine a BC-warm-started policy in place (the CTDE policy trains as cooperative CTDE under the shared
    trainer — it exposes act/value/evaluate). ``refine`` = PPO iterations; 0 = BC only. ``bc_coef`` keeps the
    refine near the demos (the BC anchor cross-ported from pick-place)."""
    if refine > 0:
        train_ppo(policy, env, PPOConfig(n_iters=refine, n_steps=2048, seed=seed, ent_coef=0.003,
                                         bc_coef=bc_coef), bc_data=bc_data)


def _env(difficulty: float, task_graph: bool) -> PlanarGraspEnv:
    return PlanarGraspEnv(robot=None, max_steps=300, difficulty=difficulty, task_graph=task_graph)


def run(*, seed: int, n_demos: int, bc_epochs: int, hidden: int, difficulty: float, n_eval: int,
        task_graph: bool, refine: int = 0, gif_dir: str | None = None, bc_coef: float = 0.0) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = _env(difficulty, task_graph)
    feat = int(env.observation_space.shape[1])   # type: ignore[index]
    n_act = env.n_actions
    obs, acts = collect_galambos_demos(env, n_demos, seed)
    out: dict[str, Any] = {"seed": seed, "task_graph": task_graph, "n_demo_steps": int(obs.shape[0])}

    hs = build_policy("hsikan", obs_dim=feat, action_dim=n_act, hg_state=env.hg, hidden=hidden)
    behaviour_clone(hs, obs, acts, n_epochs=bc_epochs, seed=seed)
    _refine(hs, env, refine, seed, bc_coef=bc_coef, bc_data=(obs, acts))
    out["hsikan_single"] = {"deliv": eval_delivery(_env(difficulty, task_graph), hs, n_eval, 9000),
                            "params": hs.n_parameters()}

    coll = build_collaborative(env, kind="hsikan", hidden=hidden)
    behaviour_clone(coll, obs, acts, n_epochs=bc_epochs, seed=seed)   # type: ignore[arg-type]  # duck-typed on action_mean
    _refine(coll, env, refine, seed, bc_coef=bc_coef, bc_data=(obs, acts))
    out["collab_ctde"] = {"deliv": eval_delivery(_env(difficulty, task_graph), coll, n_eval, 9000),
                          "params": coll.n_parameters(), "agents": list(coll.agent_action_dims)}

    if gif_dir is not None:                            # §9 animated output: the two arms cooperating (+ single)
        from pathlib import Path

        from hymeko_rl.campaign_viz import render_actor_gif
        Path(gif_dir).mkdir(parents=True, exist_ok=True)
        for name, pol in (("single_hsikan", hs), ("collab_ctde", coll)):
            try:
                render_actor_gif(_env(difficulty, task_graph), pol, f"{gif_dir}/coin_toss_{name}", seed=20_000)
            except Exception as exc:   # noqa: BLE001 — viz is best-effort; a GL/camera failure must not fail the run
                print(f"  [gif {name} skipped: {type(exc).__name__}: {exc}]", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-demos", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--task-graph", action="store_true")
    ap.add_argument("--refine", type=int, default=0, help="PPO iterations to refine past BC (0 = BC only)")
    ap.add_argument("--gif", default=None, help="render GIFs of the trained policies into this dir (§9 animated)")
    ap.add_argument("--bc-coef", type=float, default=0.0, help="BC-anchor weight for the PPO refine (cross-port)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_demos, a.bc_epochs, a.n_eval = 40, 40, 8
    print(json.dumps(run(seed=a.seed, n_demos=a.n_demos, bc_epochs=a.bc_epochs, hidden=a.hidden,
                         difficulty=a.difficulty, n_eval=a.n_eval, task_graph=a.task_graph, refine=a.refine,
                         gif_dir=a.gif, bc_coef=a.bc_coef), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
