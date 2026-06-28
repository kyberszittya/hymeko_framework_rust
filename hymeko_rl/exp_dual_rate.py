"""Discriminating experiment for Kato's reflexive/deliberative split (the rate-asymmetric dual loop).

Two questions, on Galambos via behaviour cloning of the scripted demonstrator:
  1. Does the **dual** controller (MLP reflex ⊕ HSiKAN deliberation) beat **MLP-alone** and **HSiKAN-alone**?
  2. Does running the deliberation every **N>1** steps (held in between) preserve the dual's delivery rate —
     i.e. is the slow loop's compute saving free?

All trained identically (same demos, BC epochs, hidden); delivery rate is the metric. ``dual_N1`` deliberates
every step; ``dual_N4`` / ``dual_N8`` use the :class:`RateAsymmetricLoop` (deliberation held for N steps).

    python -m hymeko_rl.exp_dual_rate --seed 0 --n-demos 200 --bc-epochs 200
    python -m hymeko_rl.exp_dual_rate --smoke
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from hymeko_rl.bc import behaviour_clone
from hymeko_rl.dual_rate import DualRateController, RateAsymmetricLoop, build_dual_rate
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.galambos_bc import collect_galambos_demos, eval_delivery
from hymeko_rl.policy import build_policy


@torch.no_grad()
def eval_delivery_rate_asym(env: PlanarGraspEnv, controller: DualRateController, n: int,
                            n_episodes: int, seed: int) -> float:
    """Greedy delivery rate with the deliberation recomputed every ``n`` steps (held otherwise), reset per episode."""
    loop = RateAsymmetricLoop(controller, deliberate_every=n)
    succ = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        loop.reset()
        delivered = False
        for _ in range(env.max_steps):
            action = loop.act_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy()
            obs, _r, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
            delivered = delivered or bool(info["in_zone"])
            if term or trunc:
                break
        succ += int(delivered)
    return succ / max(1, n_episodes)


def _env(difficulty: float, task_graph: bool = False) -> PlanarGraspEnv:
    return PlanarGraspEnv(robot=None, max_steps=300, difficulty=difficulty, task_graph=task_graph)


def run(*, seed: int, n_demos: int, bc_epochs: int, hidden: int, difficulty: float, n_eval: int,
        task_graph: bool = False) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = _env(difficulty, task_graph)
    feat = int(env.observation_space.shape[1])   # type: ignore[index]
    flat = env.hg.n_vertices * feat
    n_act = env.n_actions
    obs, acts = collect_galambos_demos(env, n_demos, seed)
    out: dict[str, Any] = {"seed": seed, "task_graph": task_graph, "n_demo_steps": int(obs.shape[0]),
                           "n_vertices": int(env.hg.n_vertices)}

    mlp = build_policy("mlp", obs_dim=flat, action_dim=n_act, hidden=hidden)
    behaviour_clone(mlp, obs, acts, n_epochs=bc_epochs, seed=seed)
    out["mlp"] = {"deliv": eval_delivery(_env(difficulty, task_graph),mlp, n_eval, 9000), "params": mlp.n_parameters()}

    hs = build_policy("hsikan", obs_dim=feat, action_dim=n_act, hg_state=env.hg, hidden=hidden)
    behaviour_clone(hs, obs, acts, n_epochs=bc_epochs, seed=seed)
    out["hsikan"] = {"deliv": eval_delivery(_env(difficulty, task_graph),hs, n_eval, 9000), "params": hs.n_parameters()}

    dual = build_dual_rate(obs_feat=feat, action_dim=n_act, hg_state=env.hg, n_vertices=env.hg.n_vertices,
                           reflex_hidden=hidden, delib_hidden=hidden, skip="highway")
    behaviour_clone(dual, obs, acts, n_epochs=bc_epochs, seed=seed)   # type: ignore[arg-type]  # duck-typed on action_mean
    out["dual_N1"] = {"deliv": eval_delivery(_env(difficulty, task_graph),dual, n_eval, 9000), "params": dual.n_parameters()}
    out["dual_N4"] = {"deliv": eval_delivery_rate_asym(_env(difficulty, task_graph),dual, 4, n_eval, 9000)}
    out["dual_N8"] = {"deliv": eval_delivery_rate_asym(_env(difficulty, task_graph),dual, 8, n_eval, 9000)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-demos", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--n-eval", type=int, default=24)
    ap.add_argument("--task-graph", action="store_true",
                    help="put coin+zone in the graph (grasp/goal hyperedges) — the structural-leverage arm")
    ap.add_argument("--smoke", action="store_true", help="tiny demos/epochs/eval — a quick end-to-end check")
    a = ap.parse_args()
    if a.smoke:
        a.n_demos, a.bc_epochs, a.n_eval = 40, 40, 8
    print(json.dumps(run(seed=a.seed, n_demos=a.n_demos, bc_epochs=a.bc_epochs, hidden=a.hidden,
                         difficulty=a.difficulty, n_eval=a.n_eval, task_graph=a.task_graph), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
