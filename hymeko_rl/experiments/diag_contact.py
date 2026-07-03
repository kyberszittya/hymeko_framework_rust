"""Diagnostic: are the galambos "deliveries" genuine grasps, or knocks?

The multi-seed verdict reports ~45% *delivery* (coin reaches the zone) but a *negative* return — which is suspect:
a clean grasp-and-place should net positive. ``evaluate`` only records the outcome (goal/death/timeout), never
contact. This trains a policy and re-rolls it with **per-step contact logging** (the env's ``info["both_contact"]``
= both fingertips touching the coin), classifying every episode as **delivered-with-grasp** (both fingers contacted
at some point) vs **delivered-by-knock** (the coin reached the zone with the fingers never both touching it) vs
death / timeout. That answers: did the fingertips ever contact the coin on a successful episode?

Reuses the galambos env factory + ``_build_actor`` + ``train_sac`` + ``_greedy_action_fn`` — no new training code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hymeko_rl.viz.campaign_viz import _greedy_action_fn
from hymeko_rl.experiments.exp_pernode_actor_ab import _build_actor, match_mlp_hidden
from hymeko_rl.train.sac import SACConfig, train_sac


def contact_rollout(env, action_fn, *, n_episodes: int, seed0: int = 40_000) -> dict[str, object]:
    """Roll out greedily, classifying each episode by outcome × whether both fingertips ever contacted the coin.
    # Postconditions counts sum to ``n_episodes``; ``contact_steps`` is total both-contact steps observed."""
    cls = {"delivered_grasp": 0, "delivered_knock": 0, "death": 0, "timeout": 0}
    contact_steps = 0
    contact_steps_when_delivered: list[int] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        contact_ever, ep_contact, terminated, truncated, death = False, 0, False, False, False
        while not (terminated or truncated):
            obs, _r, terminated, truncated, info = env.step(action_fn(env, obs))
            if info["both_contact"]:
                contact_ever = True
                ep_contact += 1
                contact_steps += 1
            death = bool(info["death"])
        if death:
            cls["death"] += 1
        elif terminated:                                   # success sustained in-zone (not a death-termination)
            cls["delivered_grasp" if contact_ever else "delivered_knock"] += 1
            contact_steps_when_delivered.append(ep_contact)
        else:
            cls["timeout"] += 1
    delivered = cls["delivered_grasp"] + cls["delivered_knock"]
    return dict(
        n_episodes=n_episodes, outcomes=cls,
        delivery_rate=round(delivered / n_episodes, 3),
        grasp_fraction_of_deliveries=(round(cls["delivered_grasp"] / delivered, 3) if delivered else None),
        any_grasp_delivery=cls["delivered_grasp"] > 0,
        median_contact_steps_per_delivery=(int(np.median(contact_steps_when_delivered))
                                           if contact_steps_when_delivered else 0),
        total_contact_steps=contact_steps)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=25_000)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="mlp", choices=["mlp", "hsikan"])
    ap.add_argument("--task-graph", action="store_true",
                    help="put the coin+zone in the graph (grasp/goal hyperedges; coin vertex carries its xy = "
                         "the shared coordination channel) — the structural grasp test")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--difficulty", type=float, default=0.3,
                    help="scenario difficulty (coin spawn spread); vary it to change the collaborative scenario")
    ap.add_argument("--design", default="none", choices=["none", "kuniform"],
                    help="augment the arm graph with a k-uniform design (the structural coin-toss k-sweep)")
    ap.add_argument("--k", type=int, default=2, help="k-uniform order when --design kuniform")
    ap.add_argument("--out-dir", default="reports/diag_contact")
    a = ap.parse_args(argv)
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv

    def env_factory() -> object:
        base = PlanarGraspEnv.from_hymeko(max_steps=160, difficulty=a.difficulty, task_graph=a.task_graph)
        if a.design == "kuniform":
            from hymeko_rl.experiments.exp_designed_control import DesignAugmentedEnv
            from hymeko_rl.control.hypergraph_designs import k_uniform_blocks
            n = base.hg.n_vertices                              # arm bodies; augment with n blocks of order k
            return DesignAugmentedEnv(base, k_uniform_blocks(n, a.k, n_blocks=n, seed=0))
        return base

    env = env_factory()
    cfg = SACConfig(total_steps=a.steps, eval_every=a.steps, n_eval=5, seed=a.seed)
    hidden = a.hidden if a.kind == "hsikan" else match_mlp_hidden("galambos", 64)[0]
    actor, critics = _build_actor(a.kind, a.kind, "pooled", "none", env, cfg, hidden=hidden)
    print(f"  [diag] training {a.kind} (task_graph={a.task_graph}) on galambos {a.steps} steps "
          f"(seed {a.seed})...", flush=True)
    train_sac(actor, critics, env, cfg)
    report = contact_rollout(env_factory(), _greedy_action_fn(actor), n_episodes=a.episodes)
    report["kind"], report["task_graph"], report["steps"] = a.kind, a.task_graph, a.steps
    report["design"], report["k"], report["difficulty"] = a.design, a.k, a.difficulty
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diag_contact.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
