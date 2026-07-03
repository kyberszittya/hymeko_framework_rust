"""PPO for the FANUC-arm pick-and-place (Phase 1, path A — the learned policy BC could not deliver).

BC fits the demos but rolls out 0% (compounding over the 620-step horizon). PPO is on-policy, so it trains on
the policy's OWN states and corrects exactly that covariate shift. From-scratch PPO on a 7-DOF arm would never
discover a grasp by random exploration, so we **warm-start from the BC clone** (imitation → RL): the clone
reaches + descends, PPO reinforces the grasp/lift/place around its own rollouts. Reuses the shared pick helpers
(``build``/``collect_demos``/``eval_success``), the env's dense approach→grasp→lift→place reward, and the
generic :func:`hymeko_rl.train.ppo.train_ppo` (the same trainer the reach + cart-pole scenarios use).

    python -m hymeko_rl.experiments.pick_place_ppo --kind hsikan --iters 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.env.pick_place_env import PickPlaceEnv
from hymeko_rl.experiments.gripper_pick_bc import build, collect_demos, eval_success
from hymeko_rl.train.ppo import PPOConfig, train_ppo
from hymeko_rl.viz.render_pick_place import fanuc_pick_env


def run_pick_place_ppo(kind: str = "hsikan", *, hidden: int = 64, seed: int = 0, n_eval: int = 8,
                       cfg: PPOConfig | None = None, pretrain_demos: int = 16,
                       pretrain_epochs: int = 120, curriculum: bool = False,
                       out: "str | None" = None, reward_profile: "str | None" = None,
                       n_envs: int = 1) -> dict[str, float | str | int]:
    """BC-warm-start (optional) then PPO on the FANUC pick env; report lift/place vs the untrained floor and
    the post-BC pre-PPO baseline.

    # Postconditions returns init/final PPO return + lift/place rates over ``n_eval`` held-out seeds.
    """
    cfg = cfg or PPOConfig(n_iters=80, n_steps=2048, value_warmup=5, ent_coef=0.002, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = fanuc_pick_env(reward_profile=reward_profile)         # training reward (eval measures success directly)
    ac = build(kind, env, hidden)
    floor_lift, floor_place = eval_success(fanuc_pick_env(), ac, n_eval, seed=10_000)
    pre_lift = pre_place = float("nan")
    bc_data: "tuple[np.ndarray, np.ndarray] | None" = None
    if pretrain_demos > 0:
        obs, acts = collect_demos(env, pretrain_demos, seed, only_success=True)
        behaviour_clone(ac, obs, acts, n_epochs=pretrain_epochs, seed=seed)
        bc_data = (obs, acts)                                  # demos kept as the BC anchor for PPO (cfg.bc_coef)
        pre_lift, pre_place = eval_success(fanuc_pick_env(), ac, n_eval, seed=15_000)
    # curriculum: ramp the env difficulty (object → reach edge, wider angles) across PPO iterations.
    on_iter = (lambda i, n: env.set_difficulty(i / max(1, n - 1))) if curriculum else None
    # n_envs > 1 → vectorised rollout (N lock-step envs, batched policy forward): the throughput lever on this
    # CPU box. The worker envs are built fresh from the same reward profile; curriculum is single-env only.
    def _mk() -> "PickPlaceEnv":
        return fanuc_pick_env(reward_profile=reward_profile)
    history = train_ppo(ac, env, cfg, on_iteration=on_iter, bc_data=bc_data,
                        n_envs=n_envs, make_env=(_mk if n_envs > 1 else None))
    lift, place = eval_success(fanuc_pick_env(), ac, n_eval, seed=20_000)
    result: dict[str, float | str | int] = dict(policy=kind, n_params=ac.n_parameters(),
                  warm_start=pretrain_demos > 0,
                  init_return=round(history[0], 2), final_return=round(history[-1], 2),
                  untrained_lift=round(floor_lift, 3), post_bc_lift=round(pre_lift, 3),
                  post_bc_place=round(pre_place, 3), ppo_lift=round(lift, 3), ppo_place=round(place, 3))
    if out is not None:                                         # persist weights + metrics for rendering/plots
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ac.state_dict(), p.with_suffix(".pt"))
        p.with_suffix(".json").write_text(json.dumps({**result, "returns": history}, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="hsikan", choices=("hsikan", "mlp", "mixture", "sa_hsikan"))
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--pretrain-demos", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="save weights (.pt) + metrics (.json) to this path stem")
    ap.add_argument("--reward", default=None,
                    help="success-aligned reward .hymeko profile (e.g. data/robotics/pick_place_aligned.hymeko)")
    ap.add_argument("--bc-coef", type=float, default=0.0,
                    help="BC-anchor weight: keep PPO near the demos so the warm-started skill is not lost")
    ap.add_argument("--n-envs", type=int, default=1, help="vectorised rollout width (N lock-step envs; the speedup)")
    a = ap.parse_args(argv)
    cfg = PPOConfig(n_iters=a.iters, n_steps=a.n_steps, value_warmup=5, ent_coef=0.002, seed=a.seed,
                    bc_coef=a.bc_coef)
    print(json.dumps(run_pick_place_ppo(a.kind, seed=a.seed, cfg=cfg, pretrain_demos=a.pretrain_demos,
                                        out=a.out, reward_profile=a.reward, n_envs=a.n_envs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
