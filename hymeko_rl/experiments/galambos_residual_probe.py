"""Residual Galambos probe: does a tiny learned delta preserve the scripted floor?

This is diagnostic, not a headline scenario result. The base controller is the push demonstrator. The actor
controls only a bounded residual delta through ResidualControllerEnv, so a zero actor reproduces the scripted
controller exactly. Two cells are useful:

* frozen: zero actor, critic-only updates. This checks that the RL loop and residual wrapper do not corrupt the
  scripted behavior.
* tiny: zero actor, short critic warmup, then actor updates inside a small delta band. This is the open test:
  does a trained correction hold the floor or collapse?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.env.residual import ResidualControllerEnv
from hymeko_rl.eval.evaluate import DwellMetric, eval_metric, experiment_dir, greedy_action_fn
from hymeko_rl.experiments.galambos_demo import PushDemonstrator
from hymeko_rl.train.ddpg import build_offpolicy, td3_config, train_offpolicy


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def make_residual_env(*, delta_scale: float, difficulty: float = 0.3, max_steps: int = 300
                      ) -> ResidualControllerEnv:
    inner = PlanarGraspEnv(robot=None, max_steps=max_steps, difficulty=difficulty)
    return ResidualControllerEnv(inner, PushDemonstrator(inner), delta_scale=delta_scale)


def build_mlp(env: Any, *, hidden: int) -> tuple[Any, list[Any]]:
    shape = env.observation_space.shape
    assert shape is not None
    n_vertices, feat = int(shape[0]), int(shape[1])
    scale = float(np.max(np.abs(env.action_space.high)))
    return build_offpolicy("mlp", obs_dim=feat, flat_dim=n_vertices * feat, action_dim=env.n_actions,
                           action_scale=scale, n_critics=2, hidden=hidden)


def zero_actor(actor: torch.nn.Module) -> None:
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()


def evaluate_delivery(actor: Any, *, delta_scale: float, n_eval: int, seed: int) -> dict[str, float]:
    env = make_residual_env(delta_scale=delta_scale)
    try:
        dwell = int(getattr(env, "success_steps", 1))
        delivery = eval_metric(env, greedy_action_fn(actor), DwellMetric("in_zone", dwell),
                               n_episodes=n_eval, seed0=seed)
        both = steps = 0
        act_fn = greedy_action_fn(actor)
        for ep in range(n_eval):
            obs, _ = env.reset(seed=seed + 10_000 + ep)
            for _ in range(env.max_steps):
                obs, _r, term, trunc, info = env.step(act_fn(env, obs))
                both += int(bool(info.get("both_contact", False)))
                steps += 1
                if term or trunc:
                    break
        return {"delivery": float(sum(delivery)) / max(1, n_eval),
                "both_contact": float(both) / max(1, steps)}
    finally:
        _close_env(env)


def run_cell(*, name: str, total_steps: int, critic_warmup: int, delta_scale: float,
             noise_scale: float, n_eval: int, seed: int, hidden: int, n_envs: int) -> dict[str, Any]:
    env = make_residual_env(delta_scale=delta_scale)
    actor, critics = build_mlp(env, hidden=hidden)
    zero_actor(actor)
    pre = evaluate_delivery(actor, delta_scale=delta_scale, n_eval=n_eval, seed=9_000)

    def eval_fn(_env: Any, ac: Any) -> float:
        return evaluate_delivery(ac, delta_scale=delta_scale, n_eval=n_eval, seed=9_000)["delivery"]

    cfg = td3_config(total_steps=total_steps, start_steps=0, batch_size=256, capacity=50_000,
                     eval_every=total_steps, n_eval=n_eval, seed=seed, warm_start=True,
                     critic_warmup=critic_warmup, noise_scale=noise_scale, log_every=total_steps,
                     device="cpu")
    history = train_offpolicy(actor, critics, env, cfg, eval_fn=eval_fn,
                              n_envs=n_envs, make_env=lambda: make_residual_env(delta_scale=delta_scale))
    post = evaluate_delivery(actor, delta_scale=delta_scale, n_eval=n_eval, seed=9_000)
    return {"name": name, "pre": pre, "post": post, "curve": history,
            "config": {"total_steps": total_steps, "critic_warmup": critic_warmup,
                       "delta_scale": delta_scale, "noise_scale": noise_scale,
                       "n_eval": n_eval, "seed": seed, "hidden": hidden, "n_envs": n_envs}}


def run_probe(*, total_steps: int = 6_000, delta_scale: float = 0.03, n_eval: int = 25,
              seed: int = 0, hidden: int = 32, n_envs: int = 4) -> dict[str, Any]:
    frozen = run_cell(name="frozen_zero_residual", total_steps=total_steps,
                      critic_warmup=total_steps + 1, delta_scale=delta_scale, noise_scale=0.0,
                      n_eval=n_eval, seed=seed, hidden=hidden, n_envs=n_envs)
    tiny = run_cell(name="tiny_trained_residual", total_steps=total_steps,
                    critic_warmup=min(1_000, max(0, total_steps // 4)),
                    delta_scale=delta_scale, noise_scale=0.05,
                    n_eval=n_eval, seed=seed, hidden=hidden, n_envs=n_envs)
    out = {"frozen_zero_residual": frozen, "tiny_trained_residual": tiny}
    d = experiment_dir("experiments", f"galambos_residual_probe_{total_steps}_s{seed}")
    (d / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["dir"] = str(d)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=6_000)
    ap.add_argument("--delta-scale", type=float, default=0.03)
    ap.add_argument("--n-eval", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-envs", type=int, default=4)
    args = ap.parse_args(argv)
    print(json.dumps(run_probe(total_steps=args.steps, delta_scale=args.delta_scale,
                               n_eval=args.n_eval, seed=args.seed, hidden=args.hidden,
                               n_envs=args.n_envs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
