"""Train PPO on the Galambos planar grasper — the whole thing from ``.hymeko``.

    python -m hymeko_rl.train.train_planar_grasp --out checkpoints/galambos/ppo

Robot + environment scene + reward come from ``.hymeko`` via ``PlanarGraspEnv.from_hymeko``; the PPO
**explore/exploit strategy** (entropy bonus, action-noise scale, curriculum, gamma/lam/clip/lr/…)
comes from ``galambos_strategy.hymeko`` via :class:`StrategySpec`. Tuning the exploration tactic is an
edit to that ``.hymeko``, not this file. The HSiKAN policy message-passes over the two-arm hypergraph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.agents.policy import build_policy
from hymeko_rl.train.ppo import train_ppo
from hymeko_rl.agents.strategy_spec import StrategySpec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategy", default="data/robotics/galambos_strategy.hymeko",
                    help="the .hymeko explore/exploit strategy (PPO config + curriculum + action noise)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--out", default="checkpoints/galambos/ppo")
    ap.add_argument("--checkpoint-every", type=int, default=0,
                    help="save the policy every K iters (0 = only at end) — for long, resumable runs")
    ap.add_argument("--resume", default=None,
                    help="load policy weights from this .pt before training (warm-start / continue)")
    a = ap.parse_args(argv)

    strat = StrategySpec.from_hymeko(a.strategy)
    torch.manual_seed(a.seed)
    # The whole MDP — robot + environment scene + reward — comes from .hymeko (three-path)…
    env = PlanarGraspEnv.from_hymeko(max_steps=a.max_steps)
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    feat = int(obs_shape[-1])
    # …and the policy's exploration noise + the PPO config + curriculum come from the strategy.
    ac = build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg,
                      hidden=a.hidden, log_std_init=strat.log_std_init)
    if a.resume:
        ac.load_state_dict(torch.load(Path(a.resume), map_location="cpu"))
        print(f"resumed policy weights from {a.resume}", flush=True)
    cfg = strat.ppo_config(seed=a.seed)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def on_iter(i: int, _n: int) -> None:
        # Reverse curriculum: the coin starts near the zone and the spawn region anneals out to the
        # full reachable table over the first `curriculum_iters` iterations (0 = off / full range).
        k = strat.curriculum_iters
        env.difficulty = min(1.0, i / k) if k > 0 else 1.0
        # Periodic checkpoint (resumable long runs + intermediate eval points): a numbered snapshot
        # plus the rolling latest. CLAUDE.md §4 — never rely on one uninterrupted multi-hour run.
        if a.checkpoint_every > 0 and i > 0 and i % a.checkpoint_every == 0:
            torch.save(ac.state_dict(), out.parent / f"{out.name}.{i}.pt")
            torch.save(ac.state_dict(), out.with_suffix(".pt"))
            print(f"  [checkpoint @ iter {i} -> {out.name}.{i}.pt]", flush=True)

    # PlanarGraspEnv is duck-compatible with the PPO loop (gym 5-tuple + node-feature obs).
    metrics: list[dict[str, float]] = []
    returns = train_ppo(ac, env, cfg, on_iteration=on_iter, metrics_out=metrics)  # type: ignore[arg-type]

    torch.save(ac.state_dict(), out.with_suffix(".pt"))
    out.with_suffix(".json").write_text(json.dumps(
        {"returns": returns, "metrics": metrics, "iters": cfg.n_iters, "steps": cfg.n_steps,
         "seed": a.seed, "strategy": a.strategy,
         "n_params": int(sum(p.numel() for p in ac.parameters()))}))
    print(f"trained {cfg.n_iters} iters; first {returns[0]:.2f} -> final {returns[-1]:.2f}; "
          f"saved {out}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
