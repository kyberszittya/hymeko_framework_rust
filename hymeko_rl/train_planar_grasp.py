"""Train PPO on the Galambos planar grasper, saving the policy + the return curve.

    python -m hymeko_rl.train_planar_grasp --iters 150 --steps 512 --out checkpoints/galambos/ppo

The HSiKAN policy message-passes over the two-arm kinematic hypergraph; the reward is the
declarative ``galambos_task.hymeko`` spec. Pure PPO (the in-repo ``ppo.py``), no scripted expert.
Sizes the policy from the env's observation space (the planar obs is not the reach obs profile).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.policy import build_policy
from hymeko_rl.ppo import PPOConfig, train_ppo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--out", default="checkpoints/galambos/ppo")
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    env = PlanarGraspEnv(max_steps=a.max_steps)
    obs_shape = env.observation_space.shape
    assert obs_shape is not None
    feat = int(obs_shape[-1])
    ac = build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg,
                      hidden=a.hidden)
    cfg = PPOConfig(n_iters=a.iters, n_steps=a.steps, seed=a.seed)
    # PlanarGraspEnv is duck-compatible with the PPO loop (gym 5-tuple + node-feature obs); the
    # ppo.py signature names ArmReachEnv but reads only the shared surface.
    returns = train_ppo(ac, env, cfg)  # type: ignore[arg-type]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ac.state_dict(), out.with_suffix(".pt"))
    out.with_suffix(".json").write_text(json.dumps(
        {"returns": returns, "iters": a.iters, "steps": a.steps, "seed": a.seed,
         "n_params": int(sum(p.numel() for p in ac.parameters()))}))
    print(f"trained {a.iters} iters; first {returns[0]:.2f} -> final {returns[-1]:.2f}; "
          f"saved {out}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
