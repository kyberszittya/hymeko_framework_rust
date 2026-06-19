"""Train a robot policy: reaching via behaviour cloning (Phase 1) or PPO (Phase 2).

    python -m hymeko_rl.train_robot_rl --task reach-bc  --policy hsikan
    python -m hymeko_rl.train_robot_rl --task reach-ppo --policy hsikan
    python -m hymeko_rl.train_robot_rl --task reach-ppo --policy mlp     # the ablation baseline

The HSiKAN policy reads the kinematic hypergraph; the MLP baseline reads the same
node-feature obs flattened. Same task, same algorithm — only the backbone differs.
"""
from __future__ import annotations

import argparse
import json

from hymeko_rl.bc import run_bc
from hymeko_rl.ppo import PPOConfig, run_ppo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="reach-bc", choices=["reach-bc", "reach-ppo"])
    ap.add_argument("--policy", default="hsikan", choices=["hsikan", "mlp"])
    ap.add_argument("--demos", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.task == "reach-ppo":
        result = run_ppo(a.policy, hidden=a.hidden, seed=a.seed,
                         cfg=PPOConfig(n_iters=a.iters))
    else:
        result = run_bc(a.policy, n_demos=a.demos, n_epochs=a.epochs,
                        hidden=a.hidden, seed=a.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
