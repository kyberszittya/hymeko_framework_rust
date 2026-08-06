"""One demo-seed cell: corrected SAC on random-task Coffee-Push, arm in {cold, demo_seed}, one seed.

Corrected stack (reward_norm off + early-concat critic + SB3-matched auto-alpha). demo_seed preloads the replay
buffer with the shared 5000 balanced scripted transitions (true rewards/dones); cold starts empty. Identical
otherwise. Eval 50 episodes every 10k steps over 200k. Writes result_<arm>_seed<seed>.json.
Usage: python exp_demo_seed.py <cold|demo_seed> <seed>
"""
from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("DEMO_SEED_THREADS", "2")))  # cap per-process threads for parallel launch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
warnings.simplefilter("ignore")
import harness  # noqa: E402
from hymeko_rl.train.flat_critic import build_flat_sac  # noqa: E402
from hymeko_rl.train.sac import SACConfig, train_sac, AlphaMode  # noqa: E402

STEPS = 200_000
EVAL = 10_000
EVAL_EP = 50


def run(arm: str, seed: int) -> dict:
    d = np.load(str(HERE / "demo_seed_setup.npz"))
    mean, std = d["mean"], d["std"]
    demos = (d["obs"], d["act"], d["rew"], d["nxt"], d["done"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = harness.make_env(mean, std)
    actor, critics = build_flat_sac(39, 4, 1.0, hidden=256)
    cfg = SACConfig(total_steps=STEPS, seed=seed, batch_size=256, gamma=0.99, tau=0.005, actor_lr=3e-4,
                    critic_lr=3e-4, alpha_lr=3e-4, init_alpha=1.0, alpha_mode=AlphaMode.AUTO, target_entropy=-4.0,
                    start_steps=1000, capacity=200_000, reward_norm=False, eval_every=EVAL, log_every=EVAL)
    init = demos if arm == "demo_seed" else None
    log: list = []
    ee = harness.make_env(mean, std)
    train_sac(actor, critics, env, cfg,
              eval_fn=lambda _e, a: (log.append(harness.eval_episodes(ee, a, critics, cfg.gamma, demos, n=EVAL_EP))
                                     or log[-1]["success_rate"]),
              init_transitions=init)
    sr = [m["success_rate"] for m in log]
    cr = [m["contact_rate"] for m in log]
    ent = [m["entropy_mean"] for m in log]
    final5 = float(np.mean(sr[-5:]))
    best = float(max(sr))
    # collapse events: a checkpoint where success drops >=0.3 from the running best AND entropy jumps (peaks)
    collapse = sum(1 for k in range(1, len(sr)) if max(sr[:k]) - sr[k] >= 0.3)
    res = {
        "arm": arm, "seed": seed, "steps": STEPS, "eval_every": EVAL, "eval_episodes": EVAL_EP,
        "success_curve": sr, "contact_curve": cr, "mug_disp_curve": [m["mug_disp_median"] for m in log],
        "entropy_curve": ent, "q_online_curve": [m["q_online_mean"] for m in log],
        "q_demo_curve": [m["q_demo_mean"] for m in log],
        "first_contact_step": next((EVAL * (k + 1) for k, v in enumerate(cr) if v > 0), None),
        "first_success_step": next((EVAL * (k + 1) for k, v in enumerate(sr) if v > 0), None),
        "best_success": round(best, 3), "final_success": round(sr[-1], 3),
        "stable_success_final5": round(final5, 3), "retention_gap": round(best - final5, 3),
        "collapse_events": int(collapse),
        "demo_success_transitions": 1250 if arm == "demo_seed" else 0,
    }
    (HERE / f"result_{arm}_seed{seed}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[demo-seed] {arm} seed{seed}: best {best:.2f} final5 {final5:.2f} retention {res['retention_gap']:.2f} "
          f"first_success {res['first_success_step']} collapse {collapse} | curve {sr}", flush=True)
    return res


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
