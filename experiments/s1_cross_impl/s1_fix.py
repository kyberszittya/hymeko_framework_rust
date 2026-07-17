"""S1 root-cause isolation @ 4 seeds — find the SMALLEST config change that makes our SAC match SB3 on S1.

Failing baseline (kato14/15 --stable config): init_alpha=0.2, alpha_lr=1e-3, reward_norm=True, AUTO  -> 0/2.
SB3 (init ent_coef=1.0, lr=3e-4, no reward_norm, AUTO)                                             -> 2/2.
Candidates (each a single/combined step toward SB3), 4 seeds, contact-distance (<7.5cm) success:
  init_alpha_1.0        : init_alpha 0.2->1.0 only            (isolate: is low initial exploration the cause?)
  reward_norm_off       : reward_norm True->False only        (isolate: is Q-inflation from reward_norm the cause?)
  sb3_matched           : init_alpha=1.0 + alpha_lr=3e-4 + reward_norm=False  (full auto-alpha match to SB3)
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import s1_env  # noqa: E402
from s1_ours import eval_metrics, STEPS  # noqa: E402  (reuse the exact eval)
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac, AlphaMode  # noqa: E402

SEEDS = (0, 1, 2, 3)
CANDIDATES = {
    "baseline_failing": dict(
        init_alpha=0.2, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-3, reward_norm=True
    ),
    "init_alpha_1.0": dict(
        init_alpha=1.0, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-3, reward_norm=True
    ),
    "reward_norm_off": dict(
        init_alpha=0.2, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-3, reward_norm=False
    ),
    "sb3_matched": dict(
        init_alpha=1.0, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, reward_norm=False
    ),
}


def run(name, kw, mean, std, rv):
    per = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = s1_env.make_s1_env(mean, std, rv)
        actor, critics = build_sac(
            "mlp", obs_dim=39, flat_dim=39, action_dim=4, action_scale=1.0, hidden=256
        )
        cfg = SACConfig(
            total_steps=STEPS,
            seed=seed,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            start_steps=1000,
            capacity=200_000,
            alpha_mode=AlphaMode.AUTO,
            target_entropy=-4.0,
            eval_every=8000,
            log_every=40000,
            **kw,
        )
        log = []
        ee = s1_env.make_s1_env(mean, std, rv)
        train_sac(
            actor,
            critics,
            env,
            cfg,
            eval_fn=lambda _e, a, _l=log, _ee=ee, _c=critics: (
                _l.append(eval_metrics(_ee, a, _c, cfg.gamma))
                or _l[-1]["success_contact_7_5cm"]
            ),
        )
        fc = log[-1]["success_contact_7_5cm"]
        per.append(
            {
                "seed": seed,
                "final_contact": fc,
                "best_min_d": min(m["min_d_best"] for m in log),
                "final_min_d": log[-1]["min_d_median"],
                "q1_final": log[-1]["q1_mean"],
            }
        )
        print(
            f"[fix] {name} seed{seed}: contact_curve {[m['success_contact_7_5cm'] for m in log]} "
            f"min_d {[m['min_d_median'] for m in log]} q1_final {log[-1]['q1_mean']}",
            flush=True,
        )
    n_reach = sum(1 for p in per if p["final_contact"] >= 0.5)
    return {
        "config": name,
        "kw": {k: v for k, v in kw.items()},
        "n_seeds_reach_contact": n_reach,
        "PASS_gate_3of4": bool(n_reach >= 3),
        "median_final_contact": float(np.median([p["final_contact"] for p in per])),
        "per_seed": per,
    }


def main():
    mean, std, rv = s1_env.load_setup(str(HERE / "s1_setup.npz"))
    res = [run(n, kw, mean, std, rv) for n, kw in CANDIDATES.items()]
    (HERE / "s1_fix_result.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n[fix] SUMMARY (>=3/4 seeds reach contact = PASS)")
    for r in res:
        print(
            f"  {r['config']:20s} {r['n_seeds_reach_contact']}/4 reach  PASS={r['PASS_gate_3of4']}  med_contact={r['median_final_contact']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
