"""S1 implementation-diff test — does the SB3-style critic (early obs+action concat) fix the reach-then-regress?

Our train_sac, reward_norm OFF (the calibration fix already applied), 4 seeds. Two candidate implementation diffs
vs SB3:
  sb3style_critic : replace QCritic (obs->backbone->feat, action concatenated LATE + 1 layer) with an SB3-style
                    early-concat critic (obs+action -> [256,256] -> 1). Hypothesis: late fusion under-discriminates
                    the action, so the actor cannot do the fine control to HOLD contact.
  gradclipoff     : keep our critic, disable gradient clipping (ours max_grad_norm=10; SB3 has none).
Usage: python s1_archtest.py <sb3style_critic|gradclipoff>   (S1_STEPS env sets budget).
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import s1_env  # noqa: E402
from s1_calib import eval_calib  # noqa: E402  (reuse the exact calibration eval)
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac, AlphaMode  # noqa: E402

STEPS = int(os.environ.get("S1_STEPS", "100000"))
EVAL = max(8000, STEPS // 5)
SEEDS = tuple(range(int(os.environ.get('S1_SEEDS', '4'))))


class EarlyConcatCritic(nn.Module):
    """SB3-style Q(s,a): concatenate obs+action at the INPUT, then a full [hidden,hidden] MLP -> 1. Returns (B,)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


def run(config: str, mean, std, rv):
    per = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env = s1_env.make_s1_env(mean, std, rv)
        actor, critics = build_sac(
            "mlp", obs_dim=39, flat_dim=39, action_dim=4, action_scale=1.0, hidden=256
        )
        max_grad = 10.0
        if config == "sb3style_critic":
            critics = [EarlyConcatCritic(39, 4, 256), EarlyConcatCritic(39, 4, 256)]
        elif config == "gradclipoff":
            max_grad = 1e9
        cfg = SACConfig(
            total_steps=STEPS,
            seed=seed,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            init_alpha=1.0,
            alpha_mode=AlphaMode.AUTO,
            target_entropy=-4.0,
            start_steps=1000,
            capacity=200_000,
            reward_norm=False,
            max_grad_norm=max_grad,
            eval_every=EVAL,
            log_every=EVAL,
        )
        log = []
        ee = s1_env.make_s1_env(mean, std, rv)
        train_sac(
            actor,
            critics,
            env,
            cfg,
            eval_fn=lambda _e, a, _l=log, _ee=ee, _c=critics: (
                _l.append(eval_calib(_ee, a, _c, cfg.gamma, False))
                or _l[-1]["contact_success"]
            ),
        )
        cs = [m["contact_success"] for m in log]
        per.append(
            {
                "seed": seed,
                "contact_curve": cs,
                "min_d_curve": [m["min_d_median"] for m in log],
                "stable_contact_final3": round(float(np.mean(cs[-3:])), 3),
                "Q_minus_MC_bias_final": log[-1]["Q_minus_MC_bias"],
            }
        )
        print(
            f"[arch:{config}] seed{seed}: contact {cs} min_d {[m['min_d_median'] for m in log]} "
            f"stable3 {per[-1]['stable_contact_final3']} bias {log[-1]['Q_minus_MC_bias']}",
            flush=True,
        )
    n = sum(1 for p in per if p["stable_contact_final3"] >= 0.5)
    res = {
        "config": config,
        "n_seeds_stable_contact": n,
        "PASS_3of4": bool(n >= 3),
        "per_seed": per,
    }
    (HERE / f"s1_arch_{config}.json").write_text(
        json.dumps(res, indent=2, default=float)
    )
    print(
        f"\n[arch:{config}] SUMMARY stable_contact={n}/4 PASS={res['PASS_3of4']}",
        flush=True,
    )
    return res


if __name__ == "__main__":
    m, s, rv = s1_env.load_setup(str(HERE / "s1_setup.npz"))
    raise SystemExit(0 if run(sys.argv[1], m, s, rv) else 0)
