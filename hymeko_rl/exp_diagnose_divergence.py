"""Reproduce the ddpg/HSiKAN seed-2 divergence and LOCALISE it with the structural diagnostic.

The delivery campaign's ddpg-hsikan-seed2 cell blew up (10x slow = denormal/NaN arithmetic). This re-runs that
exact cell (BC warm-start → DDPG refine) but diagnoses the actor at every eval; the moment it reads DIVERGED it
saves the actor + the structural health map and aborts (so we don't pay the post-divergence denormal grind). The
output answers "where did HSiKAN break?" — a NAMED structural component — which an MLP cannot tell you.

    python -m hymeko_rl.exp_diagnose_divergence --seed 2 --refine 20000
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.bc import behaviour_clone
from hymeko_rl.ddpg import OffPolicyConfig, build_offpolicy, train_offpolicy
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.galambos_bc import collect_galambos_demos
from hymeko_rl.hsikan_diagnose import diagnose, format_diagnosis, plot_diagnosis


class _Diverged(Exception):
    """Raised from the eval hook the first time the actor reads structurally DIVERGED (abort the grind)."""


def run(*, seed: int = 2, refine: int = 20_000, hidden: int = 64, difficulty: float = 0.3,
        actor_lr: float = 1e-3, critic_lr: float = 1e-3, n_critics: int = 2,
        out_dir: str = "reports/gifs/campaign") -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = PlanarGraspEnv(robot=None, max_steps=300, difficulty=difficulty)
    feat = int(env.observation_space.shape[1])   # type: ignore[index]
    flat = env.hg.n_vertices * feat
    scale = float(np.max(np.abs(env.action_space.high)))   # type: ignore[attr-defined]  # action_space is a Box
    actor, critics = build_offpolicy("hsikan", obs_dim=feat, flat_dim=flat, action_dim=env.n_actions,
                                     action_scale=scale, n_critics=n_critics, hidden=hidden, hg_state=env.hg)
    obs, acts = collect_galambos_demos(env, 200, seed)
    behaviour_clone(actor, obs, acts, n_epochs=200, seed=seed)   # type: ignore[arg-type]

    sample_obs = torch.as_tensor(env.reset(seed=0)[0][None], dtype=torch.float32)
    caught: dict[str, Any] = {}

    @torch.no_grad()
    def diag_hook(_env: Any, act: Any) -> float:
        d = diagnose(act, sample_obs=sample_obs)
        if not d.healthy:
            caught["state"] = copy.deepcopy(act.state_dict())
            caught["diag"] = d
            raise _Diverged
        return 0.0   # the curve value is irrelevant here; we are hunting the divergence

    # eval_every small so the diagnostic hook fires often → we abort within ~1k steps of divergence, not 40 min.
    # actor_lr/critic_lr/n_critics are exposed so a high critic_lr + n_critics=1 (no clipped double-Q) gives a
    # CONTROLLED divergence (Q-overestimation blow-up) the diagnostic can localise — the paper figure.
    cfg = OffPolicyConfig(total_steps=refine, seed=seed, warm_start=True, critic_warmup=2000,
                          noise_scale=0.05, eval_every=500, actor_lr=actor_lr, critic_lr=critic_lr,
                          n_critics=n_critics)
    diverged = False
    try:
        train_offpolicy(actor, critics, env, cfg, eval_fn=diag_hook)   # type: ignore[arg-type]
    except _Diverged:
        diverged = True

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    final = caught.get("diag") or diagnose(actor, sample_obs=sample_obs)   # diagnose the end state if it survived
    print(format_diagnosis(final))
    plot = plot_diagnosis(final, f"{out_dir}/hsikan_divergence_seed{seed}.png")
    if "state" in caught:
        torch.save(caught["state"], f"{out_dir}/diverged_ddpg_hsikan_seed{seed}.pt")
    return dict(seed=seed, diverged=diverged, healthy=final.healthy,
                localized_to=[p.role for p in final.bad],
                forward_error=final.forward_error, plot=str(plot))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--refine", type=int, default=20_000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--actor-lr", type=float, default=1e-3)
    ap.add_argument("--critic-lr", type=float, default=1e-3, help="crank this (e.g. 2.0) to force divergence")
    ap.add_argument("--n-critics", type=int, default=2, help="1 = plain DDPG (no clipped double-Q) — less stable")
    a = ap.parse_args(argv)
    print(json.dumps(run(seed=a.seed, refine=a.refine, hidden=a.hidden, actor_lr=a.actor_lr,
                         critic_lr=a.critic_lr, n_critics=a.n_critics), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
