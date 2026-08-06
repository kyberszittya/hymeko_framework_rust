"""Behaviour cloning for the FANUC-arm pick-and-place (Phase 1, path A — ported from the floating gripper).

Collects demos from the scripted top-down expert on the FANUC LR Mate-config arm (keeping only successful
episodes), clones the policy actor mean (reusing :func:`hymeko_rl.train.bc.behaviour_clone` and the shared
``gripper_pick_bc`` helpers), and reports the learned policy's lift/place success vs the untrained floor.
HSiKAN-on-the-arm+gripper-hypergraph or the params-matched MLP baseline — the same backbone-swap ablation.

    python -m hymeko_rl.experiments.pick_place_bc --kind hsikan
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.experiments.gripper_pick_bc import build, collect_demos, eval_success
from hymeko_rl.viz.render_pick_place import fanuc_pick_env


def run_pick_place_bc(kind: str = "hsikan", *, n_demos: int = 18, n_epochs: int = 80,
                      hidden: int = 64, seed: int = 0, n_eval: int = 8, algo: str = "bc",
                      refine: int = 0, save: str | None = None) -> dict[str, Any]:
    """Build the FANUC pick env + policy, clone the IK expert (successful demos only), optionally refine with an
    off-policy learner, and report place success vs the untrained floor.

    ``algo``: ``"bc"`` (BC only, ActorCritic), or ``"ddpg"``/``"td3"`` (off-policy DeterministicActor refine from
    the BC warm-start — DDPG = 1 critic, TD3 = twin critics + delayed actor + target smoothing). ``refine`` is the
    off-policy env-step budget. # Postconditions returns place rates: untrained → BC → post-refine."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = fanuc_pick_env()
    space_shape = env.observation_space.shape
    assert space_shape is not None
    nv, feat = int(space_shape[0]), int(space_shape[1])
    critics: list[Any] = []
    if algo in ("ddpg", "td3"):
        from gymnasium.spaces import Box

        from hymeko_rl.train.ddpg import build_offpolicy
        act_space = env.action_space
        assert isinstance(act_space, Box)
        scale = float(np.max(np.abs(act_space.high)))
        kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg}
        ac: Any
        ac, critics = build_offpolicy(kind, obs_dim=feat, flat_dim=nv * feat, action_dim=env.n_actions,
                                      action_scale=scale, n_critics=(2 if algo == "td3" else 1),
                                      hidden=hidden, **kw)
    else:
        ac = build(kind, env, hidden)
    _floor_lift, floor_place = eval_success(fanuc_pick_env(), ac, n_eval, seed=10_000)
    obs, acts = collect_demos(env, n_demos, seed, only_success=True)
    behaviour_clone(ac, obs, acts, n_epochs=n_epochs, seed=seed)
    _bc_lift, bc_place = eval_success(fanuc_pick_env(), ac, n_eval, seed=20_000)
    refine_place = float("nan")
    if refine > 0 and algo in ("ddpg", "td3"):
        from hymeko_rl.train.ddpg import OffPolicyConfig, td3_config, train_offpolicy
        # BC warm-start bridge: act with the cloned actor from step 0, warm the critic alone first, and use a
        # small exploration noise (0.1·π ≈ 0.31 rad would shatter a precise grasp). Without this the cold critic
        # destroys the clone (the "off-policy collapse").
        warm: dict[str, Any] = dict(warm_start=True, critic_warmup=2000, noise_scale=0.05)
        cfg = td3_config(total_steps=refine, seed=seed, **warm) if algo == "td3" \
            else OffPolicyConfig(total_steps=refine, seed=seed, **warm)
        train_offpolicy(ac, critics, env, cfg)  # type: ignore[arg-type]  # env-agnostic; hint is narrow
        _rl, refine_place = eval_success(fanuc_pick_env(), ac, n_eval, seed=20_000)
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ac.state_dict(), save)
    return dict(policy=kind, algo=algo, n_demos=n_demos, n_samples=int(len(obs)),
                n_params=int(sum(p.numel() for p in ac.parameters())),
                untrained_place=round(floor_place, 3), bc_place=round(bc_place, 3),
                refine_place=round(refine_place, 3) if refine > 0 else None)


def main(argv: list[str] | None = None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="hsikan", choices=("hsikan", "mlp"))
    ap.add_argument("--algo", default="bc", choices=("bc", "ddpg", "td3"))
    ap.add_argument("--refine", type=int, default=0, help="off-policy env steps (ddpg/td3)")
    ap.add_argument("--n-demos", type=int, default=18)
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="save the trained policy (for GIF rendering)")
    a = ap.parse_args(argv)
    print(json.dumps(run_pick_place_bc(a.kind, n_demos=a.n_demos, n_epochs=a.n_epochs, seed=a.seed,
                                       algo=a.algo, refine=a.refine, save=a.save), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
