"""Behaviour-clone the scripted Galambos demonstrator, then refine with PPO — to get past the two-arm grasp
hard-exploration wall (pure PPO never delivers; see reports/2026-06-24-galambos-hyperedge-ab.md).

The demonstrator reliably grips the coin (~10/12 reach the carry phase) but only delivers ~25% (a free-spinning
coin rolls out of a 2-finger clamp dragged perpendicular). BC teaches the policy that structure (approach→grip→
carry); PPO can then refine the carry — RL may find a wiggle/push the scripted form cannot express.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from hymeko_rl.bc import behaviour_clone
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.galambos_demo import GalambosDemonstrator
from hymeko_rl.policy import build_policy
from hymeko_rl.ppo import PPOConfig, train_ppo


def collect_galambos_demos(env: PlanarGraspEnv, n_episodes: int, seed: int, *,
                           only_success: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Roll the scripted demonstrator for ``n_episodes`` and return ``(obs, actions)`` for cloning.

    With ``only_success`` only the trajectories that delivered the coin are kept (clean demos).

    # Postconditions ``obs`` is ``(M, n_vertices, feat)``, ``actions`` ``(M, n_actions)``; ``M >= 1`` (raises if
      no demos collected). # Errors ``RuntimeError`` if nothing was collected (e.g. 0 successes)."""
    demo = GalambosDemonstrator(env)
    obs_all: list[np.ndarray] = []
    act_all: list[np.ndarray] = []
    n_success = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        demo.reset()
        traj_o: list[np.ndarray] = []
        traj_a: list[np.ndarray] = []
        delivered = False
        for _ in range(env.max_steps):
            action = demo.action(env)
            traj_o.append(obs)
            traj_a.append(action)
            obs, _r, term, trunc, info = env.step(action)
            delivered = delivered or bool(info["in_zone"])
            if term or trunc:
                break
        if delivered:
            n_success += 1
        if delivered or not only_success:
            obs_all.extend(traj_o)
            act_all.extend(traj_a)
    if not obs_all:
        raise RuntimeError(f"collected no demos ({n_success}/{n_episodes} delivered); lower only_success or seed")
    return np.asarray(obs_all, dtype=np.float32), np.asarray(act_all, dtype=np.float32)


@torch.no_grad()
def eval_delivery(env: PlanarGraspEnv, ac: Any, n_episodes: int, seed: int) -> float:
    """Greedy delivery rate: fraction of episodes where the coin enters the zone (the success metric)."""
    succ = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        delivered = False
        for _ in range(env.max_steps):
            action = ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy()
            obs, _r, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
            delivered = delivered or bool(info["in_zone"])
            if term or trunc:
                break
        succ += int(delivered)
    return succ / max(1, n_episodes)


def run_galambos_bc(*, kind: str = "hsikan", hidden: int = 64, difficulty: float = 0.3,
                    n_demos: int = 200, bc_epochs: int = 200, algo: str = "ppo", refine: int = 0,
                    seed: int = 0, only_success: bool = True, robot: str | None = None,
                    save: str | None = None, gif: str | None = None) -> dict[str, Any]:
    """Build env + policy, BC the demonstrator, optionally refine (PPO or off-policy DDPG/TD3), report delivery.

    ``algo``: ``"ppo"`` (on-policy ActorCritic + ``train_ppo``) or ``"ddpg"``/``"td3"`` (off-policy
    DeterministicActor + ``train_offpolicy``; both share BC via ``action_mean``). ``refine`` is PPO iterations
    for ppo, or off-policy environment steps for ddpg/td3 (``0`` = BC only). ``robot=None`` uses the
    control-fixed hand-authored arms. # Postconditions returns BC + post-refine delivery rates + the demo size."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = PlanarGraspEnv(robot=robot, max_steps=300, difficulty=difficulty)
    feat = int(env.observation_space.shape[1])  # type: ignore[index]
    flat = env.hg.n_vertices * feat
    critics: list[Any] = []
    if algo == "ppo":
        ac: Any = (build_policy("mlp", obs_dim=flat, action_dim=env.n_actions, hidden=hidden) if kind == "mlp"
                   else build_policy(kind, obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg, hidden=hidden))
    else:                                             # ddpg / td3: off-policy deterministic actor + twin critics
        from gymnasium.spaces import Box

        from hymeko_rl.ddpg import build_offpolicy
        space = env.action_space
        assert isinstance(space, Box)
        scale = float(np.max(np.abs(space.high)))
        kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg}
        ac, critics = build_offpolicy(kind, obs_dim=feat, flat_dim=flat, action_dim=env.n_actions,
                                      action_scale=scale, n_critics=2, hidden=hidden, **kw)
    obs, acts = collect_galambos_demos(env, n_demos, seed, only_success=only_success)
    bc_losses = behaviour_clone(ac, obs, acts, n_epochs=bc_epochs, seed=seed)
    bc_deliv = eval_delivery(PlanarGraspEnv(robot=robot, max_steps=300, difficulty=difficulty), ac, 24, 9000)
    refine_deliv = float("nan")
    if refine > 0:
        if algo == "ppo":
            train_ppo(ac, env, PPOConfig(n_iters=refine, n_steps=2048, seed=seed, ent_coef=0.003))
        else:
            from hymeko_rl.ddpg import OffPolicyConfig, td3_config, train_offpolicy
            # BC warm-start bridge (act with the clone from step 0; warm the critic alone first; small noise) —
            # without it a cold critic destroys the cloned actor. DDPG uses OffPolicyConfig, TD3 the td3 preset.
            warm: dict[str, Any] = dict(warm_start=True, critic_warmup=2000, noise_scale=0.05)
            cfg_op = (td3_config(total_steps=refine, seed=seed, **warm) if algo == "td3"
                      else OffPolicyConfig(total_steps=refine, seed=seed, **warm))
            # train_offpolicy is env-agnostic; its hint names InvertedPendulumEnv but the harness uses it widely.
            train_offpolicy(ac, critics, env, cfg_op)  # type: ignore[arg-type]
        refine_deliv = eval_delivery(PlanarGraspEnv(robot=robot, max_steps=300, difficulty=difficulty),
                                     ac, 24, 9000)
    if gif is not None:                               # §9 animated output: render the trained policy (high-res)
        from pathlib import Path

        from hymeko_rl.campaign_viz import render_actor_gif
        Path(gif).mkdir(parents=True, exist_ok=True)
        try:
            render_actor_gif(PlanarGraspEnv(robot=robot, max_steps=300, difficulty=difficulty), ac,
                             f"{gif}/galambos_{kind}_{algo}", seed=20_000)
        except Exception as exc:   # noqa: BLE001 — viz is best-effort; a GL/camera failure must not fail the run
            print(f"  [gif galambos/{kind}/{algo} skipped: {type(exc).__name__}: {exc}]")
    if save is not None:                              # persist for rendering successful rollouts to GIFs
        from pathlib import Path
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(ac.state_dict(), save)
    return dict(kind=kind, algo=algo, n_params=int(sum(p.numel() for p in ac.parameters())),
                demo_transitions=len(obs), bc_loss=round(bc_losses[-1], 5), bc_delivery=round(bc_deliv, 3),
                refine_delivery=round(refine_deliv, 3) if refine > 0 else None)


def main(argv: list[str] | None = None) -> int:
    import json
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", default="hsikan")
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--n-demos", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "ddpg", "td3"])
    ap.add_argument("--refine", type=int, default=0, help="PPO iters (ppo) or off-policy env steps (ddpg/td3)")
    ap.add_argument("--ppo-iters", type=int, default=None,
                    help="deprecated alias for --algo ppo --refine N (kept so the in-flight overnight loop runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=None, help="save the trained policy here (for GIF rendering)")
    ap.add_argument("--gif", default=None,
                    help="render a high-res GIF of the trained policy into this dir (§9 animated output)")
    a = ap.parse_args(argv)
    algo, refine = (a.algo, a.refine) if a.ppo_iters is None else ("ppo", a.ppo_iters)
    print(json.dumps(run_galambos_bc(kind=a.kind, difficulty=a.difficulty, n_demos=a.n_demos,
                                     bc_epochs=a.bc_epochs, algo=algo, refine=refine, seed=a.seed,
                                     save=a.save, gif=a.gif), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
