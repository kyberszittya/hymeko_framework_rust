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

from hymeko_rl.train.bc import behaviour_clone
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.eval.evaluate import DwellMetric, eval_metric, greedy_action_fn
from hymeko_rl.experiments.galambos_demo import GalambosDemonstrator
from hymeko_rl.agents.policy import build_policy
from hymeko_rl.train.ppo import PPOConfig, train_ppo


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
        if (ep + 1) % 200 == 0:   # never collect blind (§3): flush progress every 200 rollouts
            print(f"  [demos] {ep + 1}/{n_episodes} rolled | {n_success} delivered "
                  f"| {len(obs_all)} samples kept", flush=True)
    if not obs_all:
        raise RuntimeError(f"collected no demos ({n_success}/{n_episodes} delivered); lower only_success or seed")
    return np.asarray(obs_all, dtype=np.float32), np.asarray(act_all, dtype=np.float32)


def eval_delivery(env: PlanarGraspEnv, ac: Any, n_episodes: int, seed: int) -> float:
    """Greedy delivery rate: fraction of episodes where the coin **stays** in the zone — held for the env's
    ``success_steps`` (the sustained-success rule), NOT a momentary touch that rolls straight out (measured
    correction, 2026-07-01). Thin wrapper over the shared :func:`hymeko_rl.eval.evaluate.eval_metric` (§6.1)."""
    dwell = int(getattr(env, "success_steps", 1))
    res = eval_metric(env, greedy_action_fn(ac), DwellMetric("in_zone", dwell), n_episodes=n_episodes, seed0=seed)
    return float(sum(res)) / max(1, n_episodes)


def run_galambos_bc(*, kind: str = "sa_hsikan", hidden: int = 64, difficulty: float = 0.3,
                    n_demos: int = 200, bc_epochs: int = 200, algo: str = "td3", refine: int = 0,
                    seed: int = 0, only_success: bool = True, robot: str | None = None,
                    save: str | None = None, gif: str | None = None, n_envs: int = 8,
                    update_every: int = 1, device: str = "auto",
                    critic_layernorm: bool = False) -> dict[str, Any]:
    """Build env + policy, BC the demonstrator, optionally refine (off-policy TD3/DDPG or, deprecated, PPO),
    report delivery.

    Defaults follow the 2026-07-01 pivot (PPO ditched): ``algo="td3"`` (off-policy — PPO collapsed the BC
    warm-start, BC=0.125→PPO=0.042) and ``kind="sa_hsikan"`` (the Bᴸ-collapse agent). ``algo``: ``"ddpg"``/
    ``"td3"`` (off-policy DeterministicActor + ``train_offpolicy``; BC via ``action_mean``) or ``"ppo"``
    (on-policy, **deprecated** — retained only for reproduction). ``refine`` is off-policy environment steps
    for ddpg/td3, or PPO iterations for ppo (``0`` = BC only). ``robot=None`` uses the control-fixed hand-
    authored arms. # Postconditions returns BC + post-refine delivery rates + the demo size."""
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

        from hymeko_rl.train.ddpg import build_offpolicy
        space = env.action_space
        assert isinstance(space, Box)
        scale = float(np.max(np.abs(space.high)))
        kw: dict[str, Any] = {} if kind == "mlp" else {"hg_state": env.hg}
        ac, critics = build_offpolicy(kind, obs_dim=feat, flat_dim=flat, action_dim=env.n_actions,
                                      action_scale=scale, n_critics=2, hidden=hidden,
                                      critic_layernorm=critic_layernorm, **kw)
    # Live feedback across the whole pipeline (never run blind — the demo/BC/eval preamble is otherwise dark).
    print(f"  [galambos:{kind}/{algo} seed {seed}] collecting {n_demos} demo episodes...", flush=True)
    obs, acts = collect_galambos_demos(env, n_demos, seed, only_success=only_success)
    print(f"  [galambos] {len(obs)} demo transitions; BC-cloning {bc_epochs} epochs...", flush=True)
    bc_losses = behaviour_clone(ac, obs, acts, n_epochs=bc_epochs, seed=seed)
    print(f"  [galambos] BC done (loss={bc_losses[-1]:.4f}); evaluating clone delivery...", flush=True)
    bc_deliv = eval_delivery(PlanarGraspEnv(robot=robot, max_steps=300, difficulty=difficulty), ac, 24, 9000)
    print(f"  [galambos] bc_delivery={bc_deliv:.3f}; refining {refine} steps ({algo})..." if refine > 0
          else f"  [galambos] bc_delivery={bc_deliv:.3f} (no refine)", flush=True)
    refine_deliv = float("nan")
    if refine > 0:
        if algo == "ppo":
            # Vectorized rollout is the measured-faster default (~5-6x, batched forward kills the B=1
            # launch-bound cost; learning preserved — see test_ppo_vec). n_envs=1 keeps the single-env path.
            train_ppo(ac, env, PPOConfig(n_iters=refine, n_steps=2048, seed=seed, ent_coef=0.003),
                      n_envs=n_envs,
                      make_env=(lambda: PlanarGraspEnv(robot=robot, max_steps=300, difficulty=difficulty))
                      if n_envs > 1 else None)
        else:
            from hymeko_rl.train.ddpg import OffPolicyConfig, td3_bc_config, train_offpolicy
            # TD3+BC anti-collapse (2026-07-01 pivot): the BC anchor ||mu(s) - a_demo||^2 over the demos holds
            # the clone through the off-policy refine. The warm-start bridge ALONE did NOT hold on FANUC
            # (see project-fanuc-offpolicy-collapse) — the anchor + offline_data is the fix, so pass BOTH.
            # eval_every > total_steps disables the internal eval (its default eval_balance is cartpole-specific —
            # meaningless on galambos); the real metric is refine_deliv below, and cfg.log_every gives live feedback.
            cfg_op = (td3_bc_config(total_steps=refine, seed=seed, update_every=update_every, eval_every=refine + 1,
                                    device=device)
                      if algo == "td3"
                      else OffPolicyConfig(total_steps=refine, seed=seed, warm_start=True, critic_warmup=2000,
                                           noise_scale=0.05, bc_coef=2.5, update_every=update_every,
                                           eval_every=refine + 1, device=device))
            train_offpolicy(ac, critics, env, cfg_op, offline_data=(obs, acts))   # env-agnostic (env: Any)
        refine_deliv = eval_delivery(PlanarGraspEnv(robot=robot, max_steps=300, difficulty=difficulty),
                                     ac, 24, 9000)
    if gif is not None:                               # §9 animated output: render the trained policy (high-res)
        from pathlib import Path

        from hymeko_rl.viz.campaign_viz import render_actor_gif
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
    ap.add_argument("--kind", default="sa_hsikan")
    ap.add_argument("--difficulty", type=float, default=0.3)
    ap.add_argument("--n-demos", type=int, default=200)
    ap.add_argument("--bc-epochs", type=int, default=200)
    ap.add_argument("--algo", default="td3", choices=["td3", "ddpg", "ppo"],
                    help="off-policy td3/ddpg (default); ppo deprecated (2026-07-01 pivot — it collapses BC)")
    ap.add_argument("--refine", type=int, default=0, help="off-policy env steps (td3/ddpg) or PPO iters (ppo)")
    ap.add_argument("--ppo-iters", type=int, default=None,
                    help="deprecated alias for --algo ppo --refine N (kept so the in-flight overnight loop runs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8,
                    help="vectorized PPO rollout width (measured-faster default; 1 = single-env path)")
    ap.add_argument("--save", default=None, help="save the trained policy here (for GIF rendering)")
    ap.add_argument("--gif", default=None,
                    help="render a high-res GIF of the trained policy into this dir (§9 animated output)")
    a = ap.parse_args(argv)
    algo, refine = (a.algo, a.refine) if a.ppo_iters is None else ("ppo", a.ppo_iters)
    print(json.dumps(run_galambos_bc(kind=a.kind, difficulty=a.difficulty, n_demos=a.n_demos,
                                     bc_epochs=a.bc_epochs, algo=algo, refine=refine, seed=a.seed,
                                     save=a.save, gif=a.gif, n_envs=a.n_envs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
