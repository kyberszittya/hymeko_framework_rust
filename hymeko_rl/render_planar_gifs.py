"""Render Galambos planar-grasp policy rollouts to GIFs — one folder per run.

Each "best run" (a trained checkpoint) gets its own folder under ``--out-root`` with one GIF per
seed, so runs can be compared side by side. Reuses :func:`hymeko_rl.evaluate.render_episode_gif`
(the env-agnostic offscreen renderer) + a top-down camera matching the planar table.

    python -m hymeko_rl.render_planar_gifs --checkpoint checkpoints/galambos/ppo_freed.pt \
        --run galambos_freed --seeds 1000 1003 1005 1006 1007
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.evaluate import render_episode_gif, run_episode
from hymeko_rl.policy import ActorCritic, build_policy


def topdown_camera(*, lookat_y: float = 0.12, distance: float = 0.62) -> mujoco.MjvCamera:
    """A near-straight-down camera framing the table (the scene is planar, so we look at the XY
    plane from above). ``lookat_y`` centres on the workspace between the arms and the zone."""
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 90.0
    cam.elevation = -89.0
    cam.distance = distance
    cam.lookat = np.array([0.0, lookat_y, 0.04])
    return cam


def policy_action_fn(ac: ActorCritic):  # type: ignore[no-untyped-def]
    """Deterministic action-mean source for a trained policy."""
    @torch.no_grad()
    def fn(_env: object, obs: np.ndarray) -> np.ndarray:
        a = ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32))
        return np.asarray(a.squeeze(0).numpy(), dtype=np.float32)
    return fn


def demonstrator_action_fn(demo: object):  # type: ignore[no-untyped-def]
    """Action source for the scripted :class:`~hymeko_rl.galambos_demo.GalambosDemonstrator` (reset it per
    episode before rendering)."""
    def fn(env: object, _obs: np.ndarray) -> np.ndarray:
        return np.asarray(demo.action(env), dtype=np.float32)  # type: ignore[attr-defined]
    return fn


def render_demonstrator_successes(out_dir: str | Path, *, n_seeds: int = 30, difficulty: float = 0.3,
                                  robot: str | None = None, max_steps: int = 300, max_gifs: int = 6,
                                  width: int = 500, height: int = 460, fps: int = 20) -> list[Path]:
    """Roll the scripted demonstrator over ``n_seeds``, find the seeds where it DELIVERS the coin, and render
    those (up to ``max_gifs``) to GIFs — the working-grasp artifact. Reuses the env-agnostic offscreen renderer.

    # Postconditions one GIF per successful seed under ``out_dir``; returns the written paths (possibly empty)."""
    from hymeko_rl.galambos_demo import GalambosDemonstrator
    env = PlanarGraspEnv(robot=robot, max_steps=max_steps, difficulty=difficulty)
    demo = GalambosDemonstrator(env)
    successes: list[int] = []
    for s in range(n_seeds):                                    # find delivering seeds first (cheap, no render)
        env.reset(seed=s)
        demo.reset()
        delivered = False
        for _ in range(env.max_steps):
            _o, _r, term, trunc, info = env.step(demo.action(env))
            delivered = delivered or bool(info["in_zone"])
            if term or trunc:
                break
        if delivered:
            successes.append(s)
        if len(successes) >= max_gifs:
            break
    cam = topdown_camera()
    fn = demonstrator_action_fn(demo)
    written: list[Path] = []
    for s in successes:
        demo.reset()                                           # align the demo with the env reset inside render
        path = render_episode_gif(env, fn, Path(out_dir) / f"demo_seed_{s}_goal", seed=s,
                                  width=width, height=height, fps=fps, camera=cam)
        written.append(Path(path))
        print(f"demonstrator seed {s}: delivered -> {path}", flush=True)
    print(f"wrote {len(written)} demonstrator success GIF(s) to {out_dir}/ ({len(successes)} of {n_seeds} delivered)")
    return written


def load_policy(checkpoint: str | Path, env: PlanarGraspEnv, *, hidden: int = 64,
                kind: str = "hsikan", algo: str = "ppo") -> Any:
    """Build the policy sized to ``env`` (``kind`` backbone, ``algo`` architecture) and load the weights.

    ``algo`` in ``{bc, ppo}`` → an ``ActorCritic`` (Gaussian actor); ``{ddpg, td3}`` → a deterministic off-policy
    actor. Both expose ``action_mean`` for greedy rollout."""
    feat = int(env.observation_space.shape[-1])   # type: ignore[index]
    flat = env.hg.n_vertices * feat
    ac: Any
    if algo in ("ddpg", "td3"):
        from gymnasium.spaces import Box

        from hymeko_rl.ddpg import build_offpolicy
        space = env.action_space
        assert isinstance(space, Box)
        scale = float(np.max(np.abs(space.high)))
        kw = {} if kind == "mlp" else {"hg_state": env.hg}
        ac, _critics = build_offpolicy(kind, obs_dim=feat, flat_dim=flat, action_dim=env.n_actions,
                                       action_scale=scale, n_critics=(2 if algo == "td3" else 1),
                                       hidden=hidden, **kw)
    else:
        ac = (build_policy("mlp", obs_dim=flat, action_dim=env.n_actions, hidden=hidden) if kind == "mlp"
              else build_policy(kind, obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg, hidden=hidden))
    ac.load_state_dict(torch.load(Path(checkpoint), map_location="cpu", weights_only=True))
    ac.eval()
    return ac


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demonstrator", action="store_true",
                    help="render the scripted demonstrator's SUCCESSFUL deliveries (no checkpoint needed)")
    ap.add_argument("--checkpoint", help="trained policy to render (omit with --demonstrator)")
    ap.add_argument("--label", help="policy label prefixed to each GIF filename (default: --run)")
    ap.add_argument("--kind", default="hsikan", choices=("hsikan", "mlp", "signedkan"))
    ap.add_argument("--algo", default="ppo", choices=("bc", "ppo", "ddpg", "td3"),
                    help="policy architecture for loading the checkpoint")
    ap.add_argument("--hand-authored", action="store_true",
                    help="use the hand-authored arms (robot=None) — matches galambos_bc-trained policies")
    ap.add_argument("--difficulty", type=float, default=0.3, help="coin spawn difficulty")
    ap.add_argument("--run", default="demonstrator", help="run name = the per-run GIF folder under --out-root")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[1000, 1003, 1005, 1006, 1007],
                    help="diagnostic seeds to render (default: the freed-run goal seeds)")
    ap.add_argument("--out-root", default="reports/gifs")
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=500)
    ap.add_argument("--height", type=int, default=460)
    a = ap.parse_args(argv)

    if a.demonstrator:
        render_demonstrator_successes(Path(a.out_root) / a.run, difficulty=a.difficulty,
                                      max_steps=a.max_steps, width=a.width, height=a.height, fps=a.fps)
        return 0
    if not a.checkpoint:
        ap.error("--checkpoint is required unless --demonstrator is given")

    env = (PlanarGraspEnv(robot=None, max_steps=a.max_steps, difficulty=a.difficulty) if a.hand_authored
           else PlanarGraspEnv(max_steps=a.max_steps, difficulty=a.difficulty))
    ac = load_policy(a.checkpoint, env, kind=a.kind, algo=a.algo)
    action_fn = policy_action_fn(ac)
    cam = topdown_camera()
    out_dir = Path(a.out_root) / a.run
    label = a.label or a.run
    written = []
    for s in a.seeds:
        outcome, ret = run_episode(env, action_fn, seed=s)          # classify (goal/death/timeout)
        path = render_episode_gif(env, action_fn, out_dir / f"{label}_seed_{s}_{outcome}", seed=s,
                                  width=a.width, height=a.height, fps=a.fps, camera=cam)
        written.append(path)
        print(f"seed {s}: {outcome:>7}  return {ret:6.1f}  -> {path}")
    print(f"wrote {len(written)} gif(s) to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
