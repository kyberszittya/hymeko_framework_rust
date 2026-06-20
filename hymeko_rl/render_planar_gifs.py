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


def load_policy(checkpoint: str | Path, env: PlanarGraspEnv, *, hidden: int = 64) -> ActorCritic:
    """Build the HSiKAN policy sized to ``env`` and load the checkpoint weights."""
    feat = int(env.observation_space.shape[-1])   # type: ignore[index]
    ac = build_policy("hsikan", obs_dim=feat, action_dim=env.n_actions, hg_state=env.hg,
                      hidden=hidden)
    ac.load_state_dict(torch.load(Path(checkpoint), map_location="cpu"))
    ac.eval()
    return ac


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run", required=True, help="run name = the per-run GIF folder under --out-root")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[1000, 1003, 1005, 1006, 1007],
                    help="diagnostic seeds to render (default: the freed-run goal seeds)")
    ap.add_argument("--out-root", default="reports/gifs")
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=500)
    ap.add_argument("--height", type=int, default=460)
    a = ap.parse_args(argv)

    env = PlanarGraspEnv(max_steps=a.max_steps)
    ac = load_policy(a.checkpoint, env)
    action_fn = policy_action_fn(ac)
    cam = topdown_camera()
    out_dir = Path(a.out_root) / a.run
    written = []
    for s in a.seeds:
        outcome, ret = run_episode(env, action_fn, seed=s)          # classify (goal/death/timeout)
        path = render_episode_gif(env, action_fn, out_dir / f"seed_{s}_{outcome}", seed=s,
                                  width=a.width, height=a.height, fps=a.fps, camera=cam)
        written.append(path)
        print(f"seed {s}: {outcome:>7}  return {ret:6.1f}  -> {path}")
    print(f"wrote {len(written)} gif(s) to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
