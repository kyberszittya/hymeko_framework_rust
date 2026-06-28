"""Side-by-side MLP-vs-HSiKAN comparison GIFs, per task (cart-pole / coin-grasp / quadruped).

Renders each backbone's greedy episode (same seed/camera) and stitches them side-by-side with a HUD naming
the backbone. Reuses hymeko_rl.evaluate.{render_episode_frames, compare_gif}.

    uv run python scripts/compare_backbones_gif.py --task all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

from hymeko_rl.evaluate import compare_gif, render_episode_frames  # noqa: E402
from hymeko_rl.policy import ActorCritic, build_policy  # noqa: E402

OUT = REPO / "reports" / "gifs" / "compare"
OUT.mkdir(parents=True, exist_ok=True)


def greedy(ac: ActorCritic) -> Any:
    @torch.no_grad()
    def act(_e: Any, obs: np.ndarray) -> np.ndarray:
        return ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy()
    return act


def hud(label: str) -> Any:
    def f(ctx: dict[str, Any]) -> list[str]:
        lines = [label, f"step {ctx['step']}", f"return {ctx['return']:.1f}"]
        if "pole_angle" in ctx:
            lines.append(f"pole {float(ctx['pole_angle']) * 180 / np.pi:+.0f} deg")
        if "vx" in ctx:
            lines.append(f"vx {float(ctx['vx']):+.2f}")
        if "height" in ctx:
            lines.append(f"height {float(ctx['height']):+.2f}m")
        if "disk_to_zone" in ctx:
            lines.append(f"disk->zone {float(ctx['disk_to_zone']):.3f}")
        return lines
    return f


def _panels(env_factory: Any, policies: list[tuple[str, ActorCritic]], *, camera: Any,
            seed: int, w: int, h: int) -> list[list[np.ndarray]]:
    import mujoco
    from hymeko_rl.env.arm_world import decorate_scene
    out = []
    for label, ac in policies:
        env = env_factory()
        rm = (mujoco.MjModel.from_xml_string(decorate_scene(env._mjcf))   # sky + floor, not a black void
              if hasattr(env, "_mjcf") else None)
        out.append(render_episode_frames(env, greedy(ac), seed=seed, width=w, height=h,
                                         camera=camera, overlay=hud(label), render_model=rm))
    return out


# ── cart-pole ────────────────────────────────────────────────────────────────
def task_cartpole(seed: int) -> Path:
    from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
    from hymeko_rl.ppo import PPOConfig, train_ppo
    from hymeko_rl.render_inverted_pendulum import load_policy_from_hymeko, side_camera
    mj = emit_cartpole_mjcf()
    hs = load_policy_from_hymeko(REPO / "data" / "nn" / "cartpole_hsikan_policy.hymeko",
                                 InvertedPendulumEnv(mjcf=mj))
    print("  training a matched-MLP cart-pole policy for the comparison…", flush=True)
    torch.manual_seed(0)
    env = InvertedPendulumEnv(mjcf=mj)
    mlp = build_policy("mlp", obs_dim=env.hg.n_vertices * 2, action_dim=1, hidden=112)
    train_ppo(mlp, env, PPOConfig(n_iters=120, n_steps=1024, seed=0), n_envs=16,
              make_env=lambda: InvertedPendulumEnv(mjcf=mj))
    panels = _panels(lambda: InvertedPendulumEnv(mjcf=mj), [("HSiKAN", hs), ("MLP", mlp)],
                     camera=side_camera(), seed=seed, w=420, h=340)
    return compare_gif(panels, OUT / "cartpole_hsikan_vs_mlp", fps=30)


# ── coin-grasp (Galambos) ────────────────────────────────────────────────────
def task_coin(seed: int) -> Path:
    from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
    from hymeko_rl.render_planar_gifs import topdown_camera
    def mk() -> Any:
        return PlanarGraspEnv.from_hymeko(max_steps=120)
    probe = mk()
    feat = int(probe.observation_space.shape[-1])
    hs = build_policy("hsikan", obs_dim=feat, action_dim=probe.n_actions, hg_state=probe.hg, hidden=64)
    hs.load_state_dict(torch.load(REPO / "checkpoints" / "galambos" / "ppo_hsikan.pt", map_location="cpu"))
    mlp = build_policy("mlp", obs_dim=probe.hg.n_vertices * feat, action_dim=probe.n_actions, hidden=96)
    mlp.load_state_dict(torch.load(REPO / "checkpoints" / "galambos" / "ppo_mlp96.pt", map_location="cpu"))
    panels = _panels(mk, [("HSiKAN", hs), ("MLP", mlp)], camera=topdown_camera(), seed=seed, w=420, h=420)
    return compare_gif(panels, OUT / "coingrasp_hsikan_vs_mlp", fps=30)


# ── quadruped JUMP (proper HyMeKo robot, data/robotics/quadruped.hymeko) ──────
def task_quad(seed: int) -> Path:
    import mujoco

    from hymeko_rl.env.quadruped_env import QuadrupedJumpEnv
    from hymeko_rl.ppo import PPOConfig, train_ppo
    def mk() -> Any:
        return QuadrupedJumpEnv(base="free", max_steps=200)
    probe = mk()
    pols = []
    for kind, hid in (("hsikan", 64), ("mlp", 96)):
        print(f"  training jumping quadruped {kind}…", flush=True)
        torch.manual_seed(0)
        od = 2 if kind == "hsikan" else probe.hg.n_vertices * 2
        kw = {"hg_state": probe.hg} if kind == "hsikan" else {}
        ac = build_policy(kind, obs_dim=od, action_dim=probe.n_actions, hidden=hid,
                          log_std_init=0.0, **kw)
        nparams = sum(p.numel() for p in ac.parameters())
        train_ppo(ac, mk(), PPOConfig(n_iters=80, n_steps=2048, seed=0), n_envs=8, make_env=mk)
        torch.save(ac.state_dict(), REPO / "checkpoints" / f"quadruped_jump_{kind}.pt")
        print(f"    {kind}: {nparams} params", flush=True)
        pols.append(("HSiKAN" if kind == "hsikan" else "MLP", ac))
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, 2.8
    cam.lookat[:] = [0.0, 0.0, 0.5]
    panels = _panels(mk, pols, camera=cam, seed=seed, w=420, h=420)
    return compare_gif(panels, OUT / "quadruped_jump_hsikan_vs_mlp", fps=30)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="all", choices=["cartpole", "coin", "quad", "all"])
    ap.add_argument("--seed", type=int, default=20000)
    a = ap.parse_args()
    tasks = {"cartpole": task_cartpole, "coin": task_coin, "quad": task_quad}
    todo = list(tasks) if a.task == "all" else [a.task]
    for t in todo:
        print(f"[{t}] rendering MLP-vs-HSiKAN comparison…", flush=True)
        print(f"[{t}] wrote {tasks[t](a.seed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
