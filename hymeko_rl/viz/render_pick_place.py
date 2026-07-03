"""Run and visualise the FANUC LR Mate-config arm doing a top-down pick-and-place.

The arm + gripper are one HyMeKo source (``arm_gripper_fanuc_import.hymeko`` = ``fanuc_lrmate.hymeko`` with the
gripper attached to ``arm.tool``); the scripted expert reaches the box with a converged DLS pose-IK
(:class:`~hymeko_rl.env.ik.DampedPoseIK`), grasps top-down, and lifts. The FANUC Z-Y-Y-Z-Y-Z joint config +
Z-Y-Z spherical wrist + slim collision links let the tool point straight down WITHOUT the arm folding into
self-collision — the structural wall the earlier anthropomorphic arm hit (see
``reports/2026-06-22-pick-place-phase0.md``).

    python -m hymeko_rl.viz.render_pick_place --seed 2 --out reports/gifs
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

import numpy as np

from hymeko_rl.env.pick_place_env import PickPlaceEnv
from hymeko_rl.eval.evaluate import render_episode_gif

_FANUC_ARM = "data/robotics/arm_gripper_fanuc_import.hymeko"
# a bent, non-singular "ready" posture (the zero/vertical home is a kinematic singularity for IK).
_FANUC_HOME = (0.0, 1.0, 0.8, 0.0, 0.8, 0.0)


def fanuc_pick_env(**overrides: Any) -> PickPlaceEnv:
    """The FANUC-config pick-and-place env tuned so the scripted expert can grasp+lift top-down.

    The object spawns in the arm's collision-free top-down reach annulus (r∈[0.28, 0.40]); the gripper is
    floor-mounted (the spherical wrist removes the need for a pedestal). ``overrides`` patch any field.

    # Postconditions returns a ready-to-``reset`` :class:`PickPlaceEnv` on the FANUC arm+gripper.
    """
    # the arm is mounted on a pedestal at the table height (mount_height == table_top), so the base→box
    # geometry is IDENTICAL to the reliable floor-grasp — just lifted off the ground, which the robot then
    # never touches. The reliable grasp tuning transfers exactly.
    cfg: dict[str, Any] = dict(robot=_FANUC_ARM, mount_height=0.12, table_top=0.12, obj_radius=(0.28, 0.40),
                               target_xy=(0.34, 0.0), arm_home=_FANUC_HOME, box_mass=0.15,
                               lift_thresh=0.035, place_radius=0.075, max_steps=620)
    cfg.update(overrides)
    return PickPlaceEnv(**cfg)


def expert_action_fn() -> Callable[[PickPlaceEnv, np.ndarray], np.ndarray]:
    """An ``action_fn(env, obs) -> action`` driving the scripted reach→grasp→lift→place demonstrator."""
    return lambda env, _obs: env.expert_action


def pick_camera() -> Any:
    """A 3/4 view framing the pedestal-mounted arm, the table, and the lift."""
    import mujoco
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 128.0, -16.0, 1.5
    cam.lookat[:] = [0.22, 0.0, 0.16]
    return cam


def _hud(ctx: dict[str, Any]) -> list[str]:
    status = ("PLACED" if ctx.get("reached") else "CARRYING" if ctx.get("both_contact")
              else "REACHING")
    return ["FANUC LR Mate config — top-down pick",
            f"step {ctx['step']}  lift {float(ctx.get('lifted', 0.0)):.3f} m",
            f"contact {bool(ctx.get('both_contact'))}  {status}"]


def render_pick(out_dir: str = "reports/gifs", *, seed: int = 2, fps: int = 25,
                width: int = 480, height: int = 360) -> Any:
    """Render one scripted-expert pick episode on the FANUC arm to ``<out_dir>/fanuc_pick.gif``."""
    env = fanuc_pick_env()
    return render_episode_gif(env, expert_action_fn(), f"{out_dir}/fanuc_pick", seed=seed,
                              width=width, height=height, fps=fps, camera=pick_camera(), overlay=_hud)


def trained_action_fn(ac: Any) -> Callable[[PickPlaceEnv, np.ndarray], np.ndarray]:
    """An ``action_fn(env, obs)`` driving a TRAINED policy (the deterministic ``action_mean``)."""
    import torch

    def fn(_env: PickPlaceEnv, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return ac.action_mean(torch.as_tensor(obs[None], dtype=torch.float32)).squeeze(0).numpy()
    return fn


def render_trained(checkpoint: str, kind: str = "hsikan", *, out_dir: str = "reports/gifs",
                   label: "str | None" = None, seed: int = 2, fps: int = 25, width: int = 480,
                   height: int = 360, hidden: int = 64) -> Any:
    """Render a TRAINED pick policy's episode to ``<out_dir>/<label>.gif`` (mirrors the demonstrator render,
    swapping the scripted expert for the loaded policy). # Preconditions ``checkpoint`` is a saved state_dict."""
    import torch

    from hymeko_rl.experiments.gripper_pick_bc import build
    env = fanuc_pick_env()
    ac = build(kind, env, hidden)
    ac.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    ac.eval()
    return render_episode_gif(env, trained_action_fn(ac), f"{out_dir}/{label or f'fanuc_pick_{kind}'}",
                              seed=seed, width=width, height=height, fps=fps, camera=pick_camera(), overlay=_hud)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=2, help="episode seed (2 lifts ~25 cm)")
    ap.add_argument("--out", default="reports/gifs")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--checkpoint", default=None, help="trained policy .pt to render (omit = scripted expert)")
    ap.add_argument("--kind", default="hsikan", choices=("hsikan", "mlp", "mixture", "sa_hsikan"))
    ap.add_argument("--label", default=None, help="GIF filename stem (default: fanuc_pick[_kind])")
    a = ap.parse_args(argv)
    gif = (render_trained(a.checkpoint, a.kind, out_dir=a.out, label=a.label, seed=a.seed, fps=a.fps)
           if a.checkpoint else render_pick(a.out, seed=a.seed, fps=a.fps))
    print(f"wrote {gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
