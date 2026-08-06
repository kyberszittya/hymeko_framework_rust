"""Render the omni (abduction crab) AIBO reaching an OFF-AXIS goal by side-stepping — the richer-action win.

The learned omni residual drives the AIBO's unused hip-abduction DOF as a phase-locked lateral
crab-walk, so it reaches a goal off to the side by SIDE-STEPPING (forward trot + lateral crab)
instead of turning — bypassing the turn/stability wall. Overlay shows the distance, lateral offset,
uprightness, and a HyMeKo strip on the richer action space.

Usage::  PYTHONPATH=. python -m scenarios.aibo.render_omni_crab_video [--bearing 40] [--checkpoint P]
Visualization only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from hymeko_rl.train.sac import build_sac

from .render_lyapunov_video import _INK, _OK, _font, _strip
from .residual_trot import ResidualTrotConfig, ResidualTrotEnv

_OUT = Path("reports/2026-07-28-aibo-hop")
_W, _H, _FPS, _STRIDE = 720, 500, 25, 6
_TITLE_H, _CODE_H, _BAD = 44, 132, (235, 90, 90)
_CKPT = "reports/2026-07-28-aibo-residual-trot/aibo_residual_trot_omni_best.pt"

_HYMEKO = [
    ("HyMeKo model  data/robotics/quadruped.hymeko  (Aibo ERS-1000, 22 DOF)", _OK),
    ("  richer action space: the trot uses hip_flex(Y)+knee(Y); the omni policy opens hip_ABDUCT(X)", _INK),
    ("  omni residual: 4-D per-leg abduction, PHASE-LOCKED -> a lateral CRAB over the forward trot", _OK),
    ("  -> reaches OFF-AXIS goals by SIDE-STEPPING, not turning (bypasses the turn/stability wall)", _OK),
]


def _goal_marker(scene, goal) -> None:
    ident = np.eye(3, dtype=np.float64).flatten()
    for size, height, rgba in (((0.12, 0.12, 0.006), 0.006, (0.2, 0.85, 0.35, 0.35)),
                               ((0.012, 0.012, 0.22), 0.22, (0.2, 0.9, 0.4, 0.95))):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array(size, np.float64), np.array([goal[0], goal[1], height], np.float64),
                            ident, np.array(rgba, np.float32))
        scene.ngeom += 1


def _greedy(actor):
    def fn(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return fn


def _rollout(bearing_deg: float, checkpoint: str, horizon: int, seed: int):
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", max_steps=horizon), seed=0)
    actor, _ = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=4, action_scale=1.0, hidden=128)
    actor.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    act = _greedy(actor)
    inner = env._env
    inner.model.vis.global_.offwidth, inner.model.vis.global_.offheight = _W, _H
    r = mujoco.Renderer(inner.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 55.0, -32.0, 2.1
    inner.reset(seed=seed)
    tx, ty = float(inner.data.xpos[inner.torso, 0]), float(inner.data.xpos[inner.torso, 1])
    b = bearing_deg * np.pi / 180.0
    goal = np.array([tx + 0.6 * np.cos(b), ty + 0.6 * np.sin(b)], np.float32)
    inner.goal = goal
    inner._prev_dist = inner.dist_to_goal()
    env._prev_dist = float(inner.dist_to_goal())
    frames, telem, lat0, reached_ever = [], [], ty, False
    for k in range(horizon):
        env._apply(act(env._obs()))
        d = float(inner.dist_to_goal())
        up = float(inner.data.xmat[inner.torso].reshape(3, 3)[2, 2])
        reached_ever = reached_ever or d <= 0.12          # entered the reach zone (eval's definition)
        if k % _STRIDE == 0 or reached_ever:              # always capture the reaching frame
            cam.lookat[:] = [float(inner.data.xpos[inner.torso, 0]),
                             float(inner.data.xpos[inner.torso, 1]) * 0.5 + goal[1] * 0.5, 0.15]
            r.update_scene(inner.data, cam)
            _goal_marker(r.scene, goal)
            frames.append(r.render().copy())
            telem.append({"k": k, "dist": d, "up": up,
                          "lat": float(inner.data.xpos[inner.torso, 1]) - lat0,
                          "reached": reached_ever})
        if reached_ever:
            break
    frames.extend([frames[-1]] * 10)
    telem.extend([telem[-1]] * 10)
    r.close()
    return frames, telem, bearing_deg


def _panel(frame, t, bearing) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    reached = t["reached"]
    d.rectangle([0, 0, img.width - 1, 30], fill=(40, 130, 70) if reached else (40, 90, 130))
    d.text((8, 7), f"OMNI crab-walk — reach a {bearing:+.0f} deg OFF-AXIS goal by side-stepping",
           font=_font(14), fill=(255, 255, 255))
    d.text((img.width - 100, 7), "REACHED" if reached else "CRABBING", font=_font(13),
           fill=(255, 255, 255))
    fp = _font(14, mono=True)
    up_ok = t["up"] > 0.5
    d.text((8, 36), f"dist-to-goal {t['dist']:.2f} m   lateral crab {t['lat']:+.2f} m",
           font=fp, fill=_INK)
    d.text((8, 54), f"upright {t['up']:.2f} {'(stable)' if up_ok else '(TIP)'}   (no turning used)",
           font=fp, fill=(_OK if up_ok else _BAD))
    border = _OK if reached else (_BAD if not up_ok else (150, 150, 150))
    for w in range(4):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=border)
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bearing", type=float, default=40.0)
    ap.add_argument("--checkpoint", default=_CKPT)
    ap.add_argument("--horizon", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=503)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    frames, telem, bearing = _rollout(args.bearing, args.checkpoint, args.horizon, args.seed)
    title = _strip(_W, _TITLE_H, [(
        "AIBO reaches an off-axis goal by CRAB-WALKING — the richer (abduction) action space", _INK)],
        mono=False, pad=12)
    code = _strip(_W, _CODE_H, _HYMEKO, mono=True, pad=10)
    comp = [np.concatenate([title, _panel(f, t, bearing), code], axis=0) for f, t in zip(frames, telem)]
    imageio.mimsave(_OUT / "aibo_omni_crab.mp4", comp, fps=_FPS)
    imageio.mimsave(_OUT / "aibo_omni_crab.gif", comp[::3], fps=max(_FPS // 3, 6))
    print("wrote", _OUT / "aibo_omni_crab.mp4", comp[0].shape,
          f"| bearing={bearing} reached={telem[-1]['reached']} lateral={telem[-1]['lat']:.2f}m "
          f"final_dist={telem[-1]['dist']:.2f}m frames={len(comp)}")


if __name__ == "__main__":
    main()
