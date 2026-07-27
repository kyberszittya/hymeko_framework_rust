"""Render the spring-legged AIBO hopping FORWARD to a designated goal, staying upright.

A green goal disk (the reach zone) is placed ahead; the AIBO reaches it with repeated spring-powered
hops. Overlay shows the hop count, remaining distance, uprightness, and the phase (LOAD / LAUNCH /
SETTLE). The vertical lift is the passive spring; the forward drive is a motor-limited hip push.

Usage::  PYTHONPATH=. python -m scenarios.aibo.render_spring_hop_goal_video
Visualization only.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

from .render_lyapunov_video import _INK, _OK, _font, _strip
from .spring_hop_gait import SpringHopGait
from .spring_leg import SpringLegSpec, build_spring_legged

_OUT = Path("reports/2026-07-27-aibo-hop")
_W, _H, _FPS = 720, 480, 25
_TITLE_H, _CODE_H = 46, 132
_BAD = (235, 90, 90)
_GOAL_D, _REACH = 0.8, 0.12

_HYMEKO = [
    ("HyMeKo model  data/robotics/quadruped.hymeko  (Aibo ERS-1000, 22 DOF)  — spring variant", _OK),
    ("goal: reach x = start + 0.8 m  (green reach disk, radius 0.12 m) by repeated forward hops", _INK),
    ("each hop: LOAD crouch springs -> LAUNCH passive-spring lift + motor-limited hip push (<=5 N.m)", _OK),
    ("          -> SETTLE PD-hold standing posture (all torque <= 8 N.m) ; stays upright every hop", _OK),
]


def _goal_marker(scene, goal_x: float) -> None:
    ident = np.eye(3, dtype=np.float64).flatten()
    for size, height, rgba in (((_REACH, _REACH, 0.006), 0.006, (0.2, 0.85, 0.35, 0.35)),
                               ((0.012, 0.012, 0.22), 0.22, (0.2, 0.9, 0.4, 0.95))):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array(size, np.float64), np.array([goal_x, 0.0, height], np.float64),
                            ident, np.array(rgba, np.float32))
        scene.ngeom += 1


def _rollout(model):
    model.vis.global_.offwidth, model.vis.global_.offheight = _W, _H
    r = mujoco.Renderer(model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -12.0, 2.3
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    gait = SpringHopGait(model, goal_distance=_GOAL_D, reach_radius=_REACH)
    st = {"x_start": 0.0, "goal_x": None, "n": 0}
    frames, telem = [], []

    def hook(d, phase, hop):
        if phase == "stand":
            st["x_start"] = float(d.xpos[torso, 0])
            if st["n"] % 40:                     # only a few stand frames at the start
                st["n"] += 1
                return
        if st["goal_x"] is None and phase != "stand":
            st["goal_x"] = st["x_start"] + _GOAL_D
        stride = 3 if phase == "launch" else (18 if phase == "settle" else 1)
        st["n"] += 1
        if st["n"] % stride:
            return
        gx = st["goal_x"] if st["goal_x"] is not None else st["x_start"] + _GOAL_D
        cam.lookat[:] = [float(d.xpos[torso, 0]) + 0.25, 0.0, 0.18]
        r.update_scene(d, cam)
        _goal_marker(r.scene, gx)
        frames.append(r.render().copy())
        telem.append({"hop": hop, "rem": gx - float(d.xpos[torso, 0]),
                      "up": float(d.xmat[torso].reshape(3, 3)[2, 2]),
                      "z": float(d.xpos[torso, 2]), "phase": phase})

    result = gait.run(on_step=hook)
    r.close()
    return frames, telem, result


def _panel(frame, t, result) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    reached = t["rem"] <= _REACH
    col = _OK if reached else (60, 90, 150)
    d.rectangle([0, 0, img.width - 1, 30], fill=col)
    d.text((8, 8), "AIBO hops to the goal (spring-legged)", font=_font(15), fill=(255, 255, 255))
    d.text((img.width - 108, 8), "REACHED" if reached else "HOPPING", font=_font(13),
           fill=(255, 255, 255))
    fp = _font(14, mono=True)
    up_ok = t["up"] > 0.8
    d.text((8, 36), f"hop {t['hop']:>2}   remaining {t['rem']:+.2f} m   phase {t['phase'].upper()}",
           font=fp, fill=_INK)
    d.text((8, 54), f"upright {t['up']:.2f} {'(stable)' if up_ok else '(TIP)'}   torso_z {t['z']:.2f} m",
           font=fp, fill=(_OK if up_ok else _BAD))
    border = _OK if reached else (_BAD if not up_ok else (150, 150, 150))
    for w in range(4):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=border)
    return np.asarray(img)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12,
                           max_steps=200)
    model = build_spring_legged(env._mjcf, SpringLegSpec(100.0, 0.0, 0.05))
    frames, telem, result = _rollout(model)
    title = _strip(_W, _TITLE_H,
                   [("Spring-legged AIBO reaches a GOAL by hopping - upright, motor-limited drive", _INK)],
                   mono=False, pad=14)
    code = _strip(_W, _CODE_H, _HYMEKO, mono=True, pad=10)
    comp = [np.concatenate([title, _panel(f, t, result), code], axis=0) for f, t in zip(frames, telem)]
    imageio.mimsave(_OUT / "aibo_spring_hop_goal.mp4", comp, fps=_FPS)
    imageio.mimsave(_OUT / "aibo_spring_hop_goal.gif", comp[::4], fps=max(_FPS // 3, 6))
    print("wrote", _OUT / "aibo_spring_hop_goal.mp4", comp[0].shape,
          f"| reached={result.reached} hops={result.n_hops} forward={result.forward:.2f}m "
          f"min_upright={result.min_upright:.2f} frames={len(comp)}")


if __name__ == "__main__":
    main()
