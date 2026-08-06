"""Render the Vukobratovic-ZMP turning-stability certificate — stable (PASS) vs fast (FAIL) side by side.

Overlays the live ZMP point on the ground (green when inside the support polygon → certified, red when it
leaves → the fall predicted before it happens). The stable rotational-couple turn keeps the ZMP in support
(certificate PASS); the fast turn drives it out (FAIL) and the robot tips. The correct, honest result.

Usage::  PYTHONPATH=. python -m scenarios.aibo.render_zmp_certificate
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from .locomotion_gait import RotationalTurnGait
from .render_lyapunov_video import _font, _strip
from .residual_trot import ResidualTrotConfig, ResidualTrotEnv
from .turn_stability import support_margin, vukobratovic_zmp, _paw_bodies

_OUT = Path("reports/2026-07-29-aibo-turn-zmp-lyapunov-certificate")
_W, _H, _STRIDE, _FPS = 460, 420, 6, 25
_OK, _BAD, _INK = (60, 200, 110), (235, 90, 90), (235, 235, 235)
_turn = RotationalTurnGait()


def _zmp_marker(scene, zmp_xy: np.ndarray, margin: float) -> None:
    """A small sphere at the ZMP; green inside support, red outside."""
    if scene.ngeom >= scene.maxgeom:
        return
    rgba = (0.24, 0.85, 0.45, 0.95) if margin > 0 else (0.95, 0.35, 0.35, 0.95)
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.028, 0.028, 0.028], np.float64),
                        np.array([zmp_xy[0], zmp_xy[1], 0.015], np.float64), np.eye(3).flatten(),
                        np.array(rgba, np.float32))
    scene.ngeom += 1


def _rollout(g: float, steps: int = 360):
    rt = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat"), seed=0)
    env = rt._env
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = _W, _H
    r = mujoco.Renderer(env.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 55.0, -28.0, 1.9
    paws = _paw_bodies(env.model)
    env.reset(seed=0)
    dt = float(env.model.opt.timestep) * int(env.frame_skip)
    prev_v = np.asarray(env.data.subtree_linvel[0]).copy()
    prev_h = np.asarray(env.data.subtree_angmom[0]).copy()
    frames, margins, min_margin, tipped = [], [], 1.0, False
    for k in range(steps):
        env.step(rt._gov.govern(env, _turn.action(env, turn=g)))
        zmp, prev_v, prev_h = vukobratovic_zmp(env, prev_v, prev_h, dt)
        m = support_margin(env, zmp, paws)
        min_margin = min(min_margin, m)
        tipped = tipped or float(env.data.xmat[env.torso].reshape(3, 3)[2, 2]) < 0.5
        if k % _STRIDE == 0:
            cam.lookat[:] = [float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1]), 0.10]
            r.update_scene(env.data, cam)
            _zmp_marker(r.scene, zmp, m)
            frames.append(r.render().copy())
            margins.append(m)
    r.close()
    return frames, margins, min_margin, tipped


def _label(frame, rows) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame.copy())
    d = ImageDraw.Draw(img)
    y = 6
    for text, col in rows:
        d.text((8, y), text, font=_font(16), fill=col)
        y += 22
    return np.asarray(img)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    for g, name in ((1.0, "STABLE turn  g=1.0"), (1.6, "FAST turn  g=1.6")):
        frames, margins, mn, tipped = _rollout(g)
        cert = mn > 0.0
        verdict = ("ZMP in support - CERTIFIED", _OK) if cert else ("ZMP left support - CERT FAIL", _BAD)
        labelled = []
        for f, m in zip(frames, margins):
            rows = [(name, _INK), (f"ZMP margin {m:+.2f}", _OK if m > 0 else _BAD), verdict]
            if tipped and not cert:
                rows.append(("-> tips", _BAD))
            labelled.append(_label(f, rows))
        clips.append(labelled)
    n = min(len(c) for c in clips)
    width = _W * len(clips)
    title = _strip(width, 44, [("HyMeKo Aibo - Vukobratovic-ZMP turning-stability certificate (green = ZMP in support, red = out)", _INK)],
                   mono=False, pad=12)
    code = _strip(width, 96, [
        ("ZMP = CoM_xy - (m*z*a_xy + [Hdot_y, -Hdot_x]) / (m*(zdd+g))   # Vukobratovic zero-moment point (+ angular-momentum rate Hdot)", _INK),
        ("certificate: SAFETY  <=>  ZMP in support polygon for all t   # the Lyapunov / capturability boundary", (150, 200, 150)),
        ("stable g=1.0 -> PASS (<=61 deg/1000)     fast g=1.6 -> FAIL, tips   # boundary predicts the fall before it happens", (150, 150, 150)),
    ], mono=True, pad=10)
    composed = [np.concatenate([title, np.concatenate([c[k] for c in clips], axis=1), code], axis=0)
                for k in range(n)]
    imageio.mimsave(_OUT / "aibo_zmp_certificate.mp4", composed, fps=_FPS)
    imageio.mimsave(_OUT / "aibo_zmp_certificate.gif", composed[::3], fps=_FPS // 2)
    print("wrote", _OUT / "aibo_zmp_certificate.mp4", composed[0].shape)


if __name__ == "__main__":
    main()
