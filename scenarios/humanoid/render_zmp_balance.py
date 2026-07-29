"""Render the humanoid Vukobratović-ZMP balance certificate — certified vs pushed into the 1-step region.

The SAME ZMP core as the AIBO turning certificate, on the humanoid: a small push keeps the ZMP inside the
foot support (green, CERTIFIED, 0-step capturable — balances in place); a big push drives the ZMP out
(red) into the 1-step-capturable region (MUST step) — the honest stepping frontier. Overlays the live ZMP
point + the capture point on the ground. The correct, validated result.

Usage::  PYTHONPATH=. python -m scenarios.humanoid.render_zmp_balance
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from hymeko_control.stability import capture_point, capturability_level, vukobratovic_zmp
from scenarios.humanoid.balance_env import BalanceConfig, HumanoidBalanceEnv
from scenarios.humanoid.render_balance_video import _font, _strip
from scenarios.humanoid.zmp_stability import _FOOT_HALF, foot_bodies, foot_support_margin

_OUT = Path("reports/2026-07-29-humanoid-zmp-multiembodiment")
_W, _H, _STRIDE, _FPS = 440, 440, 5, 25
_OK, _BAD, _INK = (60, 200, 110), (235, 90, 90), (235, 235, 235)
_MAX_STEP = 0.30


def _marker(scene, xy: np.ndarray, rgba) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0.03, 0.03], np.float64),
                        np.array([xy[0], xy[1], 0.015], np.float64), np.eye(3).flatten(), np.array(rgba, np.float32))
    scene.ngeom += 1


def _rollout(push: float, steps: int = 280):
    e = HumanoidBalanceEnv(BalanceConfig(push_lat_lo=push, push_lat_hi=push, max_steps=steps))
    e.model.vis.global_.offwidth, e.model.vis.global_.offheight = _W, _H
    r = mujoco.Renderer(e.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -10.0, 2.6
    feet = foot_bodies(e.model)
    e.reset(seed=0)
    dt = float(e.model.opt.timestep) * 10
    pv = np.asarray(e.data.subtree_linvel[0]).copy()
    ph = np.asarray(e.data.subtree_angmom[0]).copy()
    frames, worst_margin, worst_level = [], 1.0, 0
    for k in range(steps):
        e.step(np.zeros(e.model.nu, np.float32))
        zmp, pv, ph = vukobratovic_zmp(e.model, e.data, pv, ph, dt)
        m = foot_support_margin(e.data, zmp, feet)
        cp = capture_point(e.model, e.data)
        lvl, _m0, _m1 = capturability_level(e.data, cp, feet, _FOOT_HALF, _MAX_STEP)
        worst_margin = min(worst_margin, m)
        worst_level = max(worst_level, lvl)
        if k % _STRIDE == 0:
            cam.lookat[:] = [float(e.data.subtree_com[0][0]), float(e.data.subtree_com[0][1]), 0.45]
            r.update_scene(e.data, cam)
            _marker(r.scene, zmp, (0.24, 0.85, 0.45, 0.95) if m > 0 else (0.95, 0.35, 0.35, 0.95))   # ZMP
            _marker(r.scene, cp, (0.35, 0.6, 0.95, 0.9))                                             # capture point (blue)
            frames.append((r.render().copy(), m, lvl))
    r.close()
    return frames, worst_margin, worst_level


def _label(frame, rows) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame.copy())
    d = ImageDraw.Draw(img)
    y = 6
    for text, col in rows:
        d.text((8, y), text, font=_font(15), fill=col)
        y += 21
    return np.asarray(img)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    for push, name in ((0.8, "small push"), (1.8, "big push")):
        frames, mn, lvl = _rollout(push)
        cert = mn > 0.0
        verdict = ("ZMP in support - CERTIFIED", _OK) if cert else ("ZMP out - MUST step (1-step region)", _BAD)
        labelled = []
        for f, m, lv in frames:
            rows = [(f"{name}  (lateral)", _INK), (f"ZMP margin {m:+.2f}", _OK if m > 0 else _BAD),
                    (f"capturability: {['0-step (hold)', '1-step (step!)', 'uncapturable'][lv]}",
                     _OK if lv == 0 else _BAD), verdict]
            labelled.append(_label(f, rows))
        clips.append(labelled)
    n = min(len(c) for c in clips)
    width = _W * len(clips)
    title = _strip(width, 44, [("HyMeKo humanoid - Vukobratovic-ZMP balance certificate (same core as the AIBO turn; blue = capture point)", _INK)],
                   mono=False, pad=12)
    code = _strip(width, 92, [
        ("ZMP = CoM_xy - (m*z*a_xy + [Hdot_y, -Hdot_x]) / (m*(zdd+g))   # embodiment-agnostic (hymeko_control.stability)", _INK),
        ("capturability: xi in support -> 0-step (hold); xi in support + 1 step -> MUST step; else fall   # Pratt/Koolen", (150, 200, 150)),
        ("small push -> ZMP in support (CERTIFIED, 0-step)      big push -> ZMP out, 1-step region (the 0/12 stepping frontier)", (150, 150, 150)),
    ], mono=True, pad=10)
    composed = [np.concatenate([title, np.concatenate([c[k] for c in clips], axis=1), code], axis=0) for k in range(n)]
    imageio.mimsave(_OUT / "humanoid_zmp_balance.mp4", composed, fps=_FPS)
    imageio.mimsave(_OUT / "humanoid_zmp_balance.gif", composed[::3], fps=_FPS // 2)
    print("wrote", _OUT / "humanoid_zmp_balance.mp4", composed[0].shape)


if __name__ == "__main__":
    main()
