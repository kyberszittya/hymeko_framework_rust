"""Render the AIBO actually WALKING forward to the goal — real diagonal-trot stepping.

The point is not the push-recovery sprawl (retracted) but genuine forward locomotion: the
``SteeredTrotGait`` walks the 22-DOF quadruped to the waypoint with a real diagonal trot —
measured realistic (joint speeds ~5 rad/s << the 27 rad/s exploit, never airborne, >= 2 feet
on the ground), under the ``JointVelocityGovernor`` motion contract. The feet lift ~2-3 cm in a
diagonal pattern (a low-clearance trot, not a high-step). A **foot-contact gait diagram** (4
rows, stance vs swing over time) shows the stepping; telemetry shows distance, forward
progress, max joint speed, and feet-down. It reaches the goal (dist -> 0.30 m).

Usage::  PYTHONPATH=. python -m scenarios.aibo.render_walk_to_goal
Visualization only -- env dynamics untouched.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

from .locomotion_gait import SteeredTrotGait, heading_error
from .motion_contract import JointVelocityGovernor
from .render_lyapunov_video import _INK, _OK, _add_goal_marker, _font, _strip

_OUT = Path("reports/2026-07-27-aibo-lyapunov")
_W, _H, _STRIDE, _FPS = 640, 460, 6, 25
_TITLE_H, _CODE_H = 46, 150
_BAD = (235, 90, 90)
_GDIST, _REACH, _VMAX = 1.0, 0.30, 8.0
_LEGS = ("fl", "fr", "bl", "br")

_HYMEKO = [
    ("HyMeKo model  data/robotics/quadruped.hymeko  (Aibo ERS-1000, 22 DOF)", _OK),
    ("  leg x4 : hip_abduct(X) -> hip_flex(Y) -> knee(Y)   |   SteeredTrotGait = diagonal PD-trot", _INK),
    ("motion contract  JointVelocityGovernor(v_max=8 rad/s)  — cut accelerating torque over cap, keep braking", _OK),
    ("realism (measured): joint speeds ~5 rad/s (<< the 27 exploit), 0 airborne, >= 2 feet down = a REAL walk", _OK),
]


def _paws(env):
    import mujoco
    return {k: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "paw_" + k) for k in _LEGS}


def _rollout(env):
    import mujoco
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = _W, _H
    r = mujoco.Renderer(env.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 55.0, -22.0, 2.2
    gait, gov, paws = SteeredTrotGait(), JointVelocityGovernor(v_max=_VMAX), _paws(env)
    env.reset(seed=0)
    tx = float(env.data.xpos[env.torso, 0])
    env.goal = np.array([tx + _GDIST, 0.0], np.float32)
    env._prev_dist = env.dist_to_goal()
    z0 = {k: float(env.data.xpos[v, 2]) for k, v in paws.items()}
    x0 = tx
    frames, telem, contacts, mjv, done, k = [], [], [], 0.0, False, 0
    while not done and k < 3200:
        d = float(env.dist_to_goal())
        drive = 0.0 if d <= _REACH else 1.0
        yaw = float(np.clip(1.1 * float(heading_error(env)), -0.6, 0.6))
        a = gov.govern(env, gait.action(env, yaw_cmd=yaw, drive=drive))
        _o, _r, term, trunc, _i = env.step(a)
        mjv = max(mjv, gov.max_joint_speed(env))
        down = tuple(float(env.data.xpos[v, 2]) - z0[kk] < 0.012 for kk, v in paws.items())
        contacts.append(down)
        if k % _STRIDE == 0:
            cam.lookat[:] = [float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1]), 0.12]
            r.update_scene(env.data, cam)
            _add_goal_marker(r.scene, env.goal)
            frames.append(r.render().copy())
            telem.append({"k": k + 1, "d": d, "fwd": float(env.data.xpos[env.torso, 0]) - x0,
                          "jv": mjv, "down": sum(down), "reached": d <= _REACH,
                          "contacts": list(contacts)})
        k += 1
        settled = d <= _REACH and float(np.hypot(*np.asarray(env.data.qvel)[:2])) < 0.08
        done = bool(term or trunc) or settled
    r.close()
    return frames, telem


def _gait_diagram(d: ImageDraw.ImageDraw, box, contacts) -> None:
    """4 rows (fl,fr,bl,br): green = stance (foot down), dark = swing. Diagonal trot -> checker."""
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=(90, 90, 90), fill=(15, 15, 15))
    rows, win = 4, min(len(contacts), x1 - x0 - 34)
    rh = (y1 - y0 - 4) / rows
    recent = contacts[-win:] if win > 0 else []
    for leg in range(rows):
        yy = y0 + 2 + leg * rh
        d.text((x0 + 2, yy + rh / 2 - 6), _LEGS[leg], font=_font(10), fill=_INK)
        for i, c in enumerate(recent):
            col = _OK if c[leg] else (35, 35, 45)
            d.rectangle([x0 + 30 + i, yy, x0 + 31 + i, yy + rh - 1], fill=col)
    d.text((x0 + 2, y1 - 12), "gait: stance=green (foot down)  swing=dark", font=_font(10), fill=(150, 150, 150))


def _panel(frame, t) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    ok = t["jv"] <= _VMAX + 3.0                                   # realistic joint speed (<< the 27 rad/s exploit)
    col = _OK if ok else _BAD
    d.rectangle([0, 0, img.width - 1, 30], fill=col)
    d.text((8, 8), "AIBO walking to the goal (diagonal trot)", font=_font(15), fill=(255, 255, 255))
    d.text((img.width - 118, 8), "REACHED" if t["reached"] else "REALISTIC WALK", font=_font(13), fill=(255, 255, 255))
    fp = _font(14, mono=True)
    d.text((8, 36), f"step {t['k']:>4}  dist {t['d']:.2f}m  forward {t['fwd']:+.2f}m", font=fp, fill=_INK)
    d.text((8, 54), f"max_jointspd {t['jv']:.1f} rad/s (cap {_VMAX:.0f})   feet_down {t['down']}/4", font=fp,
           fill=(_OK if ok else _BAD))
    _gait_diagram(d, (8, img.height - 96, 320, img.height - 8), t["contacts"])
    for w in range(4):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=col)
    return np.asarray(img)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=_GDIST, reach_radius=0.12, max_steps=3200)
    frames, telem = _rollout(env)
    title = _strip(_W, _TITLE_H, [(
        "AIBO walks FORWARD to the goal — real diagonal-trot stepping, under the motion contract", _INK)],
        mono=False, pad=14)
    code = _strip(_W, _CODE_H, _HYMEKO, mono=True, pad=10)
    comp = [np.concatenate([title, _panel(f, t), code], axis=0) for f, t in zip(frames, telem)]
    imageio.mimsave(_OUT / "aibo_walk_to_goal.mp4", comp, fps=_FPS)
    imageio.mimsave(_OUT / "aibo_walk_to_goal.gif", comp[::3], fps=_FPS // 2)
    print("wrote", _OUT / "aibo_walk_to_goal.mp4", comp[0].shape,
          f"| final dist {telem[-1]['d']:.2f}m forward {telem[-1]['fwd']:.2f}m max_jointspd {telem[-1]['jv']:.1f} rad/s")


if __name__ == "__main__":
    main()
