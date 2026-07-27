"""Render the spring-legged AIBO launch — rigid leg (grounded) vs series-elastic leg (airborne).

Both legs start from the same loaded crouch (knees compressed to -1.0 rad); the actuator is then
released (zero torque) so the launch is a **pure passive spring**. The rigid geared knee stores no
energy and stays grounded; the series-elastic knee returns its stored PE and carries the body
airborne. The camera is fixed so the vertical launch is visible. Overlay shows the torso rise, the
knee speed at release (a PASSIVE spring, exceeding the 8 rad/s motor cap — legitimate, not a fake
motor), and the airborne flag.

Usage::  PYTHONPATH=. python -m scenarios.aibo.render_spring_leg_video
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
from .spring_leg import LEGS, SpringLegSpec, build_spring_legged

_OUT = Path("reports/2026-07-27-aibo-hop")
_W, _H, _STRIDE, _FPS = 440, 460, 4, 25
_TITLE_H, _CODE_H = 46, 132
_BAD = (235, 90, 90)
_MOTOR_CAP, _LOAD, _STEPS = 8.0, -1.0, 200

_HYMEKO = [
    ("HyMeKo model  data/robotics/quadruped.hymeko  (Aibo ERS-1000, 22 DOF)  — spring variant", _OK),
    ("  SpringLegSpec(stiffness=150, springref=0, damping=0.05)  ->  add_knee_springs(mjcf)  on each knee", _INK),
    ("load: crouch knees to -1.0 rad (store 1/2 K theta^2) ;  release: actuator OFF -> PURE PASSIVE SPRING", _OK),
    ("the fast release (>8 rad/s motor cap) is a SPRING, not a motor — legitimate series elasticity", _OK),
]


def _addr(model):
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    return (bid("torso"),
            {n: int(model.jnt_qposadr[jid(f"knee_{n}")]) for n in LEGS},
            {n: int(model.jnt_dofadr[jid(f"knee_{n}")]) for n in LEGS},
            {n: bid(f"paw_{n}") for n in LEGS},
            int(model.jnt_qposadr[jid("base")]))


def _launch(model):
    """Loaded-crouch → actuator-off release rollout; return (frames, telemetry)."""
    model.vis.global_.offwidth, model.vis.global_.offheight = _W, _H
    r = mujoco.Renderer(model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, 1.9
    d = mujoco.MjData(model)
    torso, knee_q, knee_v, paw, base = _addr(model)
    mujoco.mj_forward(model, d)
    d.qpos[base + 2] = 0.2
    for n in LEGS:
        d.qpos[knee_q[n]] = _LOAD                       # load the springs (compressed crouch)
    mujoco.mj_forward(model, d)
    cam.lookat[:] = [float(d.xpos[torso, 0]), float(d.xpos[torso, 1]), 0.30]  # FIXED — show the rise
    z0 = float(d.xpos[torso, 2])
    paw0 = min(float(d.xpos[paw[n], 2]) for n in LEGS)
    frames, telem, peak = [], [], 0.0
    for k in range(_STEPS):
        mujoco.mj_step(model, d)                        # actuator OFF — pure passive spring
        peak = max(peak, max(abs(float(d.qvel[knee_v[n]])) for n in LEGS))
        clr = min(float(d.xpos[paw[n], 2]) for n in LEGS) - paw0
        if k % _STRIDE == 0:
            r.update_scene(d, cam)
            frames.append(r.render().copy())
            telem.append({"k": k + 1, "rise": float(d.xpos[torso, 2]) - z0,
                          "kv": peak, "air": clr > 0.03})
    r.close()
    return frames, telem


def _panel(frame, t, label: str, is_spring: bool) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    air = t["air"]
    d.rectangle([0, 0, img.width - 1, 30], fill=(60, 120, 60) if is_spring else (120, 70, 50))
    d.text((8, 8), label, font=_font(14), fill=(255, 255, 255))
    tag = ("AIRBORNE" if air else "loading") if is_spring else "GROUNDED"
    d.text((img.width - 92, 8), tag, font=_font(13), fill=(255, 255, 255))
    fp = _font(14, mono=True)
    d.text((8, 36), f"step {t['k']:>4}   torso rise {t['rise']:+.3f} m", font=fp, fill=_INK)
    kv_hot = is_spring and t["kv"] > _MOTOR_CAP
    d.text((8, 54), f"knee speed {t['kv']:5.1f} rad/s"
           + ("  <- PASSIVE SPRING" if kv_hot else f"  (motor cap {_MOTOR_CAP:.0f})"),
           font=fp, fill=(_OK if kv_hot else _INK))
    border = _OK if (is_spring and air) else (_BAD if not is_spring else (150, 150, 150))
    for w in range(4):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=border)
    return np.asarray(img)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    env = QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12, max_steps=200)
    mjcf = env._mjcf
    rigid_f, rigid_t = _launch(mujoco.MjModel.from_xml_string(mjcf))
    spring_f, spring_t = _launch(build_spring_legged(mjcf, SpringLegSpec(150.0, 0.0, 0.05)))
    n = min(len(rigid_f), len(spring_f))
    full_w = 2 * _W
    title = _strip(full_w, _TITLE_H,
                   [("Spring-legged AIBO - series-elastic knees LAUNCH where rigid legs cannot", _INK)],
                   mono=False, pad=14)
    code = _strip(full_w, _CODE_H, _HYMEKO, mono=True, pad=10)
    comp = []
    for i in range(n):
        left = _panel(rigid_f[i], rigid_t[i], "rigid geared leg", is_spring=False)
        right = _panel(spring_f[i], spring_t[i], "spring (series-elastic) leg", is_spring=True)
        body = np.concatenate([left, right], axis=1)
        comp.append(np.concatenate([title, body, code], axis=0))
    imageio.mimsave(_OUT / "aibo_spring_leg.mp4", comp, fps=_FPS)
    imageio.mimsave(_OUT / "aibo_spring_leg.gif", comp[::3], fps=_FPS // 2)
    print("wrote", _OUT / "aibo_spring_leg.mp4", comp[0].shape,
          f"| rigid rise {rigid_t[-1]['rise']:+.3f}m  spring rise {max(t['rise'] for t in spring_t):+.3f}m "
          f"airborne {sum(t['air'] for t in spring_t)}/{len(spring_t)}")


if __name__ == "__main__":
    main()
