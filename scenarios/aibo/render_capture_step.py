"""Render the AIBO capture-point protective step vs a passive stand (lateral push recovery).

Two front-view rollouts under the SAME lateral push: the passive stand tips over (recovery
V stays high, CERT FAIL) while the capture-point stepper widens the stance toward the LIPM
capture point and recovers to rest (V -> 0, CERT PASS). Live telemetry (uprightness, lateral
speed, V(t), capture-point offset, sparkline) + the AIBO HyMeKo model + capture-point law
embedded.

Usage::  PYTHONPATH=. python -m scenarios.aibo.render_capture_step
Visualization only -- env dynamics untouched.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

from .capture_step import CapturePointWidening, PushRecoveryLyapunov, capture_point_y
from .locomotion_gait import SteeredTrotGait
from .lyapunov import evaluate_lyapunov
from .motion_contract import JointVelocityGovernor
from .render_lyapunov_video import _INK, _OK, _font, _sparkline, _strip

_OUT = Path("reports/2026-07-27-aibo-lyapunov")
_W, _H, _STRIDE, _FPS = 470, 430, 6, 25
_TITLE_H, _CODE_H = 46, 160
_BAD = (235, 90, 90)
_PUSH = 1.0

_HYMEKO = [
    ("HyMeKo model  data/robotics/quadruped.hymeko  (Aibo ERS-1000, 22 DOF)", _OK),
    ("  leg x4 : hip_abduct(AXIS_X) -> hip_flex(Y) -> knee(Y)     // abduction = frontal-plane WIDENING", _INK),
    ("capture point (LIPM)  xi_y = com_y + com_y_vel * sqrt(com_z / g)", _OK),
    ("capture-point WIDENING (sprawl reflex, NOT a step): abduct legs apart toward xi_y", _OK),
    ("push-recovery V = 1/2[ w_up(1-up)^2 + w_v|v|^2 + w_off|com_lat|^2 ]", _OK),
    ("certificate (reward-independent) :  V>=0 & descent>=0.9 & V_final<=0.05", _OK),
]


def _rollout(env, controller):
    import mujoco
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = _W, _H
    r = mujoco.Renderer(env.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 180.0, -14.0, 1.7   # FRONT view (see the lateral fall)
    V = PushRecoveryLyapunov()
    stand = SteeredTrotGait()
    gov = JointVelocityGovernor(v_max=8.0)                        # realistic-motion contract (no 27 rad/s)
    env.reset(seed=0)
    env.data.qvel[1] = _PUSH
    mujoco.mj_forward(env.model, env.data)
    frames, telem, vs, mjv, done, k = [], [], [], 0.0, False, 0
    while not done and k < 400:
        a = controller.action(env) if controller is not None else stand.action(env, yaw_cmd=0.0, drive=0.0)
        a = gov.govern(env, a)                                    # apply the motion contract
        _o, _r, term, trunc, _i = env.step(a)
        vs.append(V(env))
        mjv = max(mjv, gov.max_joint_speed(env))
        if k % _STRIDE == 0:
            cam.lookat[:] = [float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1]), 0.12]
            r.update_scene(env.data, cam)
            frames.append(r.render().copy())
            telem.append({"k": k + 1, "up": float(env._torso_uprightness()),
                          "v": float(np.hypot(*np.asarray(env.data.qvel)[:2])), "jv": mjv,
                          "off": float(capture_point_y(env) - env.data.subtree_com[1][1]),
                          "V": vs[-1], "vs": list(vs),
                          "descent": _descent(vs), "converged": vs[-1] <= 0.05})
        k += 1
        done = term or trunc
    r.close()
    return frames, telem, evaluate_lyapunov(vs)


def _descent(vs) -> bool:
    if len(vs) < 2:
        return True
    return sum(1 for a, b in zip(vs, vs[1:]) if b <= a + 2e-3) / (len(vs) - 1) >= 0.9


def _panel(frame, title, t, ok) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    col = _OK if ok else _BAD
    d.rectangle([0, 0, img.width - 1, 30], fill=col)
    d.text((8, 8), title, font=_font(15), fill=(255, 255, 255))
    d.text((img.width - 92, 8), "CERT PASS" if ok else "CERT FAIL", font=_font(14), fill=(255, 255, 255))
    fp = _font(13, mono=True)
    d.text((8, 36), f"step {t['k']:>3}  up {t['up']:.2f}  v {t['v']:.2f}  max_jointspd {t.get('jv', 0):.1f} rad/s",
           font=fp, fill=_INK)
    d.text((8, 53), f"V {t['V']:.3f}  descent {'OK' if t['descent'] else 'no'}  conv {'OK' if t['converged'] else 'no'}",
           font=fp, fill=(_OK if t["descent"] and t["converged"] else _BAD))
    _sparkline(d, (8, img.height - 70, 200, img.height - 8), t["vs"])
    for w in range(4):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=col)
    return np.asarray(img)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)

    def _env():
        return QuadrupedGoalEnv(base="free", task="goal", goal_distance=0.8, reach_radius=0.12, max_steps=400)
    clips = [
        (*_rollout(_env(), None), "passive stand (falls)"),
        (*_rollout(_env(), CapturePointWidening()), "capture-point WIDENING (sprawl, not a step)"),
    ]
    width = _W * len(clips)
    title = _strip(width, _TITLE_H, [(
        f"AIBO push recovery ({_PUSH} m/s) UNDER MOTION CONTRACT (joints capped ~8 rad/s): "
        "stand vs capture-point WIDENING (a sprawl, not a step)",
        _INK)], mono=False, pad=14)
    code = _strip(width, _CODE_H, _HYMEKO, mono=True, pad=10)
    n = max(len(f) for f, _t, _e, _ti in clips)
    composed = []
    for i in range(n):
        panels = [_panel(f[min(i, len(f) - 1)], ti, t[min(i, len(t) - 1)], ev["passes"])
                  for f, t, ev, ti in clips]
        composed.append(np.concatenate([title, np.concatenate(panels, axis=1), code], axis=0))
    imageio.mimsave(_OUT / "aibo_capture_step.mp4", composed, fps=_FPS)
    imageio.mimsave(_OUT / "aibo_capture_step.gif", composed[::3], fps=_FPS // 2)
    print("wrote", _OUT / "aibo_capture_step.mp4", composed[0].shape,
          "| verdicts:", {ti: ("PASS" if ev["passes"] else "FAIL") for _f, _t, ev, ti in clips})


if __name__ == "__main__":
    main()
