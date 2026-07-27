"""Render the AIBO Lyapunov result as an informative video (SIMULATION).

Two side-by-side rollouts of the 22-DOF Aibo ERS-1000 approaching an OFF-AXIS waypoint,
framed by the HyMeKo pipeline (HyMeKo model -> MJCF -> SteeredTrotGait control -> Lyapunov
certificate) and carrying live telemetry (distance, heading error, speed, V(t), the live
Lyapunov checklist, a V(t) sparkline, and a top-down minimap of the path to the goal):

  1. approach-align-stop pursuit (state-dependent) -> CERTIFIED  (V converges: aligned, reached, stopped)
  2. constant-forward negative control            -> FAILS       (misses the off-axis goal, V stays high)

Both use the same waypoint, so the difference is the control law. The certificate is the
reward-independent AIBO Lyapunov certificate (unchanged).

Usage::  PYTHONPATH=. python -m scenarios.aibo.render_lyapunov_video
Visualization only -- env dynamics untouched.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hymeko_rl.env.quadruped_env import QuadrupedGoalEnv

from .locomotion_gait import SteeredTrotGait, heading_error
from .lyapunov import AIBOLyapunov, evaluate_lyapunov

_OUT = Path("reports/2026-07-27-aibo-lyapunov")
_W, _H, _STRIDE, _FPS = 470, 430, 8, 25
_TITLE_H, _CODE_H = 46, 178
_INK, _DIM = (235, 235, 235), (150, 150, 150)
_OK, _BAD = (60, 200, 110), (235, 90, 90)
_BEARING, _GDIST, _REACH = 28.0, 0.9, 0.42

_HYMEKO = [
    ("HyMeKo model  data/robotics/quadruped.hymeko  (Aibo ERS-1000, 22 DOF)", _OK),
    ("  quadruped : kinematics {", _DIM),
    ("    torso : link { sphere } --waist(AXIS_Z)--> rear_body        // rounded loaf", _INK),
    ("    leg x4 : hip_abduct(X) -> hip_flex(Y) -> knee(Y)            // 3-axis legs = 12 DOF", _INK),
    ("    head 3-axis + neck(Y) + waist(Z) + tail 2-axis + ears + mouth = 22 total", _INK),
    ("  }", _DIM),
    ("control  SteeredTrotGait : bidirectional reduce-inner yaw + diagonal PD-trot", _OK),
    ("pursuit  yaw = clip(1.1*heading_err, +/-0.8) ; drive ~ align x dist ; stop at reach", _OK),
    ("Lyapunov  V = 1/2[ w_d*max(0,d-reach)^2 + w_th*herr^2 + w_v*speed^2 ]", _OK),
    ("certificate (reward-independent) :  V>=0 & descent>=0.9 & V_final<=0.05", _OK),
]


def _font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    if mono:
        for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _speed(env) -> float:
    v = np.asarray(env.data.qvel)[:2]
    return float(np.hypot(v[0], v[1]))


def _set_goal(env, bearing_deg: float) -> None:
    tx, ty = float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1])
    th = np.radians(bearing_deg)
    env.goal = np.array([tx + _GDIST * np.cos(th), ty + _GDIST * np.sin(th)], np.float32)
    env._prev_dist = env.dist_to_goal()


def _add_goal_marker(scene, goal) -> None:
    """Draw the goal as a translucent green reach-zone disk + a thin pole (decorative)."""
    import mujoco
    ident = np.eye(3, dtype=np.float64).flatten()
    for size, height, rgba in (((_REACH, _REACH, 0.006), 0.006, (0.2, 0.85, 0.35, 0.35)),
                               ((0.012, 0.012, 0.18), 0.18, (0.2, 0.9, 0.4, 0.9))):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array(size, np.float64),
                            np.array([goal[0], goal[1], height], np.float64), ident,
                            np.array(rgba, np.float32))
        scene.ngeom += 1


def rollout(env, V, *, pursue: bool, fixed_cam: bool = False):
    """Roll approach-align-stop pursuit (pursue=True) or constant-forward negative control.

    Returns (frames, per-frame telemetry, evaluate_lyapunov summary). One frame per
    ``_STRIDE`` steps; the certificate is evaluated on the full V-series. ``fixed_cam``
    frames the start→goal span (for the reach video) instead of tracking the torso.
    """
    import mujoco
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = _W, _H
    renderer = mujoco.Renderer(env.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -38.0, 2.4
    env.reset(seed=0)
    _set_goal(env, _BEARING)
    if fixed_cam:                                                # frame the whole start->goal traverse
        sx, sy = float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1])
        cam.lookat[:] = [0.5 * (sx + env.goal[0]), 0.5 * (sy + env.goal[1]), 0.1]
        cam.azimuth, cam.elevation, cam.distance = 60.0, -32.0, 2.1
    gait = SteeredTrotGait()
    frames, telem, vs, path, done, k = [], [], [], [], False, 0
    while not done and k < 2400:
        d = float(env.dist_to_goal())
        herr = float(heading_error(env))
        spd = _speed(env)
        if pursue:
            if d <= _REACH:
                yaw, drive = 0.0, 0.0                            # reached -> stop/hold
            else:
                yaw = float(np.clip(1.1 * herr, -0.8, 0.8))
                align = max(0.3, float(np.cos(np.clip(herr, -np.pi, np.pi))))
                drive = align * float(np.clip(d / 0.6, 0.5, 1.0))
        else:
            yaw, drive = 0.0, 1.0                                # constant forward, no align/stop
        _o, _r, term, trunc, _i = env.step(gait.action(env, yaw_cmd=yaw, drive=drive))
        vs.append(V({"dist_to_goal": d, "heading_error": herr, "speed": spd}))
        path.append((float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1])))
        if k % _STRIDE == 0:
            if not fixed_cam:
                cam.lookat[:] = [float(env.data.xpos[env.torso, 0]), float(env.data.xpos[env.torso, 1]), 0.1]
            renderer.update_scene(env.data, cam)
            _add_goal_marker(renderer.scene, env.goal)
            frames.append(renderer.render().copy())
            telem.append({"k": k + 1, "d": d, "herr": np.degrees(herr), "spd": spd,
                          "V": vs[-1], "vmax": max(vs), "vs": list(vs), "path": list(path),
                          "goal": (float(env.goal[0]), float(env.goal[1])),
                          "reached": d <= _REACH,
                          "descent": _descent(vs), "converged": vs[-1] <= 0.05})
        k += 1
        done = term or trunc
    renderer.close()
    return frames, telem, evaluate_lyapunov(vs)


def _descent(vs) -> bool:
    if len(vs) < 2:
        return True
    return sum(1 for a, b in zip(vs, vs[1:]) if b <= a + 2e-3) / (len(vs) - 1) >= 0.9


def _sparkline(d: ImageDraw.ImageDraw, box, vs) -> None:
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=(90, 90, 90), fill=(18, 18, 18))
    ymax = max(0.1, max(vs) * 1.1) if vs else 0.1
    yb = y1 - (y1 - y0) * (0.05 / ymax)
    d.line([(x0, yb), (x1, yb)], fill=_BAD, width=1)
    d.text((x0 + 3, yb - 12), "V_final<=0.05", font=_font(10), fill=_BAD)
    if len(vs) > 1:
        pts = [(x0 + (x1 - x0) * i / (len(vs) - 1), y1 - (y1 - y0) * min(v, ymax) / ymax)
               for i, v in enumerate(vs)]
        d.line(pts, fill=(120, 200, 255), width=2)


def _minimap(d: ImageDraw.ImageDraw, box, path, goal) -> None:
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=(90, 90, 90), fill=(18, 18, 18))
    xs = [p[0] for p in path] + [goal[0]]
    ys = [p[1] for p in path] + [goal[1]]
    lo = np.array([min(xs), min(ys)]) - 0.15
    hi = np.array([max(xs), max(ys)]) + 0.15
    span = np.maximum(hi - lo, 1e-3)

    def px(p):
        return (x0 + (x1 - x0) * (p[0] - lo[0]) / span[0], y1 - (y1 - y0) * (p[1] - lo[1]) / span[1])
    gx, gy = px(goal)
    r = (x1 - x0) * _REACH / span[0]
    d.ellipse([gx - r, gy - r, gx + r, gy + r], outline=_OK)                # reach circle
    d.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], fill=_BAD)                  # goal
    if len(path) > 1:
        d.line([px(p) for p in path], fill=(120, 200, 255), width=2)       # trajectory
    hx, hy = px(path[-1])
    d.ellipse([hx - 3, hy - 3, hx + 3, hy + 3], fill=_INK)                  # current pose
    d.text((x0 + 3, y0 + 2), "path -> goal", font=_font(10), fill=_DIM)


def _panel(frame, title, t, ok_final) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    col = _OK if ok_final else _BAD
    d.rectangle([0, 0, img.width - 1, 30], fill=col)
    d.text((8, 8), title, font=_font(15), fill=(255, 255, 255))
    d.text((img.width - 92, 8), "CERT PASS" if ok_final else "CERT FAIL", font=_font(14), fill=(255, 255, 255))
    fp = _font(13, mono=True)
    d.text((8, 36), f"step {t['k']:>3}  d {t['d']:.2f}m  herr {t['herr']:+5.0f}deg  v {t['spd']:.2f}",
           font=fp, fill=_INK)
    d.text((8, 53), f"V {t['V']:.3f}  V_max {t['vmax']:.3f}  descent {'OK' if t['descent'] else 'no'}"
           f"  conv {'OK' if t['converged'] else 'no'}",
           font=fp, fill=(_OK if t["descent"] and t["converged"] else _BAD))
    if t.get("reached"):                                          # goal reached badge
        d.rectangle([img.width // 2 - 62, 40, img.width // 2 + 62, 66], fill=_OK)
        d.text((img.width // 2 - 52, 46), "GOAL REACHED", font=_font(14), fill=(255, 255, 255))
    _sparkline(d, (8, img.height - 70, 190, img.height - 8), t["vs"])
    _minimap(d, (img.width - 132, img.height - 132, img.width - 8, img.height - 8), t["path"], t["goal"])
    for w in range(4):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=col)
    return np.asarray(img)


def _strip(width, height, rows, *, mono, pad) -> np.ndarray:
    img = Image.new("RGB", (width, height), (12, 12, 16))
    d = ImageDraw.Draw(img)
    y = pad
    for text, col in rows:
        d.text((pad, y), text, font=_font(14 if mono else 18, mono=mono), fill=col)
        y += 17 if mono else 26
    return np.asarray(img)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    V = AIBOLyapunov(reach_target=_REACH)

    def _env():
        return QuadrupedGoalEnv(base="free", task="goal", goal_distance=_GDIST,
                                reach_radius=0.12, max_steps=3000)
    clips = [
        (*rollout(_env(), V, pursue=True), "approach-align-stop  (pursuit)"),
        (*rollout(_env(), V, pursue=False), "constant-forward  (negative)"),
    ]
    width = _W * len(clips)
    title = _strip(width, _TITLE_H, [(
        "HyMeKo model -> MJCF (freejoint+floor) -> SteeredTrotGait control -> Lyapunov certificate",
        _INK)], mono=False, pad=14)
    code = _strip(width, _CODE_H, _HYMEKO, mono=True, pad=10)
    n = max(len(f) for f, _t, _e, _ti in clips)
    composed = []
    for i in range(n):
        panels = []
        for frames, telem, ev, ti in clips:
            j = min(i, len(frames) - 1)
            panels.append(_panel(frames[j], ti, telem[j], ev["passes"]))
        composed.append(np.concatenate([title, np.concatenate(panels, axis=1), code], axis=0))
    imageio.mimsave(_OUT / "aibo_lyapunov_compare.mp4", composed, fps=_FPS)
    imageio.mimsave(_OUT / "aibo_lyapunov_compare.gif", composed[::3], fps=_FPS // 2)
    print("wrote", _OUT / "aibo_lyapunov_compare.mp4", composed[0].shape,
          "| verdicts:", {ti: ("PASS" if ev["passes"] else "FAIL") for _f, _t, ev, ti in clips})

    # single-panel "reaches the goal" video: fixed camera framing the traverse, visible goal marker
    rf, rt, rev = rollout(_env(), V, pursue=True, fixed_cam=True)
    reached_at = next((t["k"] for t in rt if t.get("reached")), None)
    rtitle = _strip(_W, _TITLE_H, [("HyMeKo Aibo ERS-1000 — approach-align-stop to the waypoint", _INK)],
                    mono=False, pad=14)
    rcode = _strip(_W, _CODE_H, _HYMEKO, mono=True, pad=10)
    rframes = [np.concatenate([rtitle, _panel(f, "reaches the goal", t, rev["passes"]), rcode], axis=0)
               for f, t in zip(rf, rt)]
    imageio.mimsave(_OUT / "aibo_reach_goal.mp4", rframes, fps=_FPS)
    imageio.mimsave(_OUT / "aibo_reach_goal.gif", rframes[::3], fps=_FPS // 2)
    print("wrote", _OUT / "aibo_reach_goal.mp4", rframes[0].shape,
          f"| reached at step {reached_at}, final d={rt[-1]['d']:.2f}m, cert={'PASS' if rev['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
