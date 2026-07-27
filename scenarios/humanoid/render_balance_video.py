"""Render the humanoid balance results as informative videos (SIMULATION).

Two videos, both with live telemetry (V(t), running V_max vs the certificate bound, the
live Lyapunov checklist, a V(t) sparkline) and the HyMeKo pipeline embedded:

  HyMeKo declarative model -> MJCF (freejoint+floor) -> position-servo control -> Lyapunov cert

* ``compare``  — PD-hold easy (CERTIFIED) | PD-hold hard (overshoots, FAILS) | SAC residual
  same hard perturbation (CERTIFIED). Panels 2-3 share the perturbation, so the delta is
  the learned residual.
* ``frontier`` — the SAC residual at ESCALATING perturbation: CERTIFIED (1.6) -> SURVIVES
  with a big sway (2.6) -> FALLS (4.5). The honest robustness frontier, ending in a real
  tip-over. (Measured: residual certifies to ~1.6; beyond ~4.0 it — and PD-hold — fall.)

Usage::  PYTHONPATH=. python -m scenarios.humanoid.render_balance_video
Visualization only — env dynamics untouched.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from hymeko_rl.train.sac import build_sac

from .balance_env import BalanceConfig, HumanoidBalanceEnv
from .lyapunov import evaluate_lyapunov

_OUT = Path("reports/2026-07-27-humanoid-sac-residual")
_W, _H, _STRIDE, _FPS = 460, 430, 5, 25
_TITLE_H, _CODE_H = 46, 196
_INK, _DIM = (235, 235, 235), (150, 150, 150)
_OK, _WARN, _BAD = (60, 200, 110), (225, 170, 60), (235, 90, 90)
_DESCENT_TOL, _CONV_EPS = 5e-3, 0.05

_HYMEKO = [
    ("HyMeKo model  data/robotics/humanoid.hymeko", _OK),
    ("  humanoid : kinematics {", _DIM),
    ("    pelvis : link { mass 2.0; box[.14,.20,.10] }    // +torso +head +2 legs +2 arms", _INK),
    ("    @base   : conti_joint (world -> pelvis)         // env promotes -> <freejoint>+floor", _INK),
    ("    @hip_l @knee_l @ankle_l ... : rev_joint(AXIS_Y) // 13 sagittal joints, unactuated base", _INK),
    ("  }", _DIM),
    ("control  position-servo :  tau = clip( kp*(q0 + a*delta - q) - kv*qdot + qfrc_bias , +/-tau_max )", _OK),
    ("Lyapunov  V = 1/2[ w_h(z-h_ref)^2 + w_xy|xy-supp|^2 + w_v|v|^2 + w_up(1-up)^2 ] ,  h_ref=0.645", _OK),
    ("certificate (reward-independent) :  V>=0  &  descent>=0.9  &  V_max<=0.055  &  V_final<=0.05", _OK),
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


def _greedy(actor, obs) -> np.ndarray:
    with torch.no_grad():
        return actor.action_mean(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()


def rollout_frames(env, act_fn, seed: int) -> tuple[list[np.ndarray], list[dict], dict]:
    """Roll a full episode; return (frames, per-frame telemetry, result summary).

    Preconditions: ``env`` is a built HumanoidBalanceEnv. Postconditions: one frame + one
    telemetry dict per ``_STRIDE`` steps; result = evaluate_lyapunov(V) plus ``fell``.
    """
    import mujoco
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = _W, _H
    renderer = mujoco.Renderer(env.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, 2.7
    obs, _ = env.reset(seed=seed)
    frames, telem, vs, term, done, k = [], [], [], False, False, 0
    while not done:
        obs, _r, term, trunc, info = env.step(act_fn(obs))
        vs.append(info["V"])
        if k % _STRIDE == 0:
            cam.lookat[:] = [float(env.data.xpos[env._pelvis, 0]), 0.0, 0.55]
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render().copy())
            vmax = max(vs)
            frac = sum(1 for a, b in zip(vs, vs[1:]) if b <= a + _DESCENT_TOL) / max(1, len(vs) - 1)
            telem.append({"k": k + 1, "V": info["V"], "vmax": vmax, "vs": list(vs),
                          "descent": frac >= 0.9, "bounded": vmax <= max(vs[0], _CONV_EPS) + _DESCENT_TOL})
        k += 1
        done = term or trunc
    renderer.close()
    return frames, telem, {**evaluate_lyapunov(vs), "fell": bool(term)}


def _status(res: dict) -> tuple[str, tuple[int, int, int]]:
    if res["fell"]:
        return "FELL", _BAD
    if res["passes"]:
        return "CERTIFIED", _OK
    return "SURVIVES", _WARN


def _sparkline(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], vs: list[float]) -> None:
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=(90, 90, 90), fill=(18, 18, 18))
    ymax = max(0.12, max(vs) * 1.1) if vs else 0.12
    yb = y1 - (y1 - y0) * (0.055 / ymax)
    d.line([(x0, yb), (x1, yb)], fill=_BAD, width=1)
    d.text((x0 + 3, yb - 12), "V_max<=0.055", font=_font(11), fill=_BAD)
    if len(vs) > 1:
        pts = [(x0 + (x1 - x0) * i / (len(vs) - 1), y1 - (y1 - y0) * min(v, ymax) / ymax)
               for i, v in enumerate(vs)]
        d.line(pts, fill=(120, 200, 255), width=2)


def _panel(frame: np.ndarray, title: str, t: dict, status: tuple[str, tuple[int, int, int]]) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    stext, col = status
    d.rectangle([0, 0, img.width - 1, 30], fill=col)
    d.text((8, 8), title, font=_font(15), fill=(255, 255, 255))
    d.text((img.width - 96, 8), stext, font=_font(14), fill=(255, 255, 255))
    fp = _font(14, mono=True)
    d.text((8, 38), f"step {t['k']:>3}   V {t['V']:.3f}   V_max {t['vmax']:.3f}", font=fp, fill=_INK)
    d.text((8, 56), f"descent {'OK ' if t['descent'] else 'no '}  bounded {'OK' if t['bounded'] else 'no'}",
           font=fp, fill=(_OK if t["descent"] and t["bounded"] else _BAD))
    _sparkline(d, (8, img.height - 74, 208, img.height - 8), t["vs"])
    for w in range(4):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=col)
    return np.asarray(img)


def _strip(width: int, height: int, rows, *, mono: bool, pad: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (12, 12, 16))
    d = ImageDraw.Draw(img)
    y = pad
    for text, col in rows:
        d.text((pad, y), text, font=_font(14 if mono else 18, mono=mono), fill=col)
        y += 19 if mono else 26
    return np.asarray(img)


def _compose(clips, title_text: str, stem: str) -> dict:
    """clips = list of (frames, telem, res, title); pad short clips (freeze on fall)."""
    n = max(len(f) for f, _t, _r, _ti in clips)
    width = _W * len(clips)
    title = _strip(width, _TITLE_H, [(title_text, _INK)], mono=False, pad=14)
    code = _strip(width, _CODE_H, _HYMEKO, mono=True, pad=12)
    composed = []
    for i in range(n):
        panels = []
        for frames, telem, res, ti in clips:
            j = min(i, len(frames) - 1)                          # freeze on last frame (fallen)
            panels.append(_panel(frames[j], ti, telem[j], _status(res)))
        composed.append(np.concatenate([title, np.concatenate(panels, axis=1), code], axis=0))
    imageio.mimsave(_OUT / f"{stem}.mp4", composed, fps=_FPS)
    imageio.mimsave(_OUT / f"{stem}.gif", composed[::2], fps=_FPS // 2)
    return {ti: _status(res)[0] for _f, _t, res, ti in clips}


def _load_actor(hard):
    obs_dim = int(hard.observation_space.shape[0])
    actor, _ = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                         action_dim=hard.model.nu, action_scale=1.0, hidden=128)
    actor.load_state_dict(torch.load(_OUT / "humanoid_sac_residual_best.pt"))
    actor.eval()
    return actor


def _env(pr: float) -> HumanoidBalanceEnv:
    return HumanoidBalanceEnv(cfg=BalanceConfig(perturb_lo=pr, perturb_hi=pr, max_steps=500), seed=0)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    actor = _load_actor(_env(0.7))
    zero = np.zeros(_env(0.7).model.nu)
    res = _greedy

    compare = [
        (*rollout_frames(_env(0.2), lambda _o: zero, 0), "PD-hold  easy (0.2)"),
        (*rollout_frames(_env(0.7), lambda _o: zero, 0), "PD-hold  hard (0.7)"),
        (*rollout_frames(_env(0.7), lambda o: res(actor, o), 0), "SAC residual  hard (0.7)"),
    ]
    v1 = _compose(compare, "HyMeKo model -> MJCF (freejoint+floor) -> position-servo control "
                  "-> Lyapunov certificate", "humanoid_balance_compare")

    frontier = [
        (*rollout_frames(_env(0.8), lambda o: res(actor, o), 0), "residual  kick 0.8"),
        (*rollout_frames(_env(2.6), lambda o: res(actor, o), 0), "residual  kick 2.6"),
        (*rollout_frames(_env(4.5), lambda o: res(actor, o), 0), "residual  kick 4.5"),
    ]
    v2 = _compose(frontier, "Robustness frontier: SAC residual at escalating kick  "
                  "(certified -> survives -> FALLS)", "humanoid_balance_frontier")
    print("compare:", v1)
    print("frontier:", v2)


if __name__ == "__main__":
    main()
