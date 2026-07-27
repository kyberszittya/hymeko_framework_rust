"""Render the certified-scaffold vs residual-SAC balance result as an informative video.

Three side-by-side SIMULATION rollouts of the position-servo humanoid, each with live
telemetry (V(t), running V_max vs the certificate bound, the live Lyapunov checklist, a
V(t) sparkline), framed by the HyMeKo pipeline:

  HyMeKo declarative model  ->  MJCF (freejoint + floor)  ->  position-servo control  ->  Lyapunov certificate

  1. PD-hold-q0 (a=0), EASY perturbation      -> CERTIFIED   (the certified scaffold)
  2. PD-hold-q0 (a=0), HARD perturbation       -> overshoots  (survives, FAILS certificate)
  3. SAC residual,      SAME HARD perturbation  -> CERTIFIED   (residual extends the envelope)

The top strip names the HyMeKo->MJCF->control->certificate pipeline; the bottom strip
embeds the actual HyMeKo model excerpt + the control law + the certificate definition.

Usage::  PYTHONPATH=. python -m scenarios.humanoid.render_balance_video
SIMULATION visualization only — env dynamics untouched.
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
_INK, _DIM, _OK, _BAD = (235, 235, 235), (150, 150, 150), (60, 200, 110), (235, 90, 90)
_DESCENT_TOL, _CONV_EPS, _BOUND = 5e-3, 0.05, 0.055        # certificate thresholds (evaluate_lyapunov defaults)

# HyMeKo model excerpt + control/certificate layer — the declarative description, embedded.
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
    except TypeError:                                     # older PIL: fixed-size default
        return ImageFont.load_default()


def _greedy(actor, obs) -> np.ndarray:
    with torch.no_grad():
        return actor.action_mean(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()


def rollout_frames(env, act_fn, seed: int) -> tuple[list[np.ndarray], list[dict], dict]:
    """Roll a full episode; return (frames, per-frame telemetry, evaluate_lyapunov summary).

    Preconditions: ``env`` is a built HumanoidBalanceEnv. Postconditions: one frame + one
    telemetry dict per ``_STRIDE`` steps; the certificate summary is on the full V-series.
    """
    import mujoco
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = _W, _H
    renderer = mujoco.Renderer(env.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, 2.6      # side view (sagittal tipping)
    obs, _ = env.reset(seed=seed)
    frames, telem, vs, done, k = [], [], [], False, 0
    while not done:
        obs, _r, term, trunc, info = env.step(act_fn(obs))
        vs.append(info["V"])
        if k % _STRIDE == 0:
            cam.lookat[:] = [float(env.data.xpos[env._pelvis, 0]), 0.0, 0.6]
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render().copy())
            vmax = max(vs)
            frac = sum(1 for a, b in zip(vs, vs[1:]) if b <= a + _DESCENT_TOL) / max(1, len(vs) - 1)
            telem.append({"k": k + 1, "V": info["V"], "vmax": vmax,
                          "vs": list(vs), "up": info["upright"],
                          "descent": frac >= 0.9, "bounded": vmax <= max(vs[0], _CONV_EPS) + _DESCENT_TOL})
        k += 1
        done = term or trunc
    renderer.close()
    return frames, telem, evaluate_lyapunov(vs)


def _sparkline(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], vs: list[float]) -> None:
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=(90, 90, 90), fill=(18, 18, 18))
    ymax = max(0.12, max(vs) * 1.1) if vs else 0.12
    yb = y1 - (y1 - y0) * (_BOUND / ymax)                          # certificate bound line
    d.line([(x0, yb), (x1, yb)], fill=_BAD, width=1)
    d.text((x0 + 3, yb - 12), "V_max<=0.055", font=_font(11), fill=_BAD)
    if len(vs) > 1:
        pts = [(x0 + (x1 - x0) * i / (len(vs) - 1), y1 - (y1 - y0) * min(v, ymax) / ymax)
               for i, v in enumerate(vs)]
        d.line(pts, fill=(120, 200, 255), width=2)


def _panel(frame: np.ndarray, title: str, t: dict, ok_final: bool) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    col = _OK if ok_final else _BAD
    d.rectangle([0, 0, img.width - 1, 30], fill=col)
    d.text((8, 8), title, font=_font(15), fill=(255, 255, 255))
    d.text((img.width - 92, 8), "CERT PASS" if ok_final else "CERT FAIL", font=_font(14), fill=(255, 255, 255))
    fp = _font(14, mono=True)
    d.text((8, 38), f"step {t['k']:>3}   V {t['V']:.3f}   V_max {t['vmax']:.3f}", font=fp, fill=_INK)
    d.text((8, 56), f"descent {'OK ' if t['descent'] else 'no '}  bounded {'OK' if t['bounded'] else 'no'}",
           font=fp, fill=(_OK if t["descent"] and t["bounded"] else _BAD))
    _sparkline(d, (8, img.height - 74, 208, img.height - 8), t["vs"])
    for w in range(4):
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=col)
    return np.asarray(img)


def _strip(width: int, height: int, rows, *, title: str | None, mono: bool, pad: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), (12, 12, 16))
    d = ImageDraw.Draw(img)
    y = pad
    if title:
        d.text((pad, y), title, font=_font(20), fill=_OK)
        y += 30
    for text, col in rows:
        d.text((pad, y), text, font=_font(14, mono=mono), fill=col)
        y += 19
    return np.asarray(img)


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    easy = HumanoidBalanceEnv(cfg=BalanceConfig(perturb_lo=0.2, perturb_hi=0.2), seed=0)
    hard = HumanoidBalanceEnv(cfg=BalanceConfig(perturb_lo=0.7, perturb_hi=0.7), seed=0)
    obs_dim = int(hard.observation_space.shape[0])
    actor, _ = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                         action_dim=hard.model.nu, action_scale=1.0, hidden=128)
    actor.load_state_dict(torch.load(_OUT / "humanoid_sac_residual_best.pt"))
    actor.eval()
    zero = np.zeros(hard.model.nu)
    clips = [
        (*rollout_frames(easy, lambda _o: zero, 0), "PD-hold  easy (0.2)"),
        (*rollout_frames(hard, lambda _o: zero, 0), "PD-hold  hard (0.7)"),
        (*rollout_frames(hard, lambda o: _greedy(actor, o), 0), "SAC residual  hard (0.7)"),
    ]
    width = _W * len(clips)
    title = _strip(width, _TITLE_H, [(
        "HyMeKo model  ->  MJCF (freejoint+floor)  ->  position-servo control  ->  Lyapunov certificate",
        _INK)], title=None, mono=False, pad=14)
    code = _strip(width, _CODE_H, _HYMEKO, title=None, mono=True, pad=12)
    n = min(len(f) for f, _t, _e, _ti in clips)
    composed = []
    for i in range(n):
        panels = [_panel(f[i], ti, t[i], ev["passes"]) for f, t, ev, ti in clips]
        composed.append(np.concatenate([title, np.concatenate(panels, axis=1), code], axis=0))
    imageio.mimsave(_OUT / "humanoid_balance_compare.mp4", composed, fps=_FPS)
    imageio.mimsave(_OUT / "humanoid_balance_compare.gif", composed[::2], fps=_FPS // 2)
    verdicts = {ti: ("PASS" if ev["passes"] else "FAIL") for _f, _t, ev, ti in clips}
    print("wrote", _OUT / "humanoid_balance_compare.mp4", composed[0].shape, "| verdicts:", verdicts)


if __name__ == "__main__":
    main()
