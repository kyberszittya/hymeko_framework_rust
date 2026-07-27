"""Render the certified-scaffold vs residual-SAC balance result as a 3-panel video.

Three side-by-side clips, all SIMULATION rollouts of the position-servo humanoid:

  1. PD-hold-q0 (a=0), EASY perturbation  -> CERTIFIED   (the certified scaffold)
  2. PD-hold-q0 (a=0), HARD perturbation  -> overshoots  (survives, FAILS certificate)
  3. SAC residual,      SAME HARD perturbation -> CERTIFIED (residual extends the envelope)

Panels 2 and 3 share the identical perturbation seed, so the difference is purely the
learned residual. The per-panel banner shows the Lyapunov-certificate verdict (computed,
reward-independent). Writes an MP4 (+GIF) under the residual report dir.

Usage::  PYTHONPATH=. python -m scenarios.humanoid.render_balance_video
SIMULATION visualization only — env dynamics untouched.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from hymeko_rl.train.sac import build_sac

from .balance_env import BalanceConfig, HumanoidBalanceEnv
from .lyapunov import evaluate_lyapunov

_OUT = Path("reports/2026-07-27-humanoid-sac-residual")
_W, _H, _STRIDE, _FPS = 440, 480, 5, 25


def _greedy(actor, obs) -> np.ndarray:
    with torch.no_grad():
        return actor.action_mean(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()


def rollout_frames(env, act_fn, seed: int) -> tuple[list[np.ndarray], dict]:
    """Roll a full episode; return (rendered RGB frames, evaluate_lyapunov summary).

    Preconditions: ``env`` is a built HumanoidBalanceEnv. Postconditions: one frame per
    ``_STRIDE`` steps; the Lyapunov summary is computed on the full (un-subsampled) V-series.
    """
    import mujoco
    env.model.vis.global_.offwidth = _W
    env.model.vis.global_.offheight = _H
    renderer = mujoco.Renderer(env.model, height=_H, width=_W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, 2.6      # side view (sagittal tipping)
    obs, _ = env.reset(seed=seed)
    frames, vs, done, k = [], [], False, 0
    while not done:
        obs, _r, term, trunc, info = env.step(act_fn(obs))
        vs.append(info["V"])
        if k % _STRIDE == 0:
            cam.lookat[:] = [float(env.data.xpos[env._pelvis, 0]), 0.0, 0.6]
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render().copy())
        k += 1
        done = term or trunc
    renderer.close()
    return frames, evaluate_lyapunov(vs)


def _banner(frame: np.ndarray, title: str, ok: bool) -> np.ndarray:
    """Top banner + colored border: green if the certificate passes, red otherwise."""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    col = (34, 160, 74) if ok else (200, 60, 60)
    d.rectangle([0, 0, img.width - 1, 34], fill=col)
    d.text((8, 9), title, fill=(255, 255, 255))
    d.text((img.width - 96, 9), "CERT PASS" if ok else "CERT FAIL", fill=(255, 255, 255))
    for w in range(4):                                             # border
        d.rectangle([w, w, img.width - 1 - w, img.height - 1 - w], outline=col)
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
    n = min(len(f) for f, _e, _t in clips)
    composed = []
    for i in range(n):
        panels = [_banner(f[i], title, ev["passes"]) for f, ev, title in clips]
        composed.append(np.concatenate(panels, axis=1))
    imageio.mimsave(_OUT / "humanoid_balance_compare.mp4", composed, fps=_FPS)
    imageio.mimsave(_OUT / "humanoid_balance_compare.gif", composed[::2], fps=_FPS // 2)
    verdicts = {title: ("PASS" if ev["passes"] else "FAIL") for _f, ev, title in clips}
    print("wrote", _OUT / "humanoid_balance_compare.mp4", "| verdicts:", verdicts)


if __name__ == "__main__":
    main()
