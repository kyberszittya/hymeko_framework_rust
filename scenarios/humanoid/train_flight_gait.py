"""Parallel CEM search for a DYNAMIC FLIGHT-PHASE humanoid gait (see ``flight_gait.py``).

The quasi-static WBC footstep stack is mechanism-walled (shuffle vs lunge-fall). ``FlightGaitEnv`` is a
momentum-based alternative: a cyclic PD gait that pushes off toward a genuine (contact-verified) flight phase.
HONEST STATUS: from a properly PLANTED start (the first result's ~20 % flight was a drop-from-height
artifact, caught + fixed), a small local CEM converged to a grounded forward shuffle (+0.05 m, ~0 % flight)
— it avoids flight because launching tends to fall. This scales the search on katolab (parallel CEM, many
restarts / seeds, HIGH flight weight to force lift-off) to test whether a *controlled forward flight* gait
exists at all, with the corrected flight metric; the result is reported honestly either way.

Usage::  PYTHONPATH=. python -m scenarios.humanoid.train_flight_gait --iters 60 --pop 48 --workers 12 --out <dir>
         PYTHONPATH=. MUJOCO_GL=egl python -m scenarios.humanoid.train_flight_gait --render --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scenarios.humanoid.flight_gait import PDIM, FlightGaitConfig, FlightGaitEnv

_MU0 = np.array([2.0, 0.5, 0.0, 0.5, 0.0, 0.3, 0.3, 0.0, 0.8, 0.0, 0.4])   # a flight-producing seed
_SIG0 = np.array([0.5, 0.4, 0.2, 0.4, 0.2, 0.3, 0.3, 0.2, 0.5, 0.3, 0.3])


def _cfg(w_flight: float, w_forward: float) -> FlightGaitConfig:
    # w_flight up => bigger, more visible hops; w_forward up => faster. The CEM balances against falling.
    return FlightGaitConfig(steps=900, w_flight=w_flight, w_forward=w_forward)


def _eval(args: "tuple[np.ndarray, float, float]") -> "tuple[float, float, float, float]":
    theta, wf, wfo = args
    return FlightGaitEnv(_cfg(wf, wfo), seed=0).rollout(theta, seed=0)


def train(iters: int, pop: int, elite: int, workers: int, out: Path,
          w_flight: float = 3.0, w_forward: float = 6.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu, sig = _MU0.copy(), _SIG0.copy()
    best = (-1e9, _MU0.copy(), 0.0, 0.0, 0.0)                  # (ret, theta, fwd, flight, upright)
    out.mkdir(parents=True, exist_ok=True)
    journal = (out / "journal.jsonl").open("w")
    for it in range(iters):
        cand = mu + sig * rng.standard_normal((pop, PDIM))
        args = [(c, w_flight, w_forward) for c in cand]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(_eval, args))
        else:
            res = [_eval(a) for a in args]
        scores = np.array([r[0] for r in res])
        idx = np.argsort(scores)[::-1][:elite]
        mu, sig = cand[idx].mean(0), cand[idx].std(0) + 0.03
        for c, (r, fwd, fl, up) in zip(cand, res):
            if r > best[0]:
                best = (r, c.copy(), fwd, fl, up)
        journal.write(json.dumps({"iter": it, "best_ret": float(best[0]), "best_fwd": float(best[2]),
                                  "best_flight": float(best[3]), "best_upright": float(best[4])}) + "\n")
        journal.flush()
        print(f"[flight] iter{it} ret={best[0]:+.1f} fwd={best[2]:+.3f} flight={best[3]*100:.0f}% "
              f"up={best[4]*100:.0f}%", flush=True)
    journal.close()
    np.save(out / "best_gait.npy", best[1])
    (out / "result.json").write_text(json.dumps({
        "best_fwd": best[2], "best_flight": best[3], "best_upright": best[4],
        "w_flight": w_flight, "w_forward": w_forward, "iters": iters, "pop": pop}, indent=2))
    print(f"[flight] DONE fwd={best[2]:+.3f} flight={best[3]*100:.0f}% upright={best[4]*100:.0f}%", flush=True)
    return best[1]


def render(gait_path: Path, out: Path) -> None:
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    theta = np.load(gait_path)
    env = FlightGaitEnv(FlightGaitConfig(steps=900), seed=0)
    env.reset(seed=0)
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = 560, 432
    r = mujoco.Renderer(env.model, height=432, width=560)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    freq = float(np.clip(theta[0], 0.5, 4.0))
    dt = float(env.model.opt.timestep)
    px0 = float(env.data.xpos[env._pelvis, 0])
    frames, air = [], 0
    for k in range(900):
        ph = 2.0 * np.pi * freq * k * dt
        tgt = env._phase_targets(theta, ph)
        tau = np.empty(env.model.nu)
        for i, (dof, qa) in enumerate(zip(env._act_dof, env._act_qadr)):
            servo = env.cfg.kp * (tgt[i] - env.data.qpos[qa]) - env.cfg.kv * env.data.qvel[dof]
            tau[i] = np.clip(servo + float(env.data.qfrc_bias[dof]), -env.cfg.tau_max, env.cfg.tau_max)
        env.data.ctrl[:] = tau
        env._mj.mj_step(env.model, env.data)
        if k % 12 == 0:
            fly = env._both_feet_airborne()
            air += int(fly)
            cam.azimuth, cam.elevation, cam.distance = 90, -6, 3.0
            cam.lookat[:] = [float(env.data.xpos[env._pelvis, 0]), 0.0, 0.6]
            r.update_scene(env.data, cam)
            img = Image.fromarray(r.render().copy())
            dr = ImageDraw.Draw(img)
            dr.text((8, 6), "HyMeKo humanoid - DYNAMIC FLIGHT-PHASE gait (push-off + flight + land)",
                    font=font, fill=(240, 240, 240))
            fwd = float(env.data.xpos[env._pelvis, 0]) - px0
            dr.text((8, 26), f"fwd {fwd:+.2f} m   {'AIRBORNE (both feet up)' if fly else 'stance'}",
                    font=font, fill=(90, 220, 120) if fly else (230, 200, 120))
            frames.append(np.asarray(img))
    r.close()
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out / "humanoid_flight_gait.mp4", frames, fps=18)
    print(f"[flight] rendered {out / 'humanoid_flight_gait.mp4'} airborne_frames={air}/{len(frames)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--elite", type=int, default=10)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("HYMEKO_WORKERS", "1")))
    ap.add_argument("--w_flight", type=float, default=3.0)
    ap.add_argument("--w_forward", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/humanoid_flight")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.render:
        render(Path(args.out) / "best_gait.npy", Path(args.out))
        return
    train(args.iters, args.pop, args.elite, args.workers, Path(args.out),
          w_flight=args.w_flight, w_forward=args.w_forward, seed=args.seed)


if __name__ == "__main__":
    main()
