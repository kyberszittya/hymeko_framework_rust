"""Train a FORWARD-walking footstep policy over the WBC/DCM footstep env (CEM policy search).

The hand-tuned analytical control falls backward; a learned footstep policy walks forward (see
`reports/2026-07-29-humanoid-footstep-rl.md`). This scales that up: a parallel cross-entropy-method search
over a small MLP foothold policy, saving the best policy + a metrics journal, and (``--render``) a video of
the learned humanoid walking forward. CPU-bound (MuJoCo + the WBC KKT solve per tick) — throughput scales
with cores, so it runs well on a many-core box (katolab). Deterministic per seed.

Usage:
    PYTHONPATH=. python -m scenarios.humanoid.train_footstep_walk --iters 40 --pop 24 --out <dir>
    PYTHONPATH=. python -m scenarios.humanoid.train_footstep_walk --render --policy <dir>/best_policy.npy
Env knobs: HYMEKO_ITERS, HYMEKO_POP, HYMEKO_WORKERS, HYMEKO_FWD_STRIDE, HYMEKO_W_FORWARD, HYMEKO_OUT.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scenarios.humanoid.footstep_env import FootstepConfig, HumanoidFootstepEnv

def _cfg(fwd_stride: float, w_forward: float, max_footsteps: int, fall_penalty: float = 25.0,
         model_src: str = "humanoid.hymeko", toe_off: float = 0.0) -> FootstepConfig:
    # heavy fall penalty + the per-step forward cap (in FootstepConfig) => the policy must walk forward
    # SUSTAINABLY, not lunge into a terminal fall (which gamed the earlier reward). model_src/toe_off select
    # the articulated-toe (push-off) model + a scripted toe-off.
    return FootstepConfig(max_footsteps=max_footsteps, forward_stride=fwd_stride,
                          w_forward=w_forward, residual_xy=0.06, fall_penalty=fall_penalty,
                          model_src=model_src, toe_off=toe_off)


def _dim(obs_dim: int) -> int:
    return 2 * obs_dim + 2                    # a linear foothold policy: a = tanh(W·obs + b)


def policy(theta: np.ndarray, obs: np.ndarray, obs_dim: int) -> np.ndarray:
    w = theta[:2 * obs_dim].reshape(2, obs_dim)
    b = theta[2 * obs_dim:]
    return np.tanh(w @ obs + b)


def rollout(theta: np.ndarray, cfg: FootstepConfig, seed: int = 0) -> "tuple[float, int, float]":
    """One episode; returns (return, footsteps survived, net forward displacement)."""
    env = HumanoidFootstepEnv(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    od = obs.shape[0]
    px0 = float(env.data.xpos[env._pel, 0])
    ret, steps = 0.0, 0
    for _ in range(cfg.max_footsteps):
        obs, r, done, trunc, info = env.step(policy(theta, obs, od).astype(np.float32))
        ret += r
        steps = info["steps"]
        if done or trunc:
            break
    return ret, steps, float(env.data.xpos[env._pel, 0]) - px0


def _eval(args):
    theta, cfg = args
    return rollout(theta, cfg)


def train(iters: int, pop: int, elite: int, cfg: FootstepConfig, workers: int, out: Path) -> np.ndarray:
    env0 = HumanoidFootstepEnv(cfg, seed=0)
    od = env0.observation_space.shape[0]
    dim = _dim(od)
    rng = np.random.default_rng(0)
    mu, sig = np.zeros(dim), np.ones(dim) * 0.5
    best = (-1e9, np.zeros(dim), 0, -1e9)                          # (ret, theta, steps, fwd)
    out.mkdir(parents=True, exist_ok=True)
    journal = (out / "journal.jsonl").open("w")
    scaff = rollout(np.zeros(dim), cfg)
    for it in range(iters):
        cand = mu + sig * rng.standard_normal((pop, dim))
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                res = list(ex.map(_eval, [(c, cfg) for c in cand]))
        else:
            res = [rollout(c, cfg) for c in cand]
        scores = np.array([r[0] for r in res])
        idx = np.argsort(scores)[::-1][:elite]
        mu, sig = cand[idx].mean(0), cand[idx].std(0) + 0.04
        for c, (rr, ss, ff) in zip(cand, res):
            if rr > best[0]:                                      # track the highest-RETURN policy (survive+forward)
                best = (rr, c.copy(), ss, ff)
        row = {"iter": it, "elite_ret": float(scores[idx].mean()), "best_fwd": float(best[3])}
        journal.write(json.dumps(row) + "\n")
        journal.flush()
        print(f"[fwalk] iter{it} elite_ret={row['elite_ret']:.1f} best_fwd={best[3]:+.3f}", flush=True)
    journal.close()
    np.save(out / "best_policy.npy", best[1])
    (out / "result.json").write_text(json.dumps({
        "scaffold_fwd": scaff[2], "best_fwd": best[3], "best_steps": best[2],
        "iters": iters, "pop": pop}, indent=2))
    print(f"[fwalk] DONE scaffold_fwd={scaff[2]:+.3f} best_fwd={best[3]:+.3f} steps={best[2]}", flush=True)
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=int(os.environ.get("HYMEKO_ITERS", "40")))
    ap.add_argument("--pop", type=int, default=int(os.environ.get("HYMEKO_POP", "24")))
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("HYMEKO_WORKERS", "1")))
    ap.add_argument("--fwd_stride", type=float, default=float(os.environ.get("HYMEKO_FWD_STRIDE", "0.06")))
    ap.add_argument("--w_forward", type=float, default=float(os.environ.get("HYMEKO_W_FORWARD", "45.0")))
    ap.add_argument("--max_footsteps", type=int, default=60)
    ap.add_argument("--out", type=str, default=os.environ.get("HYMEKO_OUT", "experiments/humanoid_fwalk"))
    ap.add_argument("--model_src", type=str, default=os.environ.get("HYMEKO_MODEL", "humanoid.hymeko"))
    ap.add_argument("--toe_off", type=float, default=float(os.environ.get("HYMEKO_TOE_OFF", "0.0")))
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--policy", type=str, default="")
    args = ap.parse_args()
    cfg = _cfg(args.fwd_stride, args.w_forward, args.max_footsteps,
               model_src=args.model_src, toe_off=args.toe_off)
    if args.render:
        render(Path(args.policy or Path(args.out) / "best_policy.npy"), cfg, Path(args.out))
        return
    train(args.iters, args.pop, args.elite, cfg, args.workers, Path(args.out))


def render(policy_path: Path, cfg: FootstepConfig, out: Path) -> None:
    import imageio.v2 as imageio
    import mujoco
    theta = np.load(policy_path)
    env = HumanoidFootstepEnv(cfg, seed=0)
    od = env.observation_space.shape[0]
    obs, _ = env.reset(seed=0)
    env.model.vis.global_.offwidth, env.model.vis.global_.offheight = 520, 460
    r = mujoco.Renderer(env.model, height=460, width=520)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 90.0, -8.0, 2.4
    frames, px0 = [], float(env.data.xpos[env._pel, 0])
    for _ in range(cfg.max_footsteps):
        a = policy(theta, obs, od).astype(np.float32)
        # render mid-footstep frames by stepping the env's internal loop is not exposed; snapshot per footstep
        obs, _rw, done, trunc, info = env.step(a)
        cam.lookat[:] = [float(env.data.xpos[env._pel, 0]), float(env.data.xpos[env._pel, 1]), 0.5]
        r.update_scene(env.data, cam)
        frames.append(r.render().copy())
        if done or trunc:
            break
    r.close()
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out / "humanoid_forward_walk.mp4", frames, fps=8)
    print(f"[fwalk] rendered {out / 'humanoid_forward_walk.mp4'} "
          f"net_forward={float(env.data.xpos[env._pel, 0]) - px0:+.3f} m", flush=True)


if __name__ == "__main__":
    main()
