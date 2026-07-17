"""Humanoid RUNNING (fast) — SAC from scratch with the Hamiltonian fast-run reward (forward centroidal momentum,
linear/unbounded → faster is better, under a control-Lyapunov stability stack). Longer horizon (H=500) so the gait
builds up speed; saves the trained actor for a GIF. Metric = forward displacement dx + est. speed (m/s) + CIP edges."""
# ruff: noqa: E402, E702  — the §4 RSS cap (setrlimit) MUST precede the torch import; one-liners are operational.
import json
import os
import resource
import time

import numpy as np

try:
    resource.setrlimit(resource.RLIMIT_DATA, (16 * 1024**3, 16 * 1024**3))
except (ValueError, OSError) as e:
    print("[hrun] rlimit note:", e, flush=True)
import torch
torch.set_float32_matmul_precision('high')
from hymeko_rl.env.locomotion_env import make_humanoid
from hymeko_rl.experiments.exp_cip_verification_campaign import humanoid_run_reward
from hymeko_rl.experiments.exp_aibo_cip_walk import cip_diagnose
from hymeko_rl.experiments.exp_sac_walk_validation import _greedy
from hymeko_rl.train.sac import SACConfig, build_sac, train_sac

H = 500
TARGET = float(os.environ.get("HYMEKO_TARGET", "3.0"))          # m/s — the running speed the energy orbit targets
STEPS = int(os.environ.get("HYMEKO_STEPS", "1500000"))          # running is harder than walking → more steps
SEEDS = tuple(int(x) for x in os.environ.get("HYMEKO_SEEDS", "0,1,2").split(","))
OUT = os.environ.get("HYMEKO_OUT", "experiments/2026_07_17_humanoid_run")
DEV = os.environ.get("HYMEKO_DEVICE", "cuda")


def make_runner() -> object:
    e = make_humanoid(max_steps=H, reward_spec=humanoid_run_reward(TARGET))
    e.target_speed = TARGET
    return e


os.makedirs(OUT, exist_ok=True)
journal = os.path.join(OUT, "cells.jsonl")
done = set()
if os.path.exists(journal):
    for line in open(journal):
        try:
            r = json.loads(line)
            if "dx" in r:
                done.add(r["seed"])
        except Exception:
            pass
print(f"[hrun] humanoid RUNNING target={TARGET} m/s steps={STEPS} seeds={SEEDS} H={H} -> {OUT}", flush=True)
with open(journal, "a") as fh:
    for s in SEEDS:
        if s in done:
            continue
        env = make_runner()
        flat = int(np.prod(env.observation_space.shape))
        actor, critics = build_sac("mlp", obs_dim=flat, flat_dim=flat, action_dim=env.n_actions,
                                   action_scale=1.0, n_critics=2, hidden=256, device=DEV)
        cfg = SACConfig(total_steps=STEPS, start_steps=2000, batch_size=256, eval_every=STEPS, n_eval=1,
                        log_every=min(5000, max(2000, STEPS // 10)), seed=s, bc_coef=0.0, compile=True)
        t0 = time.time()
        try:
            train_sac(actor, critics, env, cfg, eval_fn=lambda e, a: 0.0)
            torch.save({"state_dict": actor.to("cpu").state_dict(), "kind": "mlp", "target": TARGET},
                       os.path.join(OUT, f"runner_s{s}.pt"))                # for a GIF later
            actor.to(DEV)
            d = cip_diagnose(make_runner, _greedy(actor), seeds=4, steps=H)
            dt = float(getattr(env, "dt", env.model.opt.timestep))
            speed = d["mean_dx"] / (H * dt) if dt > 0 else 0.0
            rec = {"seed": s, "dx": d["mean_dx"], "speed_mps": speed, "propel_edge": d["propel_edge"],
                   "bounce_edge": d["bounce_edge"], "wall_s": time.time() - t0}
        except Exception as exc:
            rec = {"seed": s, "error": f"{type(exc).__name__}: {exc}"}
        fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
        print(f"[hrun] s{s} dx={rec.get('dx', 'ERR')} speed={rec.get('speed_mps', '')}m/s "
              f"propel={rec.get('propel_edge', '')} | {(time.time() - t0) / 60:.0f}m", flush=True)
print("[hrun] DONE", flush=True)
