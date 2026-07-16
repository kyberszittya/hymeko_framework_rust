"""Local SAC-from-scratch walking validation (cheetah) — does **RL from scratch** un-collapse (actually walk)
where short TD3+BC froze and DAgger-from-a-weak-gait could only shuffle? Two arms, an A/B on the demonstrator
(the arc's BC-warm-start-traps question):

  * ``sac_scratch``    — pure SAC, no demonstrator (learns the gait outright; not teacher-capped).
  * ``sac_warmstart``  — SAC with a BC anchor from the scripted CpgGait (does the weak demo help or trap?).

Metric: forward dx + the CIP propel-edge (`leg_speed ⇒ forward_vx`) of each learned policy vs the scripted
gait baseline. Bounded (CPU, ~200k steps/arm); the full multi-body multi-seed campaign is the kato15 job.
Reuses `make_cheetah`, `train_sac`, `cip_diagnose`; nothing re-implemented."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hymeko_rl.env.locomotion_env import make_cheetah
from hymeko_rl.experiments.exp_aibo_cip_walk import cip_diagnose

_OUT = Path("experiments/2026_07_16_sac_walk_validation")
_H = 300


def _greedy(actor: Any) -> Callable[[Any], np.ndarray]:
    import torch

    def fn(env: Any) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(env.node_features()[None], dtype=torch.float32)
            mean = actor.action_mean(o) if hasattr(actor, "action_mean") else actor.act(o)[0]
            return np.asarray(np.reshape(mean.squeeze(0).numpy(), (-1,)), dtype=np.float32)
    return fn


def _gait_demos(make: Callable[[], Any], n: int = 3) -> tuple:
    obs_d, act_d = [], []
    for s in range(n):
        env = make()
        env.reset(seed=200 + s)
        for _ in range(_H):
            a = np.asarray(env.expert_action, dtype=np.float32)
            obs_d.append(env.node_features())
            act_d.append(a)
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    return np.asarray(obs_d, np.float32), np.asarray(act_d, np.float32)


def run_arm(make: Callable[[], Any], name: str, *, warmstart: bool, steps: int, seed: int) -> dict:
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    env = make()
    flat = int(np.prod(env.observation_space.shape))
    actor, critics = build_sac("mlp", obs_dim=flat, flat_dim=flat, action_dim=env.n_actions,
                               action_scale=1.0, n_critics=2, hidden=256, device="cpu")
    offline = _gait_demos(make) if warmstart else None
    cfg = SACConfig(total_steps=steps, start_steps=2000, batch_size=256, eval_every=steps, n_eval=1,
                    log_every=max(2000, steps // 8), seed=seed, bc_coef=(2.5 if warmstart else 0.0))
    t0 = time.time()
    train_sac(actor, critics, env, cfg, offline_data=offline)
    d = cip_diagnose(make, _greedy(actor), seeds=4, steps=_H)
    return {"arm": name, "warmstart": warmstart, "steps": steps, "seed": seed, "wall_s": time.time() - t0,
            "dx": d["mean_dx"], "propel_edge": d["propel_edge"], "bounce_edge": d["bounce_edge"]}


def run(*, steps: int = 200_000, seed: int = 0) -> dict:
    _OUT.mkdir(parents=True, exist_ok=True)
    make = lambda: make_cheetah(max_steps=_H)                       # noqa: E731
    scripted = cip_diagnose(make, lambda e: e.expert_action, seeds=4, steps=_H)
    print(f"[sac-val] scripted gait: dx={scripted['mean_dx']:+.2f} propel={scripted['propel_edge']:+.3f}", flush=True)
    rows = [{"arm": "scripted_gait", "dx": scripted["mean_dx"], "propel_edge": scripted["propel_edge"],
             "bounce_edge": scripted["bounce_edge"]}]
    for name, ws in [("sac_scratch", False), ("sac_warmstart", True)]:
        r = run_arm(make, name, warmstart=ws, steps=steps, seed=seed)
        rows.append(r)
        print(f"[sac-val] {name}: dx={r['dx']:+.2f} propel={r['propel_edge']:+.3f} "
              f"(scripted dx {scripted['mean_dx']:+.2f}) | {r['wall_s']:.0f}s", flush=True)
    result = {"scripted": scripted, "arms": rows, "steps": steps,
              "sac_scratch_walks": rows[1]["dx"] > max(0.05, scripted["mean_dx"]),
              "warmstart_helps": rows[2]["dx"] > rows[1]["dx"]}
    (_OUT / "result.json").write_text(json.dumps(result, indent=2, default=float))
    return result


if __name__ == "__main__":
    import sys
    n = 20_000 if "--smoke" in sys.argv else 200_000
    print(json.dumps(run(steps=n), indent=2, default=float))
