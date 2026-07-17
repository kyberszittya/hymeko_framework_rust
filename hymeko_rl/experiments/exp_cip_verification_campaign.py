"""Overnight CIP-verification campaign — verify the *publishable* finding across scenarios + robots:

    a **structural (relational) actor** preserves/improves the CIP causal **propel-edge** (leg_speed ⇒
    forward_vx) where a **flat MLP degrades it**, even in a collapse-prone off-policy smoke.

Each cell: CIP-diagnose the scripted demonstrator → train {flat | structural} under the CIP-informed reward
(`vertical_bounce`) + asymmetric CTDE critic + demonstrator BC anchor → CIP **re-diagnose** the learned policy.
Scenarios span **Aibo goal-reach at several distances + the biped humanoid**. Multi-seed; the claim rests on
the **median/IQR of the structural−flat propel-edge delta** (§3), not a single run.

Results stream to a JSONL (resumable, interrupt-safe — the verifiable on-disk artifact §3). Live per-cell
logging + the trainer's per-step logs (no blind run §3). Reuses `exp_aibo_cip_walk` (diagnosis + enhanced
stack), the substrate factories, and the scripted gaits; nothing re-implemented.

Reward alignment: the training reward is `goal_progress` (optimum = reach the goal) + `vertical_bounce`
(shaping; does not move the optimum). Task-aligned by construction; the manipulation `reward_oracle` does not
apply (it certifies grasp/zone/deliver phase rewards).

Run (background, overnight):  ./.venv/bin/python -m hymeko_rl.experiments.exp_cip_verification_campaign
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hymeko_rl.env.locomotion_env import make_cheetah, make_humanoid
from hymeko_rl.env.locomotion_experts import QuadrupedTrotGait
from hymeko_rl.env.reward import RewardSpec
from hymeko_rl.experiments.exp_aibo_cip_walk import CipAiboEnv, cip_diagnose

_OUT = Path("experiments/2026_07_16_cip_verification")

# CIP-informed forward reward for the legged runners (my LeggedLocomotionEnv): forward driver + anti-bounce.
# `bounce` is the anti-vertical-bounce weight; the 2026-07-17 teacher CIP discovery measured the humanoid CpG as
# 3.0× bounce-dominated (bounce-edge 0.766 ≫ propel-edge 0.254), motivating the bounce ∈ {3, 8} A/B.
def legged_cip_reward(bounce: float = 3.0) -> RewardSpec:
    """CIP-informed legged forward reward with a configurable anti-bounce weight.

    # Preconditions ``bounce >= 0``.  # Postconditions ``vertical_bounce`` term carries ``bounce``; others fixed;
    ``bounce=3.0`` reproduces the original ``LEGGED_CIP_REWARD``."""
    return RewardSpec((
        ("goal_progress", 60.0),
        ("vertical_bounce", bounce),    # CIP anti-bounce (A/B axis)
        ("alive", 1.0),
        ("action_cost", 0.05),
    ))


LEGGED_CIP_REWARD = legged_cip_reward()     # back-compat constant: the bounce=3.0 default


def humanoid_run_reward(target_speed: float = 3.0) -> RewardSpec:
    """Fast-running humanoid reward (2026-07-17): drive forward CENTROIDAL momentum (``forward_momentum`` is linear
    in speed → *faster is better*, unbounded) under a control-Lyapunov stability stack (``capture_point`` up-weighted
    — running is unstable, the lateral DCM is the fall mode; ``centroidal_angular_momentum`` for balance;
    ``energy_regulation`` softly pins the high-speed orbit). Set ``env.target_speed = target_speed`` so the energy
    reference matches the running speed."""
    return RewardSpec((
        ("forward_momentum", 12.0),                 # DRIVE: reward forward speed (linear, unbounded)
        ("alive", 2.0),                             # stay up (load-bearing at running speed)
        ("capture_point", 3.0),                     # lateral DCM fall-bound (running is unstable → up-weight)
        ("centroidal_angular_momentum", 1.0),       # balance
        ("transverse_momentum", 1.0),               # forward, not sideways/up
        ("energy_regulation", 0.2),                 # soft: pin the running-orbit energy (H_ref at target_speed)
        ("joint_acceleration", 0.001),              # light smoothness (don't over-damp a dynamic gait)
    ))


def _aibo(goal_distance: float, horizon: int) -> Callable[[], Any]:
    return lambda: CipAiboEnv(goal_distance=goal_distance, max_steps=horizon)


def _legged(make: Callable[..., Any], horizon: int) -> Callable[[], Any]:
    return lambda: make(max_steps=horizon, reward_spec=LEGGED_CIP_REWARD)


def scenarios(horizon: int) -> list[dict]:
    """Env factory + scripted demonstrator per scenario. The demonstrator is the CIP-baseline + BC anchor."""
    trot = QuadrupedTrotGait()
    return [
        {"name": "aibo_goal_3m", "make": _aibo(3.0, horizon), "expert": lambda e: trot.action(e)},
        {"name": "aibo_goal_5m", "make": _aibo(5.0, horizon), "expert": lambda e: trot.action(e)},
        {"name": "humanoid_walk", "make": _legged(make_humanoid, horizon), "expert": lambda e: e.expert_action},
        {"name": "cheetah_run", "make": _legged(make_cheetah, horizon), "expert": lambda e: e.expert_action},
    ]


def _greedy(actor: Any) -> Callable[[Any], np.ndarray]:
    import torch

    def fn(env: Any) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(env.node_features()[None], dtype=torch.float32)
            return np.asarray(actor.action_mean(o).squeeze(0).numpy(), dtype=np.float32)
    return fn


def run_cell(sc: dict, actor_kind: str, seed: int, *, steps: int, horizon: int) -> dict:
    """One verification cell: CIP-diagnose the demonstrator, train {flat|structural}, CIP re-diagnose."""
    import torch

    from hymeko_rl.train.ddpg import build_offpolicy, td3_bc_config, train_offpolicy
    torch.manual_seed(seed)
    make, expert = sc["make"], sc["expert"]

    base = cip_diagnose(make, expert, seeds=3, steps=horizon)          # demonstrator CIP baseline

    obs_d, act_d = [], []                                             # demonstrator demos (BC anchor)
    for s in range(2):
        env = make()
        env.reset(seed=100 + s)
        for _ in range(horizon):
            a = np.asarray(expert(env), dtype=np.float32)
            obs_d.append(env.node_features())
            act_d.append(a)
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
    offline = (np.asarray(obs_d, np.float32), np.asarray(act_d, np.float32))

    env = make()
    flat = int(np.prod(env.observation_space.shape))
    if actor_kind == "structural":
        actor, critics = build_offpolicy("hsikan", obs_dim=2, flat_dim=flat, action_dim=env.n_actions,
                                         action_scale=1.0, n_critics=2, hidden=128, device="cpu", hg_state=env.hg)
    else:
        actor, critics = build_offpolicy("mlp", obs_dim=flat, flat_dim=flat, action_dim=env.n_actions,
                                         action_scale=1.0, n_critics=2, hidden=128, device="cpu")
    cfg = td3_bc_config(total_steps=steps, start_steps=500, batch_size=128, eval_every=steps, n_eval=1,
                        log_every=max(1000, steps // 4), seed=seed)
    train_offpolicy(actor, critics, env, cfg, offline_data=offline)

    learned = cip_diagnose(make, _greedy(actor), seeds=3, steps=horizon)
    return {"scenario": sc["name"], "actor": actor_kind, "seed": seed, "steps": steps,
            "propel_before": base["propel_edge"], "propel_after": learned["propel_edge"],
            "propel_delta": learned["propel_edge"] - base["propel_edge"],
            "bounce_before": base["bounce_edge"], "bounce_after": learned["bounce_edge"],
            "dx_before": base["mean_dx"], "dx_after": learned["mean_dx"]}


def run(*, steps: int = 5000, seeds: tuple = (0, 1, 2, 3, 4), horizon: int = 300) -> dict:
    """Loop scenarios × {flat, structural} × seeds; stream cells to JSONL (resumable); aggregate at the end."""
    _OUT.mkdir(parents=True, exist_ok=True)
    journal = _OUT / "cells.jsonl"
    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            done.add((r["scenario"], r["actor"], r["seed"]))
    scs = scenarios(horizon)
    cells = [(sc, k, s) for sc in scs for k in ("flat", "structural") for s in seeds]
    print(f"[cip-verify] {len(cells)} cells ({len(scs)} scenarios × 2 actors × {len(seeds)} seeds), "
          f"{len(done)} already done; steps={steps}", flush=True)
    t0 = time.time()
    with journal.open("a") as fh:
        for i, (sc, k, s) in enumerate(cells):
            if (sc["name"], k, s) in done:
                continue
            c0 = time.time()
            try:
                rec = run_cell(sc, k, s, steps=steps, horizon=horizon)
            except Exception as exc:                                  # a cell fails → log + continue
                rec = {"scenario": sc["name"], "actor": k, "seed": s, "error": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            el = time.time() - t0
            print(f"[cip-verify] cell {i + 1}/{len(cells)} {sc['name']}/{k}/s{s} "
                  f"propel_delta={rec.get('propel_delta', 'ERR')} | {time.time() - c0:.0f}s "
                  f"| elapsed {el / 60:.0f}m", flush=True)
    return _aggregate(journal)


def _aggregate(journal: Path) -> dict:
    """Median/IQR of the structural−flat propel-edge advantage per scenario (the publishable claim)."""
    recs = [json.loads(x) for x in journal.read_text().splitlines() if x]
    by = {}
    for r in recs:
        if "propel_delta" in r:
            by.setdefault((r["scenario"], r["actor"]), []).append(float(r["propel_delta"]))
    out = {}
    for sc in sorted({s for s, _ in by}):
        f = np.array(by.get((sc, "flat"), [np.nan]))
        st = np.array(by.get((sc, "structural"), [np.nan]))
        out[sc] = {"flat_propel_delta_median": float(np.nanmedian(f)),
                   "structural_propel_delta_median": float(np.nanmedian(st)),
                   "structural_advantage": float(np.nanmedian(st) - np.nanmedian(f)),
                   "structural_iqr": [float(np.nanpercentile(st, 25)), float(np.nanpercentile(st, 75))],
                   "n_seeds": int(min(len(f), len(st)))}
    summary = {"per_scenario": out,
               "verdict_structural_beats_flat": {sc: v["structural_advantage"] > 0 for sc, v in out.items()}}
    (journal.parent / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    import sys
    n = 1500 if "--smoke" in sys.argv else 5000
    sd = (0,) if "--smoke" in sys.argv else (0, 1, 2, 3, 4)
    print(json.dumps(run(steps=n, seeds=sd), indent=2, default=float))
