"""kato15 SAC-from-scratch walking campaign — the scaled, multi-body, multi-seed version of the local
validation (`exp_sac_walk_validation`), which proved: **pure SAC from scratch walks the cheetah** (dx +0.23,
CIP propel-edge +0.84) while a **weak-gait BC warm-start traps** (dx +0.01). So this campaign is **pure SAC
from scratch only** — no warm-start — and asks the original thread's question at scale:

    across bodies (Aibo goal-reach / biped humanoid / planar cheetah), does the **structural (relational)
    actor** learn a faster walk + a stronger CIP propel-edge than a **flat MLP**, over seeds?

Per cell: build {flat | structural} SAC → train from scratch → measure forward dx + the CIP propel-edge
(`leg_speed ⇒ forward_vx`). Results stream to JSONL (resumable, the verifiable artifact); aggregate to
per-(body, actor) median/IQR. GPU (~1135 steps/s on kato15) makes 500k-step × multi-seed × multi-body tractable
(the local CPU could only do the cheetah at 200k). Reuses the SAC validation cell + `cip_diagnose`.

Run on kato15 (see scripts/kato15/run_sac_walk.sh):
    python -m hymeko_rl.experiments.exp_sac_walk_campaign            # full
    python -m hymeko_rl.experiments.exp_sac_walk_campaign --smoke    # 20k, 1 seed
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hymeko_rl.env.locomotion_env import make_cheetah, make_humanoid
from hymeko_rl.experiments.exp_aibo_cip_walk import CipAiboEnv, cip_diagnose
from hymeko_rl.experiments.exp_cip_verification_campaign import LEGGED_CIP_REWARD
from hymeko_rl.experiments.exp_sac_walk_validation import _greedy

_DEVICE = os.environ.get("HYMEKO_DEVICE", "cpu")   # kato15 launch script exports HYMEKO_DEVICE=cuda
_OUT = Path("experiments/2026_07_16_sac_walk_campaign")
_H = 300


def scenarios() -> list[dict]:
    return [
        {"name": "aibo_goal", "make": lambda: CipAiboEnv(goal_distance=4.0, max_steps=_H)},
        {"name": "humanoid_walk", "make": lambda: make_humanoid(max_steps=_H, reward_spec=LEGGED_CIP_REWARD)},
        {"name": "cheetah_run", "make": lambda: make_cheetah(max_steps=_H, reward_spec=LEGGED_CIP_REWARD)},
    ]


def run_cell(make: Callable[[], Any], actor_kind: str, seed: int, *, steps: int) -> dict:
    """Pure SAC from scratch with a {flat|structural} actor → dx + CIP propel-edge of the learned policy."""
    from hymeko_rl.train.sac import SACConfig, build_sac, train_sac
    env = make()
    flat = int(np.prod(env.observation_space.shape))
    try:
        if actor_kind == "structural":
            actor, critics = build_sac("hsikan", obs_dim=2, flat_dim=flat, action_dim=env.n_actions,
                                       action_scale=1.0, n_critics=2, hidden=256, device=_DEVICE, hg_state=env.hg)
            kind = "structural"
        else:
            raise ValueError("flat")
    except Exception:
        actor, critics = build_sac("mlp", obs_dim=flat, flat_dim=flat, action_dim=env.n_actions,
                                   action_scale=1.0, n_critics=2, hidden=256, device=_DEVICE)
        kind = actor_kind if actor_kind == "flat" else "flat(fallback)"
    cfg = SACConfig(total_steps=steps, start_steps=2000, batch_size=256, eval_every=steps, n_eval=1,
                    log_every=max(2000, steps // 10), seed=seed, bc_coef=0.0)     # bc_coef 0 = pure scratch
    t0 = time.time()
    train_sac(actor, critics, env, cfg)
    d = cip_diagnose(make, _greedy(actor), seeds=4, steps=_H)
    return {"actor": kind, "seed": seed, "steps": steps, "wall_s": time.time() - t0,
            "dx": d["mean_dx"], "propel_edge": d["propel_edge"], "bounce_edge": d["bounce_edge"]}


def run(*, steps: int = 500_000, seeds: tuple = (0, 1, 2, 3, 4)) -> dict:
    _OUT.mkdir(parents=True, exist_ok=True)
    journal = _OUT / "cells.jsonl"
    done = {(r["scenario"], r["actor"], r["seed"]) for r in
            (json.loads(x) for x in journal.read_text().splitlines()) if "dx" in r} if journal.exists() else set()
    scs = scenarios()
    cells = [(sc, k, s) for sc in scs for k in ("flat", "structural") for s in seeds]
    print(f"[sac-walk] {len(cells)} cells, {len(done)} done, steps={steps}", flush=True)
    t0 = time.time()
    with journal.open("a") as fh:
        for i, (sc, k, s) in enumerate(cells):
            if any((sc["name"], a, s) in done for a in (k, "flat(fallback)")):
                continue
            try:
                rec = {"scenario": sc["name"], **run_cell(sc["make"], k, s, steps=steps)}
            except Exception as exc:
                rec = {"scenario": sc["name"], "actor": k, "seed": s, "error": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            print(f"[sac-walk] {i + 1}/{len(cells)} {sc['name']}/{k}/s{s} dx={rec.get('dx', 'ERR')} "
                  f"propel={rec.get('propel_edge', '')} | elapsed {(time.time() - t0) / 60:.0f}m", flush=True)
    return _aggregate(journal)


def _aggregate(journal: Path) -> dict:
    recs = [json.loads(x) for x in journal.read_text().splitlines() if x]
    by: dict = {}
    for r in recs:
        if "dx" in r:
            by.setdefault((r["scenario"], r["actor"].replace("(fallback)", "")), {"dx": [], "propel": []})
            by[(r["scenario"], r["actor"].replace("(fallback)", ""))]["dx"].append(float(r["dx"]))
            by[(r["scenario"], r["actor"].replace("(fallback)", ""))]["propel"].append(float(r["propel_edge"]))
    out: dict = {}
    for (sc, ak), v in by.items():
        out.setdefault(sc, {})[ak] = {"dx_median": float(np.median(v["dx"])),
                                      "propel_median": float(np.median(v["propel"])), "n": len(v["dx"])}
    (journal.parent / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    print(json.dumps(run(steps=20_000 if smoke else 500_000, seeds=(0,) if smoke else (0, 1, 2, 3, 4)),
                     indent=2, default=float))
