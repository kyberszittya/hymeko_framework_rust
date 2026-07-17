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
from hymeko_rl.experiments.exp_cip_verification_campaign import legged_cip_reward
from hymeko_rl.experiments.exp_sac_walk_validation import _greedy

_DEVICE = os.environ.get("HYMEKO_DEVICE", "cpu")   # kato15 launch script exports HYMEKO_DEVICE=cuda
# Per-box output dir (kato14/kato15 share an NFS home, so give each its own JSONL to avoid concurrent writes).
_OUT = Path(os.environ.get("HYMEKO_OUT", "experiments/2026_07_16_sac_walk_campaign"))
_H = 300


def scenarios(bounce: float = 3.0, bodies: "tuple[str, ...] | None" = None) -> list[dict]:
    """Env factories for the campaign, at a given anti-bounce reward weight.

    ``bounce`` sets the ``vertical_bounce`` weight (CIP anti-bounce A/B axis; default 3.0 reproduces prior runs).
    ``bodies`` filters to a subset by name (e.g. only the bounce-dominated tall bodies for the reward A/B)."""
    allsc = [
        {"name": "aibo_goal", "make": lambda: CipAiboEnv(goal_distance=4.0, max_steps=_H, bounce=bounce)},
        {"name": "humanoid_walk", "make": lambda: make_humanoid(max_steps=_H, reward_spec=legged_cip_reward(bounce))},
        {"name": "cheetah_run", "make": lambda: make_cheetah(max_steps=_H, reward_spec=legged_cip_reward(bounce))},
    ]
    return [s for s in allsc if bodies is None or s["name"] in bodies]


def run_cell(make: Callable[[], Any], actor_kind: str, seed: int, *, steps: int,
             compile: bool = False, update_every: int = 1) -> dict:
    """Pure SAC from scratch with a {flat|structural} actor → dx + CIP propel-edge of the learned policy.

    ``compile`` torch.compiles the update (CUDA-only, no-op on CPU — the structural 8.25× lever, 2026-07-17);
    ``update_every>1`` trades sample efficiency for wall (§6.5 #19)."""
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
                    log_every=max(2000, steps // 10), seed=seed, bc_coef=0.0,       # bc_coef 0 = pure scratch
                    compile=compile, update_every=update_every)
    t0 = time.time()
    # No-op eval_fn: skip train_sac's default greedy eval (its CPU obs mismatches a cuda actor on GPU); the
    # real measurement is cip_diagnose below (device-safe via _greedy).
    train_sac(actor, critics, env, cfg, eval_fn=lambda env, actor: 0.0)
    d = cip_diagnose(make, _greedy(actor), seeds=4, steps=_H)
    return {"actor": kind, "seed": seed, "steps": steps, "wall_s": time.time() - t0,
            "dx": d["mean_dx"], "propel_edge": d["propel_edge"], "bounce_edge": d["bounce_edge"]}


def run(*, steps: int = 500_000, seeds: tuple = (0, 1, 2, 3, 4),
        bounce_weights: tuple = (3.0,), bodies: "tuple[str, ...] | None" = None,
        actors: tuple = ("flat", "structural"), compile: bool = False, update_every: int = 1) -> dict:
    """Pure-SAC-from-scratch grid over {body × actor × seed × anti-bounce weight}.

    Defaults reproduce the original 30-cell (bounce=3.0, all bodies, both actors) grid. The 2026-07-17 reward
    A/B narrows to the bounce-dominated tall bodies at ``bounce_weights=(3.0, 8.0)``. ``compile`` gives the
    structural actor the CUDA-graph update speedup (no-op on CPU). Resumable via the JSONL."""
    _OUT.mkdir(parents=True, exist_ok=True)
    journal = _OUT / "cells.jsonl"
    done = {(r["scenario"], r["actor"], r["seed"], float(r.get("bounce", 3.0))) for r in
            (json.loads(x) for x in journal.read_text().splitlines()) if "dx" in r} if journal.exists() else set()
    cells = [(sc, k, s, b) for b in bounce_weights for sc in scenarios(b, bodies)
             for k in actors for s in seeds]
    print(f"[sac-walk] {len(cells)} cells, {len(done)} done, steps={steps} "
          f"bounce={bounce_weights} bodies={bodies or 'all'} compile={compile} update_every={update_every}", flush=True)
    t0 = time.time()
    with journal.open("a") as fh:
        for i, (sc, k, s, b) in enumerate(cells):
            if any((sc["name"], a, s, b) in done for a in (k, "flat(fallback)")):
                continue
            try:
                rec = {"scenario": sc["name"], "bounce": b,
                       **run_cell(sc["make"], k, s, steps=steps, compile=compile, update_every=update_every)}
            except Exception as exc:
                rec = {"scenario": sc["name"], "bounce": b, "actor": k, "seed": s,
                       "error": f"{type(exc).__name__}: {exc}"}
            fh.write(json.dumps(rec, default=float) + "\n")
            fh.flush()
            print(f"[sac-walk] {i + 1}/{len(cells)} {sc['name']}/{k}/s{s}/b{b:g} dx={rec.get('dx', 'ERR')} "
                  f"propel={rec.get('propel_edge', '')} | elapsed {(time.time() - t0) / 60:.0f}m", flush=True)
    return _aggregate(journal)


def _aggregate(journal: Path) -> dict:
    recs = [json.loads(x) for x in journal.read_text().splitlines() if x]
    by: dict = {}
    for r in recs:
        if "dx" in r:
            key = (r["scenario"], r["actor"].replace("(fallback)", ""), float(r.get("bounce", 3.0)))
            by.setdefault(key, {"dx": [], "propel": []})
            by[key]["dx"].append(float(r["dx"]))
            by[key]["propel"].append(float(r["propel_edge"]))
    out: dict = {}
    for (sc, ak, b), v in by.items():
        out.setdefault(sc, {})[f"{ak}@bounce{b:g}"] = {
            "dx_median": float(np.median(v["dx"])), "propel_median": float(np.median(v["propel"])),
            "n": len(v["dx"])}
    (journal.parent / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="SAC-from-scratch walking campaign (structural vs flat; CIP anti-bounce A/B)")
    ap.add_argument("--smoke", action="store_true", help="20k, 1 seed, Aibo flat, bounce {3,8} — new-path check")
    ap.add_argument("--bounce-ab", action="store_true",
                    help="CIP anti-bounce reward A/B on the bounce-dominated tall bodies (Aibo+humanoid), bounce {3,8}")
    ap.add_argument("--with-structural", action="store_true",
                    help="also run the structural actor (its update is compiled — the 2026-07-17 CUDA-graph speedup)")
    ap.add_argument("--steps", type=int, default=None, help="override steps/cell")
    ap.add_argument("--seeds", type=int, default=5, help="seed count (0..n-1)")
    ap.add_argument("--no-compile", action="store_true", help="disable the compiled update (default: on; no-op on CPU)")
    ap.add_argument("--update-every", type=int, default=1, help="1 update per N env steps (>1 = §6.5 #19 A/B, not default)")
    a = ap.parse_args()
    seeds = tuple(range(a.seeds))
    actors = ("flat", "structural") if a.with_structural else ("flat",)
    common = dict(compile=not a.no_compile, update_every=a.update_every)   # compile-on by default; CPU no-op, GPU 8.25×
    if a.smoke:
        kw = dict(steps=a.steps or 20_000, seeds=(0,), bounce_weights=(3.0, 8.0),
                  bodies=("aibo_goal",), actors=("flat",), **common)
    elif a.bounce_ab:
        kw = dict(steps=a.steps or 800_000, seeds=seeds, bounce_weights=(3.0, 8.0),
                  bodies=("aibo_goal", "humanoid_walk"), actors=actors, **common)
    else:  # original grid reproduction (all bodies, both actors, bounce=3.0)
        kw = dict(steps=a.steps or 500_000, seeds=seeds, bounce_weights=(3.0,),
                  actors=("flat", "structural"), **common)
    print(json.dumps(run(**kw), indent=2, default=float))
