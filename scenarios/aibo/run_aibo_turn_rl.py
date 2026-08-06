"""HSiKAN vs MLP on the TURNING / goal-reaching problem — the whole-body-coordination testbed.

The crab was a simple lateral push (HSiKAN ≈ MLP). Turning is whole-body coordination: route the heading
error through the body's kinematic structure into a coordinated per-leg stride correction. This trains a
BOUNDED RESIDUAL (``leg`` mode, 12-dim) over the working ``turn_then_walk`` scaffold (a=0 already reaches
0.50) on a WIDE bearing distribution (±135°), and compares MLP (flat obs) vs HSiKAN (signedkan over the
body hypergraph) — same pooled head, same action, multi-seed — on goal-reach. The user's hypothesis:
structural propagation helps here where it did not for the crab.

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_turn_rl --seeds 0 1 2 --steps 25000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from hymeko_rl.train.sac import AlphaMode, SACConfig, build_sac, train_sac

from .residual_trot import ResidualTrotConfig, ResidualTrotEnv

_OUT = Path("reports/2026-07-29-aibo-turn-rl")
_TEST = [(d, b) for d in (0.5, 0.7) for b in (0, 20, -20, 40, -40, 90, -90, 135, -135)]
# VAL spans difficulties: mid bearings (must PRESERVE) + wide (the headroom) — so mean-dist selection
# catches a residual that destroys the easy goals while chasing the hard ones.
_VAL = [(0.6, 40), (0.6, -40), (0.6, 90), (0.6, -90), (0.6, 135)]


def _cfg_env(obs_mode: str) -> ResidualTrotEnv:
    return ResidualTrotEnv(ResidualTrotConfig(
        residual_mode="phase", obs_mode=obs_mode, heading_mode="turn_then_walk",
        residual_scale=0.12, bearing_deg=135.0, dist_lo=0.5, dist_hi=0.7, max_steps=1600), seed=0)


def _greedy(actor):
    def fn(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return fn


def _eval(env: ResidualTrotEnv, act_fn, grid=_TEST, horizon=2400) -> "tuple[float, float]":
    """Return ``(reach_rate, mean_min_dist)``. mean_min_dist is the DENSE selection signal — binary reach
    is 0 across the wide-bearing VAL early on, giving no gradient to pick a checkpoint (the bug that made
    the first run select an untrained checkpoint)."""
    hit = 0.0
    dists = []
    for i, (d, b) in enumerate(grid):
        md, ok, up = env.rollout_min_dist(act_fn, (d, b), seed=500 + i, horizon=horizon)
        hit += float(bool(ok and up > 0.5))
        dists.append(md)
    return round(hit / len(grid), 3), round(float(sum(dists) / len(dists)), 4)


def _reach(env: ResidualTrotEnv, act_fn, grid=_TEST, horizon=2400) -> float:
    return _eval(env, act_fn, grid, horizon)[0]


def _train(kind: str, seed: int, steps: int) -> float:
    obs_mode = "flat" if kind == "mlp" else "hypergraph"
    env = _cfg_env(obs_mode)
    torch.manual_seed(seed)
    if kind == "mlp":
        actor, critics = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=12, action_scale=1.0, hidden=128)
    else:
        n = env._n_vtx
        actor, critics = build_sac("signedkan", obs_dim=4, flat_dim=n * 4, action_dim=12,
                                   action_scale=1.0, hidden=64, actor_head="pooled", hg_state=env.hg)
    best = {"dist": 1e9, "sd": None}

    def eval_fn(e, a) -> float:
        reach, mean_dist = _eval(e, _greedy(a), _VAL, horizon=1600)
        if mean_dist < best["dist"]:                       # select by DENSE mean-distance, not binary reach
            best["dist"] = mean_dist
            best["sd"] = {k: v.detach().clone() for k, v in a.state_dict().items()}
        return reach

    cfg = SACConfig(total_steps=steps, start_steps=1_000, batch_size=256, update_every=2,
                    eval_every=max(steps // 4, 1_000), log_every=steps, seed=seed,
                    alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)
    train_sac(actor, critics, env, cfg, eval_fn=eval_fn)
    if best["sd"] is not None:
        actor.load_state_dict(best["sd"])
        torch.save(best["sd"], _OUT / f"turn_{kind}_seed{seed}_best.pt")   # persist for post-hoc analysis
    reach, per_goal = _eval_per_goal(env, _greedy(actor))
    return {"reach": reach, "per_goal": per_goal}


def _eval_per_goal(env: ResidualTrotEnv, act_fn, grid=_TEST, horizon=2400) -> "tuple[float, list]":
    hit = 0.0
    per = []
    for i, (d, b) in enumerate(grid):
        md, ok, up = env.rollout_min_dist(act_fn, (d, b), seed=500 + i, horizon=horizon)
        v = bool(ok and up > 0.5)
        hit += float(v)
        per.append({"dist": d, "bearing": b, "min_dist": round(md, 3), "reached": v})
    return round(hit / len(grid), 3), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25_000)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    # scaffold baseline (a=0 turn_then_walk) on the same grid
    zfn = lambda o: __import__("numpy").zeros(12, dtype="float32")  # noqa: E731
    scaffold = _reach(_cfg_env("flat"), zfn)

    per_seed = []
    for seed in args.seeds:
        row = {"seed": seed, "mlp": _train("mlp", seed, args.steps), "hsikan": _train("hsikan", seed, args.steps)}
        per_seed.append(row)
        print(f"seed {seed}: scaffold {scaffold} | mlp {row['mlp']['reach']} | hsikan {row['hsikan']['reach']}")

    def _agg(k: str) -> dict:
        reaches = [r[k]["reach"] for r in per_seed]
        return {"median_reach": median(reaches), "max_reach": max(reaches), "reaches": reaches,
                "beats_scaffold_seeds": sum(x > scaffold for x in reaches)}   # the ceiling question

    agg = {k: _agg(k) for k in ("mlp", "hsikan")}
    result = {"scaffold_reach": scaffold, "n_seeds": len(args.seeds), "steps": args.steps,
              "aggregate": agg, "per_seed": per_seed,
              "note": "SIMULATION. Bounded phase-residual over turn_then_walk, wide bearings (+-135). MLP (flat) "
                      "vs HSiKAN (signedkan/33-vtx body-hg, pooled). beats_scaffold_seeds = the ceiling test."}
    (_OUT / "result_turn_rl_multiseed.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for k in ("mlp", "hsikan"):
        a = agg[k]
        print(f"{k:7s}: reaches {a['reaches']} | median {a['median_reach']} max {a['max_reach']} "
              f"| beats scaffold {a['beats_scaffold_seeds']}/{len(args.seeds)}")


if __name__ == "__main__":
    main()
