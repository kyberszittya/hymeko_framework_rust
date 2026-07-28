"""Multi-seed test of the central symmetry claim — no single-measurement conclusions.

The one-seed result was: post-hoc symmetrization gives a two-sided crab (0.60) while in-loop equivariance
degrades to symmetric-null (0.20). This runs the three recipes across SEEDS and reports per-seed +y/-y and
the aggregate (how many seeds reach BOTH sides), so the ordering is a distribution, not a point:

  raw        - unconstrained MLP (discovers the crab by breaking symmetry, one-sided)
  post-hoc   - the SAME raw policy, mirror-symmetrized at deploy (Reynolds average)
  in-loop    - a from-scratch hard mirror-equivariant MLP (symmetry imposed during training)

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_symmetry_multiseed --seeds 0 1 2 --steps 20000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import torch

from hymeko_rl.train.sac import AlphaMode, SACConfig, build_sac, train_sac

from .equivariant_actor import MirrorEquivariantActor, mirror_obs_flat, mirror_pre_act
from .residual_trot import ResidualTrotConfig, ResidualTrotEnv

_OUT = Path("reports/2026-07-28-aibo-residual-trot")
_VAL = [(0.6, 20), (0.6, -20)]
_TEST = [(0.6, 0), (0.6, 20), (0.6, -20), (0.6, 40), (0.6, -40)]


def _greedy(actor):
    def fn(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return fn


def _reach(env, act_fn, grid=_TEST, horizon=2000) -> dict:
    reached = {"+y": 0, "-y": 0, "n+": 0, "n-": 0}
    hit = 0
    for i, (d, b) in enumerate(grid):
        _md, ok, up = env.rollout_min_dist(act_fn, (d, b), seed=500 + i, horizon=horizon)
        v = bool(ok and up > 0.5)
        hit += int(v)
        if b > 0:
            reached["n+"] += 1
            reached["+y"] += int(v)
        elif b < 0:
            reached["n-"] += 1
            reached["-y"] += int(v)
    return {"reach": round(hit / len(grid), 3), "plus_y": reached["+y"], "minus_y": reached["-y"],
            "two_sided": reached["+y"] > 0 and reached["-y"] > 0}


def _cfg(steps: int, seed: int) -> SACConfig:
    return SACConfig(total_steps=steps, start_steps=1_000, batch_size=256,
                     eval_every=max(steps // 4, 1_000), log_every=steps, seed=seed,
                     alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)


def _train_mlp(env, steps: int, seed: int, *, equivariant: bool):
    torch.manual_seed(seed)
    base, critics = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=4, action_scale=1.0, hidden=128)
    actor = MirrorEquivariantActor(base, mirror_obs_flat, mirror_pre_act) if equivariant else base
    best = {"rate": -1.0, "sd": None}

    def eval_fn(e, a) -> float:
        r = _reach(e, _greedy(a), _VAL, horizon=800)["reach"]
        if r > best["rate"]:
            best["rate"] = r
            best["sd"] = {k: v.detach().clone() for k, v in base.state_dict().items()}
        return r

    train_sac(actor, critics, env, _cfg(steps, seed), eval_fn=eval_fn)
    if best["sd"] is not None:
        base.load_state_dict(best["sd"])
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=20_000)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat", gait_phase="diag"), seed=0)

    per_seed = []
    for seed in args.seeds:
        raw = _train_mlp(env, args.steps, seed, equivariant=False)
        raw_g = _greedy(raw)
        post = MirrorEquivariantActor(raw, mirror_obs_flat, mirror_pre_act)   # SAME policy, symmetrized at deploy
        inloop = _train_mlp(env, args.steps, seed, equivariant=True)
        row = {"seed": seed,
               "raw": _reach(env, raw_g),
               "post_hoc": _reach(env, _greedy(post)),
               "in_loop": _reach(env, _greedy(inloop))}
        per_seed.append(row)
        print(f"seed {seed}: raw {row['raw']['plus_y']}/{row['raw']['minus_y']} "
              f"| post-hoc {row['post_hoc']['plus_y']}/{row['post_hoc']['minus_y']} "
              f"| in-loop {row['in_loop']['plus_y']}/{row['in_loop']['minus_y']}")

    agg = {r: {"two_sided_seeds": sum(x[r]["two_sided"] for x in per_seed),
               "median_reach": median(x[r]["reach"] for x in per_seed)}
           for r in ("raw", "post_hoc", "in_loop")}
    result = {"n_seeds": len(args.seeds), "seeds": args.seeds, "steps": args.steps,
              "aggregate": agg, "per_seed": per_seed,
              "note": "SIMULATION. MLP omni crab, diag scaffold. two_sided = reaches BOTH +y and -y. "
                      "Multi-seed so the recipe ordering is a distribution, not one measurement."}
    (_OUT / "result_symmetry_multiseed.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\naggregate (two-sided seeds / median reach):")
    for r in ("raw", "post_hoc", "in_loop"):
        print(f"  {r:9s}: {agg[r]['two_sided_seeds']}/{len(args.seeds)} two-sided | median reach {agg[r]['median_reach']}")


if __name__ == "__main__":
    main()
