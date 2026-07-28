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
_VAL = [(0.6, 90), (0.6, -90), (0.6, 135)]                 # the wide bearings = the headroom


def _cfg_env(obs_mode: str) -> ResidualTrotEnv:
    return ResidualTrotEnv(ResidualTrotConfig(
        residual_mode="leg", obs_mode=obs_mode, heading_mode="turn_then_walk",
        bearing_deg=135.0, dist_lo=0.5, dist_hi=0.7, max_steps=1600), seed=0)


def _greedy(actor):
    def fn(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return fn


def _reach(env: ResidualTrotEnv, act_fn, grid=_TEST, horizon=2400) -> float:
    hit = 0
    for i, (d, b) in enumerate(grid):
        _md, ok, up = env.rollout_min_dist(act_fn, (d, b), seed=500 + i, horizon=horizon)
        hit += int(bool(ok and up > 0.5))
    return round(hit / len(grid), 3)


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
    best = {"rate": -1.0, "sd": None}

    def eval_fn(e, a) -> float:
        r = _reach(e, _greedy(a), _VAL, horizon=1600)
        if r > best["rate"]:
            best["rate"] = r
            best["sd"] = {k: v.detach().clone() for k, v in a.state_dict().items()}
        return r

    cfg = SACConfig(total_steps=steps, start_steps=1_000, batch_size=256, update_every=2,
                    eval_every=max(steps // 4, 1_000), log_every=steps, seed=seed,
                    alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)
    train_sac(actor, critics, env, cfg, eval_fn=eval_fn)
    if best["sd"] is not None:
        actor.load_state_dict(best["sd"])
    return _reach(env, _greedy(actor))


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
        print(f"seed {seed}: scaffold {scaffold} | mlp {row['mlp']} | hsikan {row['hsikan']}")

    agg = {k: {"median_reach": median(r[k] for r in per_seed), "max_reach": max(r[k] for r in per_seed)}
           for k in ("mlp", "hsikan")}
    result = {"scaffold_reach": scaffold, "n_seeds": len(args.seeds), "steps": args.steps,
              "aggregate": agg, "per_seed": per_seed,
              "note": "SIMULATION. Bounded leg-residual over turn_then_walk, wide bearings (+-135). MLP (flat) "
                      "vs HSiKAN (signedkan/body-hg, pooled). Does structure help the whole-body turning?"}
    (_OUT / "result_turn_rl.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nscaffold {scaffold} | MLP median {agg['mlp']['median_reach']} max {agg['mlp']['max_reach']} "
          f"| HSiKAN median {agg['hsikan']['median_reach']} max {agg['hsikan']['max_reach']}")


if __name__ == "__main__":
    main()
