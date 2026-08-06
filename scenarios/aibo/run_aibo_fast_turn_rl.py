"""Speed objective — can a balance-entropy-rewarded policy turn FASTER without tipping? (user's idea, 1a)

The rotational-couple turn caps at ~47°/1000 (deterministic ceiling; faster tips). The main goal
(reachability) is solved at a long horizon, but at a SHORT horizon (2400) the stable turn only reaches
0.56 — the wide bearings need faster turning. This trains a leg-residual (larger authority) over the
turn_then_walk scaffold with the movement/balance-grounded reward ``+ balance_w · H_bal`` (foot-support
entropy — a dense stay-upright signal where a reactive gate failed) and asks: at horizon 2400, does it
beat the 0.56 stable scaffold by turning faster while staying up? Conditions: MLP+balance, MLP no-balance
(ablation), HSiKAN+balance (does structure help under real tipping pressure?). Multi-seed.

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_fast_turn_rl --seeds 0 1 2 --steps 25000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from hymeko_rl.train.sac import AlphaMode, SACConfig, build_sac, train_sac

from .residual_trot import ResidualTrotConfig, ResidualTrotEnv

_OUT = Path("reports/2026-07-29-aibo-fast-turn-rl")
_TEST = [(d, b) for d in (0.5, 0.7) for b in (0, 20, -20, 40, -40, 90, -90, 135, -135)]
_VAL = [(0.6, 40), (0.6, -40), (0.6, 90), (0.6, -90), (0.6, 135)]
_HORIZON = 2400                                            # SHORT horizon = the speed pressure (stable turn = 0.56 here)


def _cfg_env(obs_mode: str, balance_w: float, stability_w: float = 0.0) -> ResidualTrotEnv:
    return ResidualTrotEnv(ResidualTrotConfig(
        residual_mode="leg", obs_mode=obs_mode, heading_mode="turn_then_walk", residual_scale=0.4,
        balance_w=balance_w, stability_w=stability_w,
        bearing_deg=135.0, dist_lo=0.5, dist_hi=0.7, max_steps=1600), seed=0)


def _greedy(actor):
    def fn(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return fn


def _eval(env: ResidualTrotEnv, act_fn, grid=_TEST, horizon=_HORIZON) -> "tuple[float, float]":
    hit = 0.0
    dists = []
    for i, (d, b) in enumerate(grid):
        md, ok, up = env.rollout_min_dist(act_fn, (d, b), seed=500 + i, horizon=horizon)
        hit += float(bool(ok and up > 0.5))
        dists.append(md)
    return round(hit / len(grid), 3), round(float(np.mean(dists)), 4)


def _train(kind: str, balance_w: float, stability_w: float, seed: int, steps: int) -> float:
    obs_mode = "flat" if kind == "mlp" else "hypergraph"
    env = _cfg_env(obs_mode, balance_w, stability_w)
    torch.manual_seed(seed)
    if kind == "mlp":
        actor, critics = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=12, action_scale=1.0, hidden=128)
    else:
        n = env._n_vtx
        actor, critics = build_sac("signedkan", obs_dim=4, flat_dim=n * 4, action_dim=12,
                                   action_scale=1.0, hidden=64, actor_head="pooled", hg_state=env.hg)
    best = {"dist": 1e9, "sd": None}

    def eval_fn(e, a) -> float:
        _reach, mean_dist = _eval(e, _greedy(a), _VAL, horizon=1600)
        if mean_dist < best["dist"]:
            best["dist"] = mean_dist
            best["sd"] = {k: v.detach().clone() for k, v in a.state_dict().items()}
        return _reach

    cfg = SACConfig(total_steps=steps, start_steps=1_000, batch_size=256, update_every=2,
                    eval_every=max(steps // 4, 1_000), log_every=steps, seed=seed,
                    alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)
    train_sac(actor, critics, env, cfg, eval_fn=eval_fn)
    if best["sd"] is not None:
        actor.load_state_dict(best["sd"])
    return _eval(env, _greedy(actor))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=25_000)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    zfn = lambda o: np.zeros(12, dtype=np.float32)  # noqa: E731
    scaffold = _eval(_cfg_env("flat", 0.0), zfn)[0]        # a=0 turn_then_walk at horizon 2400 (the 0.56 baseline)

    # (name, kind, balance_w, stability_w): isolate the PREDICTIVE stability signal + the HSiKAN structure
    conds = [("mlp_bal_stab", "mlp", 0.5, 1.0), ("mlp_bal_only", "mlp", 0.5, 0.0),
             ("hsikan_bal_stab", "hsikan", 0.5, 1.0)]
    per_seed = []
    for seed in args.seeds:
        row = {"seed": seed}
        for name, kind, bw, sw in conds:
            row[name] = _train(kind, bw, sw, seed, args.steps)
        per_seed.append(row)
        print(f"seed {seed}: scaffold {scaffold} | " + " | ".join(f"{n} {row[n]}" for n, _, _, _ in conds))

    agg = {n: {"median": median(r[n] for r in per_seed), "max": max(r[n] for r in per_seed),
               "beats_scaffold": sum(r[n] > scaffold for r in per_seed)} for n, _, _, _ in conds}
    result = {"scaffold_reach_h2400": scaffold, "horizon": _HORIZON, "n_seeds": len(args.seeds),
              "steps": args.steps, "aggregate": agg, "per_seed": per_seed,
              "note": "SIMULATION. Leg-residual (scale 0.4) over turn_then_walk, WIDE bearings, SHORT horizon "
                      "2400 (speed pressure). balance_w·H_bal (foot-support entropy) reward. Does it turn faster "
                      "w/o tipping vs the 0.56 stable scaffold, and does HSiKAN structure help?"}
    (_OUT / "result_fast_turn.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nscaffold(h2400) {scaffold}")
    for n, _, _, _ in conds:
        print(f"  {n:15s}: median {agg[n]['median']} max {agg[n]['max']} beats_scaffold {agg[n]['beats_scaffold']}/{len(args.seeds)}")


if __name__ == "__main__":
    main()
