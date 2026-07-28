"""HSiKAN vs MLP for the omni crab — does hypergraph STRUCTURE PROPAGATION fix the crab asymmetry?

The MLP omni residual (flat 9-D obs) learned a ONE-SIDED crab (+y reached, −y not) — a flat net has
separate weights per action dim and no left/right structure. This trains a **signedkan** actor over
the body's kinematic **hypergraph** (per-vertex obs, a per-node head reading the 4 abduction
vertices, signed hyperedges routing a GLOBAL lateral goal-demand to the per-leg abduction). The
per-node weight-sharing across legs makes a SYMMETRIC crab representable — the user's hypothesis:
HSiKAN's value is structure propagation between the observation and action spaces, not raw capacity.

Compares signedkan (structured) vs the trained MLP (flat) on a SYMMETRIC held-out grid (both signs).

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_hsikan_omni --steps 120000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hymeko_rl.train.sac import AlphaMode, SACConfig, build_sac, train_sac

from .residual_trot import ResidualTrotConfig, ResidualTrotEnv

_OUT = Path("reports/2026-07-28-aibo-residual-trot")
_MLP_CKPT = _OUT / "aibo_residual_trot_omni_best.pt"
# a SYMMETRIC grid (both +y and -y) — the asymmetry is the whole point. VAL is small (signedkan
# per-step is ~15x MLP, so eval rollouts are the bottleneck); TEST is the full symmetric grid.
_VAL_GRID = [(0.6, 20), (0.6, -20)]
_TEST_GRID = [(0.6, 0), (0.6, 20), (0.6, -20), (0.6, 40), (0.6, -40)]


def _greedy(actor):
    def fn(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return fn


def _reach(env: ResidualTrotEnv, act_fn, grid, seed0=500, horizon=2400) -> dict:
    reached, sym = 0, {"+y": 0, "-y": 0, "n+": 0, "n-": 0}
    per = []
    for i, (d, b) in enumerate(grid):
        md, ok, up = env.rollout_min_dist(act_fn, (d, b), seed=seed0 + i, horizon=horizon)
        valid = bool(ok and up > 0.5)
        reached += int(valid)
        per.append({"bearing": b, "min_dist": md, "reached": valid})
        if b > 0:
            sym["n+"] += 1
            sym["+y"] += int(valid)
        elif b < 0:
            sym["n-"] += 1
            sym["-y"] += int(valid)
    return {"reach_rate": reached / len(grid), "per_goal": per,
            "plus_y": f"{sym['+y']}/{sym['n+']}", "minus_y": f"{sym['-y']}/{sym['n-']}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--symmetric", action="store_true",
                    help="encode the LEFT/RIGHT symmetry axis in the leg-hg signs (the sharper test)")
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    tag = "signedkan_sym" if args.symmetric else "signedkan"

    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="leg_hypergraph",
                                             leg_hg_symmetric=args.symmetric), seed=0)
    n, feat = env._n_vtx, 4
    torch.manual_seed(0)
    actor, critics = build_sac("signedkan", obs_dim=feat, flat_dim=n * feat, action_dim=4,
                               action_scale=1.0, hidden=64, actor_head="per_node",
                               act_vertices=env._abd_vtx, hg_state=env.hg)

    best_path = _OUT / f"aibo_residual_trot_omni_{tag}_best.pt"
    best = {"rate": -1.0}

    def eval_fn(e, a) -> float:
        rate = _reach(e, _greedy(a), _VAL_GRID, horizon=800)["reach_rate"]   # fast selection eval
        if rate > best["rate"]:
            best["rate"] = rate
            torch.save(a.state_dict(), best_path)
        return rate

    cfg = SACConfig(total_steps=args.steps, start_steps=1_000, batch_size=128, update_every=3,
                    eval_every=max(args.steps // 4, 1_000), log_every=4_000, seed=0,
                    alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)
    train_sac(actor, critics, env, cfg, eval_fn=eval_fn)

    if not best_path.exists():
        torch.save(actor.state_dict(), best_path)
    best_actor, _ = build_sac("signedkan", obs_dim=feat, flat_dim=n * feat, action_dim=4,
                              action_scale=1.0, hidden=64, actor_head="per_node",
                              act_vertices=env._abd_vtx, hg_state=env.hg)
    best_actor.load_state_dict(torch.load(best_path))
    hsikan = _reach(env, _greedy(best_actor), _TEST_GRID, horizon=2000)

    # MLP baseline (flat obs) on the SAME symmetric grid
    mlp_env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat"), seed=0)
    mlp_actor, _ = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=4, action_scale=1.0, hidden=128)
    mlp = None
    if _MLP_CKPT.exists():
        mlp_actor.load_state_dict(torch.load(_MLP_CKPT))
        mlp = _reach(mlp_env, _greedy(mlp_actor), _TEST_GRID, horizon=2000)

    result = {
        "verdict": ("SIGNEDKAN_FIXES_CRAB_ASYMMETRY" if mlp and hsikan["minus_y"] != "0/2"
                    and hsikan["minus_y"] > mlp["minus_y"]
                    else "SIGNEDKAN_MATCHES_MLP" if mlp and abs(hsikan["reach_rate"] - mlp["reach_rate"]) < 0.15
                    else "SEE_NUMBERS"),
        "signedkan_test": hsikan, "mlp_test": mlp,
        "signedkan_reach": hsikan["reach_rate"], "mlp_reach": mlp["reach_rate"] if mlp else None,
        "total_steps": args.steps,
        "note": "SIMULATION. signedkan per-node actor over the AIBO kinematic hypergraph (structure "
                "propagation) vs MLP over the flat obs, both omni (abduction crab), symmetric grid.",
    }
    (_OUT / f"result_hsikan_omni_{tag}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("verdict", "signedkan_reach", "mlp_reach")}, indent=2))
    print("signedkan +y/-y:", hsikan["plus_y"], hsikan["minus_y"],
          "| mlp +y/-y:", (mlp["plus_y"], mlp["minus_y"]) if mlp else None)


if __name__ == "__main__":
    main()
