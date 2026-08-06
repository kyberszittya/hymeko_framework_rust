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
    ap.add_argument("--hstar", type=float, default=0.0,
                    help="structural entropy H★ exploration coef (HSiKAN-only seat) — escape the crab local optimum")
    ap.add_argument("--kind", default="signedkan", choices=("signedkan", "mixture", "hsikan", "sa_hsikan"),
                    help="backbone: signedkan | mixture (HSiKAN+MLP gated MoE, Kato's mixed) | ...")
    ap.add_argument("--head", default="per_node", choices=("per_node", "pooled"),
                    help="per_node (each action from its vertex) | pooled (the whole structured LATENT -> action)")
    ap.add_argument("--skip", default="none", choices=("none", "residual", "highway"),
                    help="per-layer skip in the signed-KAN backbone: none (plain signed-conv, the default so far) "
                         "| residual | highway (the Schmidhuber gate — the 'H' in HSiKAN). Ignored by "
                         "mixture/sa_hsikan (structural constants).")
    ap.add_argument("--gait", default="diag", choices=("diag", "bound", "pace", "pronk"),
                    help="base-gait phase (Phase A): diag (trot, asymmetric — default) | bound (front/back, "
                         "instantaneously LEFT-RIGHT SYMMETRIC — the symmetric-scaffold test for a two-sided crab)")
    ap.add_argument("--hidden", type=int, default=64, help="backbone width (128 = MLP-matched, to escape the null residual)")
    ap.add_argument("--explore", action="store_true",
                    help="stronger exploration (AUTO alpha) to break the ride-the-scaffold null-residual local optimum")
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.kind}_{args.head}"
    if args.symmetric:
        tag += "_sym"
    if args.hstar > 0.0:
        tag += f"_hstar{args.hstar:g}"
    if args.skip != "none":
        tag += f"_{args.skip}"
    if args.gait != "diag":
        tag += f"_{args.gait}"
    if args.hidden != 64:
        tag += f"_h{args.hidden}"
    if args.explore:
        tag += "_explore"

    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="leg_hypergraph",
                                             leg_hg_symmetric=args.symmetric, gait_phase=args.gait), seed=0)
    n, feat = env._n_vtx, 4
    torch.manual_seed(0)

    def _build():
        kw = dict(obs_dim=feat, flat_dim=n * feat, action_dim=4, action_scale=1.0, hidden=args.hidden,
                  actor_head=args.head, hg_state=env.hg, skip=args.skip)  # skip: mixture/sa_hsikan ignore it
        if args.head == "per_node":
            kw["act_vertices"] = env._abd_vtx
        return build_sac(args.kind, **kw)

    actor, critics = _build()

    best_path = _OUT / f"aibo_residual_trot_omni_{tag}_best.pt"
    best = {"rate": -1.0}

    def eval_fn(e, a) -> float:
        rate = _reach(e, _greedy(a), _VAL_GRID, horizon=800)["reach_rate"]   # fast selection eval
        if rate > best["rate"]:
            best["rate"] = rate
            torch.save(a.state_dict(), best_path)
        return rate

    _alpha = (dict(alpha_mode=AlphaMode.AUTO, init_alpha=0.1) if args.explore
              else dict(alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6))
    cfg = SACConfig(total_steps=args.steps, start_steps=1_000, batch_size=128, update_every=3,
                    eval_every=max(args.steps // 4, 1_000), log_every=4_000, seed=0,
                    struct_entropy_coef=args.hstar, **_alpha)  # H★ seat + exploration schedule
    train_sac(actor, critics, env, cfg, eval_fn=eval_fn)

    if not best_path.exists():
        torch.save(actor.state_dict(), best_path)
    best_actor, _ = _build()
    best_actor.load_state_dict(torch.load(best_path))
    hsikan = _reach(env, _greedy(best_actor), _TEST_GRID, horizon=2000)

    # MLP baseline (flat obs) on the SAME symmetric grid
    mlp_env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat", gait_phase=args.gait), seed=0)
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
