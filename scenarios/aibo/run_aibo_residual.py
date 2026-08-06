"""Residual SAC over the trot scaffold — does a bounded residual reach MORE goal positions?

Trains a bounded residual (coin-R8 regime) on a distribution of goal positions and asks the campaign
question: does the learned residual raise the multi-position REACH RATE above the trot-gait scaffold
(``a = 0``) on a **held-out** grid of (distance × bearing) goals? Best-validation checkpoint; the
held-out test grid is reported once. Baseline = the scaffold on the same grid.

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_residual --steps 60000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hymeko_rl.train.sac import AlphaMode, SACConfig, build_sac, train_sac

from .residual_trot import ResidualTrotConfig, ResidualTrotEnv

_OUT = Path("reports/2026-07-28-aibo-residual-trot")
# held-out goal grids (distance m, bearing deg) — VAL selects the checkpoint, TEST reported once
_VAL_GRID = [(0.55, 15), (0.55, -15), (0.7, 35), (0.7, -35)]
_TEST_GRID = [(0.6, 0), (0.6, 20), (0.6, -20), (0.6, 40), (0.6, -40)]


def _greedy(actor, obs) -> np.ndarray:
    with torch.no_grad():
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return actor.action_mean(t).squeeze(0).cpu().numpy()


def _reach_rate(env: ResidualTrotEnv, act_fn, grid, seed0: int = 500, horizon: int = 1200) -> tuple[float, float]:
    """Return (reach rate within reach_radius, mean min-distance) on a fixed goal grid."""
    reached, dists = 0, []
    for i, goal in enumerate(grid):
        md, ok, _up = env.rollout_min_dist(act_fn, goal, seed=seed0 + i, horizon=horizon)
        reached += int(ok)
        dists.append(md)
    return reached / len(grid), float(np.mean(dists))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--mode", choices=("leg", "steer", "phase", "omni"), default="leg",
                    help="residual over raw leg targets (leg), gait steering+speed params (steer), "
                         "per-leg PHASE-GATED leg targets (phase), or per-leg ABDUCTION for lateral "
                         "omnidirectional crab (omni — the richer action space)")
    ap.add_argument("--smoke", action="store_true", help="short production-scale smoke run")
    ap.add_argument("--mirror", action="store_true", help="mirror-augment training (symmetry preservation, both crab sides)")
    args = ap.parse_args()
    steps = 3_000 if args.smoke else args.steps
    _OUT.mkdir(parents=True, exist_ok=True)

    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode=args.mode, mirror_augment=args.mirror), seed=0)
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    torch.manual_seed(0)
    actor, critics = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                               action_dim=act_dim, action_scale=1.0, hidden=128)

    zero = np.zeros(act_dim, np.float32)
    base_rate, base_dist = _reach_rate(env, lambda _o: zero, _TEST_GRID)   # scaffold baseline (test)

    _msuf = "_mirror" if args.mirror else ""
    best_path = _OUT / f"aibo_residual_trot_{args.mode}{_msuf}_best.pt"
    best = {"rate": -1.0}

    def eval_fn(e, a) -> float:
        rate, _ = _reach_rate(e, lambda o: _greedy(a, o), _VAL_GRID)       # selection = VAL reach rate
        if rate > best["rate"]:
            best["rate"] = rate
            torch.save(a.state_dict(), best_path)
        return rate

    cfg = SACConfig(total_steps=steps, start_steps=1_000, batch_size=256,
                    eval_every=max(steps // 6, 1_000), log_every=2_000, seed=0,
                    alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)
    curve = train_sac(actor, critics, env, cfg, eval_fn=eval_fn)

    if not best_path.exists():
        torch.save(actor.state_dict(), best_path)
    best_actor, _ = build_sac("mlp", obs_dim=obs_dim, flat_dim=obs_dim,
                              action_dim=act_dim, action_scale=1.0, hidden=128)
    best_actor.load_state_dict(torch.load(best_path))
    res_rate, res_dist = _reach_rate(env, lambda o: _greedy(best_actor, o), _TEST_GRID)
    delta = round(res_rate - base_rate, 3)
    result = {
        "mode": args.mode,
        "verdict": ("RESIDUAL_REACHES_MORE_POSITIONS" if delta > 0.15
                    else "RESIDUAL_MATCHES_SCAFFOLD" if abs(delta) <= 0.15
                    else "RESIDUAL_REGRESSES_SCAFFOLD"),
        "scaffold_reach_rate_test": round(base_rate, 3),
        "scaffold_mean_min_dist_test": round(base_dist, 3),
        "residual_best_val_reach_rate_val": round(best["rate"], 3),
        "residual_reach_rate_test": round(res_rate, 3),
        "residual_mean_min_dist_test": round(res_dist, 3),
        "reach_rate_delta_test": delta,
        "eval_curve_val": [round(c, 3) for c in curve],
        "total_steps": steps,
        "smoke": args.smoke,
        "note": "SIMULATION. Bounded residual (scale 0.25) over the trot-gait scaffold (a=0), multi-goal "
                "(dist 0.5-0.75, bearing +-40). Baseline = scaffold on the same held-out grid.",
    }
    tag = "smoke" if args.smoke else (args.mode + ("_mirror" if args.mirror else ""))
    (_OUT / f"result_{tag}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
