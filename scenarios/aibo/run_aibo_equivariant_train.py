"""Phase B culmination — TRAIN a hard mirror-equivariant policy (equivariant by construction).

Post-hoc symmetrization (Phase B) fixed the trained MLP but not the null-residual HSiKAN. This trains
the equivariant policy IN THE LOOP: the actor is Reynolds-symmetrized every step, so SAC optimises a
policy that is exactly two-sided by construction, and (unlike post-hoc) it is FORCED to learn an active
crab that works on both sides. The mirror is the hand-validated flat one by default, or the one READ
FROM THE HYMEKO STRUCTURE (--structural), the concrete "structure as the equivariance signal".

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_equivariant_train --steps 30000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hymeko_rl.train.sac import AlphaMode, SACConfig, build_sac, train_sac

from .equivariant_actor import MirrorEquivariantActor, equivariance_residual, mirror_obs_flat, mirror_pre_act
from .residual_trot import ResidualTrotConfig, ResidualTrotEnv

_OUT = Path("reports/2026-07-28-aibo-residual-trot")
_VAL = [(0.6, 20), (0.6, -20)]
_TEST = [(0.6, 0), (0.6, 20), (0.6, -20), (0.6, 40), (0.6, -40)]


def _greedy(actor):
    def fn(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return fn


def _reach(env, act_fn, grid, horizon=2000) -> dict:
    reached, sym = 0, {"+y": 0, "-y": 0, "n+": 0, "n-": 0}
    per = []
    for i, (d, b) in enumerate(grid):
        md, ok, up = env.rollout_min_dist(act_fn, (d, b), seed=500 + i, horizon=horizon)
        v = bool(ok and up > 0.5)
        reached += int(v)
        per.append({"bearing": b, "min_dist": round(md, 3), "reached": v})
        if b > 0:
            sym["n+"] += 1
            sym["+y"] += int(v)
        elif b < 0:
            sym["n-"] += 1
            sym["-y"] += int(v)
    return {"reach_rate": reached / len(grid), "per_goal": per,
            "plus_y": f"{sym['+y']}/{sym['n+']}", "minus_y": f"{sym['-y']}/{sym['n-']}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--gait", default="diag", choices=("diag", "bound"))
    ap.add_argument("--warmstart", action="store_true",
                    help="init the base from the raw active-crab MLP (break-symmetry-to-DISCOVER first), then "
                         "continue equivariant (impose-symmetry-to-GENERALISE) — the best-of-both probe")
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat", gait_phase=args.gait), seed=0)
    torch.manual_seed(0)
    base, critics = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=4, action_scale=1.0, hidden=128)
    if args.warmstart:
        raw = _OUT / "aibo_residual_trot_omni_best.pt"          # the unconstrained policy that DISCOVERED the +y crab
        base.load_state_dict(torch.load(raw))
    actor = MirrorEquivariantActor(base, mirror_obs_flat, mirror_pre_act)

    tag = f"{args.gait}{'_warmstart' if args.warmstart else ''}"
    best_path = _OUT / f"aibo_equivariant_mlp_{tag}_best.pt"
    best = {"rate": -1.0}

    def eval_fn(e, a) -> float:
        rate = _reach(e, _greedy(a), _VAL, horizon=800)["reach_rate"]
        if rate > best["rate"]:
            best["rate"] = rate
            torch.save(a.base.state_dict(), best_path)
        return rate

    cfg = SACConfig(total_steps=args.steps, start_steps=1_000, batch_size=256,
                    eval_every=max(args.steps // 6, 1_000), log_every=4_000, seed=0,
                    alpha_mode=AlphaMode.ANNEAL, init_alpha=0.1, alpha_final=0.005, anneal_frac=0.6)
    train_sac(actor, critics, env, cfg, eval_fn=eval_fn)

    if not best_path.exists():
        torch.save(base.state_dict(), best_path)
    eval_base, _ = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=4, action_scale=1.0, hidden=128)
    eval_base.load_state_dict(torch.load(best_path))
    eval_actor = MirrorEquivariantActor(eval_base, mirror_obs_flat, mirror_pre_act)
    test = _reach(env, _greedy(eval_actor), _TEST)
    eqres = equivariance_residual(eval_actor, mirror_obs_flat,
                                  lambda a: -a[..., [1, 0, 3, 2]], torch.randn(4, 9))

    result = {
        "verdict": ("EQUIVARIANT_TRAIN_REACHES_BOTH_SIDES" if test["minus_y"] != "0/2" and test["plus_y"] != "0/2"
                    else "STILL_ONE_SIDED"),
        "test": test, "equivariance_residual": round(eqres, 6), "gait": args.gait, "total_steps": args.steps,
        "note": "SIMULATION. In-loop hard mirror-equivariant MLP (Reynolds-symmetrized mean), omni crab, "
                "flat validated mirror. Equivariant BY CONSTRUCTION -> two-sided if it learns any crab.",
    }
    result["warmstart"] = args.warmstart
    (_OUT / f"result_equivariant_mlp_{tag}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("verdict", "equivariance_residual")}, indent=2))
    print("test +y/-y:", test["plus_y"], test["minus_y"], "| reach", test["reach_rate"])


if __name__ == "__main__":
    main()
