"""Phase B — does EXACT mirror-equivariance fix the one-sided crab, or only preserve symmetry?

Takes the already-trained one-sided omni policy (+y reached, -y not), forces it to be exactly
mirror-equivariant by Reynolds symmetrization (no retraining), and measures both sides over the
ASYMMETRIC diagonal scaffold and the SYMMETRIC bound scaffold. The central-hypothesis test:
 - Over diag (asymmetric dynamics): symmetrization must NOT produce a two-sided reach — the mirrored
   -y recipe fails on the asymmetric dynamics, so +y is either kept (with -y still missed) or cancelled
   (symmetric mediocrity). Prediction: not two-sided.
 - Over bound (symmetric dynamics): the mirror is a true symmetry -> the two sides match (both weak,
   since bound barely locomotes — Phase A).

Usage::  PYTHONPATH=. python -m scenarios.aibo.run_aibo_mirror_equivariant
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from hymeko_rl.train.sac import build_sac

from .mirror_equivariant import equivariance_residual, symmetrize
from .residual_trot import ResidualTrotConfig, ResidualTrotEnv

_OUT = Path("reports/2026-07-28-aibo-residual-trot")
_MLP_CKPT = _OUT / "aibo_residual_trot_omni_best.pt"
_GRID = [(0.6, 0), (0.6, 20), (0.6, -20), (0.6, 40), (0.6, -40)]


def _greedy(actor):
    def fn(o):
        with torch.no_grad():
            return actor.action_mean(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return fn


def _reach(env: ResidualTrotEnv, act_fn, seed0: int = 500, horizon: int = 2000) -> dict:
    reached, sym, per = 0, {"+y": 0, "-y": 0, "n+": 0, "n-": 0}, []
    for i, (d, b) in enumerate(_GRID):
        md, ok, up = env.rollout_min_dist(act_fn, (d, b), seed=seed0 + i, horizon=horizon)
        valid = bool(ok and up > 0.5)
        reached += int(valid)
        per.append({"bearing": b, "min_dist": round(md, 3), "reached": valid})
        if b > 0:
            sym["n+"] += 1
            sym["+y"] += int(valid)
        elif b < 0:
            sym["n-"] += 1
            sym["-y"] += int(valid)
    return {"reach_rate": reached / len(_GRID), "per_goal": per,
            "plus_y": f"{sym['+y']}/{sym['n+']}", "minus_y": f"{sym['-y']}/{sym['n-']}"}


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    actor, _ = build_sac("mlp", obs_dim=9, flat_dim=9, action_dim=4, action_scale=1.0, hidden=128)
    actor.load_state_dict(torch.load(_MLP_CKPT))
    raw = _greedy(actor)
    sym_greedy = symmetrize(raw, ResidualTrotEnv.mirror_obs, ResidualTrotEnv.mirror_act)

    result = {"note": "SIMULATION. The trained one-sided omni MLP, RAW vs EXACT mirror-equivariant "
                      "(Reynolds symmetrization, no retraining), over the asymmetric diag scaffold and "
                      "the symmetric bound scaffold. Central-hypothesis test: equivariance PRESERVES "
                      "symmetry, cannot MANUFACTURE one the dynamics lacks."}
    for gait in ("diag", "bound"):
        env = ResidualTrotEnv(ResidualTrotConfig(residual_mode="omni", obs_mode="flat", gait_phase=gait), seed=0)
        # equivariance sanity: the symmetrized policy is exactly mirror-equivariant (residual ~ 0)
        obs0 = env.reset()[0]
        eqres = equivariance_residual(sym_greedy, ResidualTrotEnv.mirror_obs, ResidualTrotEnv.mirror_act, obs0)
        result[gait] = {"raw": _reach(env, raw), "mirror_equivariant": _reach(env, sym_greedy),
                        "equivariance_residual": round(eqres, 6)}

    d = result["diag"]["mirror_equivariant"]
    two_sided = d["minus_y"] != "0/2" and d["plus_y"] != "0/2"
    result["verdict"] = ("MIRROR_EQUIVARIANCE_MANUFACTURES_SYMMETRY" if two_sided
                         else "EQUIVARIANCE_PRESERVES_NOT_MANUFACTURES")  # the central-hypothesis outcome
    (_OUT / "result_mirror_equivariant.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
