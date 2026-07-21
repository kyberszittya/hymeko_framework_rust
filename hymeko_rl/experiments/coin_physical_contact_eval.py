"""Coin physical-contact rerun — §10 causal evaluation + §11 verdicts (2026-07-22).

Evaluates each trained checkpoint against the scripted-base ceiling on THREE distributions that differ sharply under
the corrected physics — so a single number cannot be conflated across them:

  VAL      seeds 64100-64113 — the in-training eval distribution (BC-init floor: native 0.286, strict 0/14).
  panel    seeds 1011..1568  — the curated 9-state regression panel (scripted base: native 9/9, strict 6/9).
  heldout  seeds 1000-1074   — broader generalization (scripted base: native 0.70).

Reports NATIVE (zone_rate) and STRICT (certified) SEPARATELY (§8) for each of {scripted_base, SAC, TD3} on each
distribution, with the per-seed delta vs the scripted base. The verdict per algorithm is one of
PHYSICS_FIXED_POSITIVE / NO_EFFECT / REGRESSION / RUN_INVALID, decided on the NATIVE metric on the VAL distribution
(the training distribution) with the strict metric reported for context — never the reverse (a metric a failure can
inflate must not drive the verdict).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.experiments.coin_physical_contact_rerun import bc_init_zero_residual
from hymeko_rl.experiments.coin_two_arm_sac import _VAL_SEEDS, direct_env, evaluate

_PANEL = (1011, 1045, 1164, 1174, 1202, 1278, 1358, 1447, 1568)
_HELDOUT = tuple(s for s in range(1000, 1075) if s not in _PANEL)[:50]


def _load_actor(algo: str, path: str):
    if algo == "SAC":
        from hymeko_rl.train.sac import build_sac
        ac, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    else:
        from hymeko_rl.train.ddpg import build_offpolicy
        ac, _ = build_offpolicy("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0, n_critics=2)
    ac.load_state_dict(torch.load(path, weights_only=True))
    ac.eval()
    return ac


def _eval_on(actor, eval_env, seeds) -> dict[str, Any]:
    m = evaluate(eval_env, actor, seeds)
    return {"n": len(seeds), "native_zone": round(float(m["zone_rate"]), 4),
            "strict_rate": round(float(m["strict_rate"]), 4), "strict_count": int(m["strict_count"]),
            "mean_progress": round(float(m["mean_progress"]), 4)}


def _scripted_baseline(eval_env, dists) -> dict[str, Any]:
    """BC zero-residual actor == the scripted grasp_carry base; evaluate it on each distribution as the ceiling."""
    from hymeko_rl.train.sac import build_sac
    ac, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    bc_init_zero_residual(ac)
    ac.eval()
    return {name: _eval_on(ac, eval_env, seeds) for name, seeds in dists.items()}


def _verdict(scripted: dict, trained: dict, *, native_eps: float = 0.05) -> str:
    """Decide on the NATIVE metric on the VAL distribution (training distribution); strict is context only."""
    s = scripted["VAL"]["native_zone"]
    t = trained["VAL"]["native_zone"]
    if t != t:                                            # NaN guard -> invalid run
        return "RUN_INVALID"
    if t >= s + native_eps:
        return "PHYSICS_FIXED_POSITIVE"
    if t <= s - native_eps:
        return "REGRESSION"
    return "NO_EFFECT"


def evaluate_campaign(campaign_dir: str, out: str) -> dict[str, Any]:
    eval_env = direct_env()
    dists = {"VAL": _VAL_SEEDS, "panel": _PANEL, "heldout": _HELDOUT}
    scripted = _scripted_baseline(eval_env, dists)
    print("scripted-base ceiling:", {k: (v["native_zone"], v["strict_count"]) for k, v in scripted.items()}, flush=True)
    results: dict[str, Any] = {"scripted_base": scripted, "runs": {}, "verdicts": {}}
    per_algo: dict[str, list] = {"SAC": [], "TD3": []}
    for run_json in sorted(glob.glob(f"{campaign_dir}/*/run.json")):
        meta = json.loads(Path(run_json).read_text())
        algo, seed = meta["algo"], meta["seed"]
        best_pt = Path(run_json).parent / f"{algo.lower()}_actor_best.pt"
        if not best_pt.is_file():
            print(f"  [skip] {algo} s{seed}: no best checkpoint", flush=True)
            continue
        actor = _load_actor(algo, str(best_pt))
        ev = {name: _eval_on(actor, eval_env, seeds) for name, seeds in dists.items()}
        key = f"{algo}_s{seed}"
        results["runs"][key] = {"best_native_train": meta.get("best_native"), "eval": ev}
        per_algo[algo].append(ev)
        print(f"  {key}: VAL native={ev['VAL']['native_zone']} strict={ev['VAL']['strict_count']} | "
              f"panel native={ev['panel']['native_zone']} strict={ev['panel']['strict_count']}", flush=True)
    # per-algo median across seeds + verdict (median VAL native vs scripted)
    for algo, evs in per_algo.items():
        if not evs:
            continue
        med = {d: {"native_zone": round(float(np.median([e[d]["native_zone"] for e in evs])), 4),
                   "strict_count": int(np.median([e[d]["strict_count"] for e in evs]))}
               for d in dists}
        v = _verdict(scripted, med)
        results["verdicts"][algo] = {"median": med, "verdict": v, "n_seeds": len(evs)}
        print(f"VERDICT {algo}: {v} | VAL native {scripted['VAL']['native_zone']}->{med['VAL']['native_zone']} "
              f"strict {scripted['VAL']['strict_count']}->{med['VAL']['strict_count']} (median over {len(evs)} seeds)",
              flush=True)
    Path(out).write_text(json.dumps(results, indent=1, default=float))
    print(f"WROTE {out}", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    evaluate_campaign(a.campaign, a.out)


if __name__ == "__main__":
    main()
