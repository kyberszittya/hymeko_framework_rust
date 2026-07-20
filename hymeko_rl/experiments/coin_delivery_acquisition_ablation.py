"""COIN-DELIVERY-OVERNIGHT-2 PART III — acquisition component ablation.

Takes the oracle's best acquisition params and measures the MARGINAL recovery per component by disabling each at equal
budget, against the load-bearing structural control (correct vs scrambled geometry). No RL. The claim of geometry
usefulness rests on correct>scrambled at equal capacity, NOT on full-primitive vs grasp_carry.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace

import numpy as np

from hymeko_rl.experiments.coin_delivery_acquisition1 import _OUT, _SCRAMBLE, _states
from hymeko_rl.train.coin_delivery_acquisition import AcqParams, ApproachMode, eval_acquisition, make_acq_env

_MAN = _OUT / "manifests" / "coin_delivery_acquisition.json"


def _log(m: str) -> None:
    print(m, flush=True)


def _best_params() -> AcqParams:
    d = json.loads(_MAN.read_text())["best_params"]
    d = {k: (ApproachMode(v) if k == "approach_mode" else v) for k, v in d.items()}
    return AcqParams(**d)


def _ablations(base: AcqParams) -> dict:
    """The 12 preregistered ablations as (name -> params or 'random')."""
    return {
        "full_primitive": base,
        "no_geometry_angle": replace(base, geometry_conditioned=False, approach_angle=0.0),
        "symmetric_only": replace(base, approach_mode=ApproachMode.SYMMETRIC, asym_offset=0.0),
        "no_asymmetric_correction": replace(base, asym_offset=0.0),
        "no_staged_closure": replace(base, align_steps=0),
        "no_staggered_closure": replace(base, staggered=False, stagger=0.0),
        "no_retry": replace(base, retry=False),
        "no_regrasp": replace(base, regrasp=False),
        "no_stable_dwell": replace(base, stabilize_dwell=1),
        "short_horizon": base,                                  # evaluated at horizon 40 (below)
        "scrambled_geometry": base,                             # evaluated with scramble_perm (below)
    }


def run() -> dict:
    t0 = time.perf_counter()
    env = make_acq_env()
    wall = _states()["acquisition_wall"]
    base = _best_params()
    _log(f"=== PART III acquisition component ablation (n={len(wall)}, best params from oracle) ===")
    abl = _ablations(base)
    rng = np.random.default_rng(0)
    results = {}
    for name, p in abl.items():
        if name == "short_horizon":
            ev = eval_acquisition(p, wall, env=env, horizon=40)
        elif name == "scrambled_geometry":
            ev = eval_acquisition(p, wall, env=env, scramble_perm=_SCRAMBLE)
        else:
            ev = eval_acquisition(p, wall, env=env)
        results[name] = {"n_stable": ev["n_stable"], "two_finger_rate": ev["two_finger_rate"],
                         "pregrasp_rate": ev["pregrasp_rate"], "recovered": ev["recovered_seeds"]}
        _log(f"  {name:26s} n_stable={ev['n_stable']:2d}/19 two_finger={ev['two_finger_rate']} pregrasp={ev['pregrasp_rate']}")
    # random-action + random-param controls (equal candidate count)
    rand_dir = _random_action_control(env, wall, rng)
    rand_par = _random_param_control(env, wall, base, rng, n=8)
    results["random_action_direction"] = rand_dir
    results["random_param_search"] = rand_par
    _log(f"  random_action_direction    n_stable={rand_dir['n_stable']}/19")
    _log(f"  random_param_search(x8)    n_stable={rand_par['n_stable']}/19")

    full_n = results["full_primitive"]["n_stable"]
    marginals = {name: full_n - results[name]["n_stable"] for name in abl if name != "full_primitive"}
    correct_beats_scrambled = full_n > results["scrambled_geometry"]["n_stable"]
    out = {"best_params": {k: (v.value if hasattr(v, "value") else v) for k, v in base.__dict__.items()},
           "full_n_stable": full_n, "ablations": results, "marginal_loss_per_component": marginals,
           "correct_vs_scrambled": {"correct": full_n, "scrambled": results["scrambled_geometry"]["n_stable"],
                                    "correct_beats_scrambled": correct_beats_scrambled},
           "load_bearing_components": sorted([k for k, v in marginals.items() if v > 0], key=lambda k: -marginals[k]),
           "wall_s": round(time.perf_counter() - t0, 1)}
    (_OUT / "manifests").mkdir(parents=True, exist_ok=True)
    (_OUT / "manifests" / "coin_delivery_acquisition_ablation.json").write_text(json.dumps(out, indent=2, default=str))
    _log(f"[ABLATION] full {full_n}/19 | correct>scrambled={correct_beats_scrambled} | "
         f"load-bearing={out['load_bearing_components']} | {out['wall_s']}s")
    return out


def _random_action_control(env, seeds, rng) -> dict:
    """Random 6-DoF actions (same horizon) — the non-structural control (does structure matter, or just motion?)."""
    n_stable = 0
    recovered = []
    for sd in seeds:
        obs, _i = env.reset(seed=int(sd))
        env._horizon = 128
        inner = env._env
        both_run = 0
        ok = False
        for _ in range(120):
            a = rng.uniform(-1, 1, 6).astype(np.float32)
            obs, _r, _t, _tr, info = env.step(a)
            m = inner._planar_metrics
            both_run = both_run + 1 if (m.left_contact and m.right_contact) else 0
            if both_run >= 6:
                ok = True
                break
            if info.get("safety_violation"):
                break
        n_stable += int(ok)
        if ok:
            recovered.append(int(sd))
    return {"n_stable": n_stable, "recovered": recovered}


def _random_param_control(env, seeds, base, rng, n: int = 8) -> dict:
    """Best of n random parameterisations of the SAME primitive (equal candidate budget, no geometry search bias)."""
    best = {"n_stable": -1, "recovered": []}
    for _ in range(n):
        p = AcqParams.from_unit(rng.uniform(0, 1, 8), base)
        ev = eval_acquisition(p, seeds, env=env)
        if ev["n_stable"] > best["n_stable"]:
            best = {"n_stable": ev["n_stable"], "recovered": ev["recovered_seeds"]}
    return best


if __name__ == "__main__":
    run()
    sys.exit(0)
