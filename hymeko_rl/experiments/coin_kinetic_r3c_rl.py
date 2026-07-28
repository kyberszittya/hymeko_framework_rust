"""R3-C — action-preserving authority-unlock RL campaign (distributed TD3), toward the first teacher-free strict learned s1 K6.

Architecture (see `kinetic_authority_unlock`): frozen K2 clone + frozen R2 residual (α = 0.15) + a zero-initialised state-dependent
expansion head β·tanh δ_expand on the SAME 4-D basis, `u = clip(u_clone + 0.15·a_R2 + β·δ_expand, −1, 1)`. Two families — C1
β = 0.85 (total ≈ 1.0), C2 β = 1.85 (total ≈ 2.0). Per seed: Phase A (600 options, 80 % KINETIC-entry / 20 % healthy frontier,
eval every 25, best checkpoint), then Phase B (another 600) only for clean-advancing seeds. TD3 only. Exploration is measured in
applied-action units (β-scaled). Champion / freeze order: strict K6 ≻ safety ≻ 0 clamp/stall/reversal ≻ K6 dwell ≻ min_dtz ≻ clean
release. The run freezes IMMEDIATELY on the first strict learned K6.

Run (smoke):  ``python -m hymeko_rl.experiments.coin_kinetic_r3c_rl --smoke``
Run (full):   ``python -m hymeko_rl.experiments.coin_kinetic_r3c_rl``
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable


from hymeko_rl.coin_delivery.theta_option import kinetic_authority_unlock as ku
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import AUG_DIM, deterministic_residual
from hymeko_rl.coin_delivery.theta_option.residual_option_env import distill_zero_residual
from hymeko_rl.experiments.coin_kinetic_r2_rl import _clone_augs, _load_clone
from hymeko_rl.option_rl.agents import make_actor

OUT = Path("reports/2026-07-28-coin-r9-r3c-rl")
FAMILIES = {"C1": ku.C1_BETA, "C2": ku.C2_BETA}
R2_OPTIONS = 300                                 # the frozen R2 champion regen budget (deterministic; matches the audit)
R2_BASELINE_MIN_DTZ = 39.58                      # the R2 champion's clean min_dtz — a seed must beat this to earn Phase B
BASE_CFG = krl2.PerStepConfig(warmup_options=20, eval_every=25, alpha=0.1)


def _regen_r2_champion(snap: Any, model: Any, norm: Any, bounds: ResidualBounds, *, seed: int, options: int) -> Any:
    """Deterministically reproduce the frozen R2 champion actor (same path as the audit/torque-span)."""
    def clone() -> CloneActor:
        return CloneActor(model, norm)
    a = make_actor("td3", AUG_DIM, krl2.ACT_DIM)
    distill_zero_residual(a, _clone_augs(snap, clone(), bounds), seed=seed)
    champ, _ = krl2.train_perstep("td3", snap, clone(), bounds, krl2.Reward2Weights(),
                                  krl2.PerStepConfig(total_options=options), seed=seed)
    return champ


def _train_phase(entry_snap: Any, frontiers: list, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds,
                 w: krl2.Reward2Weights, r2_fn: Callable, beta: float, *, warm_actor: Any, options: int, seed: int,
                 log: Callable) -> "tuple[Any, list]":
    """One TD3 phase for a (family, seed): 80/20 curriculum through the unlock controller, β-scaled exploration, freeze on the
    first strict K6. Returns (best_expansion_actor, history)."""
    cfg = replace(BASE_CFG, total_options=options, expl_noise=ku.expl_noise_for(beta))
    collect = ku.make_collect_unlock(entry_snap, frontiers, clone_factory, bounds, w, r2_fn, beta, seed=seed)
    champ = ku.make_dev_eval_unlock(entry_snap, clone_factory, bounds, w, r2_fn, beta)
    return krl2.train_perstep("td3", entry_snap, clone_factory(), bounds, w, cfg, seed=seed, warm_actor=warm_actor,
                              log=log, collect_override=collect, champion_override=champ, stop_when=ku.stop_on_strict_k6)


def _eval_actor(entry_snap: Any, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds, w: krl2.Reward2Weights,
                r2_fn: Callable, beta: float, actor: Any) -> dict:
    return ku.make_dev_eval_unlock(entry_snap, clone_factory, bounds, w, r2_fn, beta)(actor)[1]


def run_seed(entry_snap: Any, frontiers: list, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds,
             w: krl2.Reward2Weights, r2_fn: Callable, family: str, beta: float, *, seed: int, phase_a: int, phase_b: int,
             log: Callable = print) -> dict:
    """Run one (family, seed): zero-init expansion actor → Phase A → (clean-advancing) Phase B. Stops at the first strict K6."""
    warm = ku.zero_init_detactor(make_actor("td3", AUG_DIM, krl2.ACT_DIM))
    best_a, hist_a = _train_phase(entry_snap, frontiers, clone_factory, bounds, w, r2_fn, beta,
                                  warm_actor=warm, options=phase_a, seed=seed, log=log)
    aux = _eval_actor(entry_snap, clone_factory, bounds, w, r2_fn, beta, best_a)
    res = {"family": family, "beta": beta, "seed": seed, "phase_a_aux": aux, "phase": "A"}
    advancing = bool(aux["clean"] and aux["min_dtz"] < R2_BASELINE_MIN_DTZ)
    if not aux["k6_strict"] and phase_b > 0 and advancing:
        best_b, hist_b = _train_phase(entry_snap, frontiers, clone_factory, bounds, w, r2_fn, beta,
                                      warm_actor=copy.deepcopy(best_a), options=phase_b, seed=seed + 1000, log=log)
        aux_b = _eval_actor(entry_snap, clone_factory, bounds, w, r2_fn, beta, best_b)
        # keep whichever phase's champion is better by the R3-C key
        if ku.champion_key_unlock(aux_b["k6"], aux_b["safe"], aux_b["clean"], aux_b["min_dtz"], aux_b["released"]) > \
           ku.champion_key_unlock(aux["k6"], aux["safe"], aux["clean"], aux["min_dtz"], aux["released"]):
            best_a, aux = best_b, aux_b
        res.update({"phase": "B", "phase_b_aux": aux_b, "advancing": True})
    res.update({"final_aux": aux, "k6_strict": bool(aux["k6_strict"]), "min_dtz": aux["min_dtz"], "actor": best_a})
    return res


def _freeze(res: dict, entry_snap: Any, clone_factory: Callable[[], CloneActor], bounds: ResidualBounds,
            w: krl2.Reward2Weights, r2_champ: Any, out: Path, log: Callable) -> dict:
    """First strict learned K6: deterministic-replay check, teacher-absence assertion, checkpoint + trace."""
    beta, actor = res["beta"], res["actor"]
    factory = ku.unlock_controller_factory(deterministic_residual(r2_champ), beta)
    ctrl = factory(entry_snap, clone_factory(), deterministic_residual(actor), bounds)
    assert not hasattr(ctrl, "teacher") and ctrl.r2_fn is not None                     # teacher-absent: only clone + learned heads
    a1 = _eval_actor(entry_snap, clone_factory, bounds, w, deterministic_residual(r2_champ), beta, actor)
    a2 = _eval_actor(entry_snap, clone_factory, bounds, w, deterministic_residual(r2_champ), beta, actor)
    assert a1 == a2, "non-deterministic replay"                                        # deterministic replay
    ckpt = {"expansion_state": {k: v.tolist() for k, v in actor.state_dict().items()},
            "r2_champ_state": {k: v.tolist() for k, v in r2_champ.state_dict().items()},
            "beta": beta, "alpha0": ku.ALPHA0, "family": res["family"], "seed": res["seed"], "aux": res["final_aux"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "first_k6_checkpoint.json").write_text(json.dumps(ckpt))
    log(f"  ✅ FROZEN first learned strict K6: {res['family']} seed {res['seed']} β {beta} min_dtz {res['min_dtz']}")
    return {"frozen": True, "replay_deterministic": True, "teacher_absent": True, **{k: res[k] for k in
            ("family", "beta", "seed", "min_dtz", "final_aux")}}


@dataclass
class _Campaign:
    """The frozen, shared campaign context regenerated deterministically for every seed."""
    entry_snap: Any
    frontiers: list
    cf: Callable[[], CloneActor]
    bounds: ResidualBounds
    w: krl2.Reward2Weights
    r2_fn: Callable
    r2_champ: Any


def _setup() -> _Campaign:
    """Acquire s1, freeze the KINETIC entry, regenerate the frozen R2 champion + its healthy frontiers (deterministic)."""
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    model, norm = _load_clone()
    snap, meta = acquire_snapshot(load_harness(), kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    entry = kc.freeze_kinetic_entry(snap, seed=kc.S1_SEED)
    bounds = ResidualBounds(alpha=ku.ALPHA0)

    def cf() -> CloneActor:
        return CloneActor(model, norm)
    r2_champ = _regen_r2_champion(snap, model, norm, bounds, seed=0, options=R2_OPTIONS)
    frontiers, _boundary = krl3.capture_frontiers(snap, cf(), r2_champ, bounds)
    return _Campaign(entry.tsnap, frontiers, cf, bounds, krl2.Reward2Weights(), deterministic_residual(r2_champ), r2_champ)


def _run_panel(camp: _Campaign, families: dict, seeds: list, *, phase_a: int, phase_b: int,
               out: Path) -> "tuple[list[dict], dict | None]":
    """Sweep (family × seed) until the first strict K6, then freeze immediately. Returns (results, frozen-or-None)."""
    results: list[dict] = []
    for fam, beta in families.items():
        for sd in seeds:
            res = run_seed(camp.entry_snap, camp.frontiers, camp.cf, camp.bounds, camp.w, camp.r2_fn, fam, beta,
                           seed=sd, phase_a=phase_a, phase_b=phase_b)
            results.append({k: v for k, v in res.items() if k != "actor"})
            print(f"  [{fam} seed {sd} β{beta}] min_dtz {res['min_dtz']}mm k6_strict {res['k6_strict']} "
                  f"clean {res['final_aux']['clean']} safe {res['final_aux']['safe']}")
            if res["k6_strict"]:
                return results, _freeze(res, camp.entry_snap, camp.cf, camp.bounds, camp.w, camp.r2_champ, out, log=print)
    return results, None


def _panel_config(smoke: bool, seeds_per_family: int, phase_a: int, phase_b: int) -> "tuple[dict, list, int, int]":
    """(families, seeds, phase_a, phase_b) — the smoke is one C1 seed, Phase A only; the full run is both families × N seeds."""
    if smoke:
        return {"C1": ku.C1_BETA}, [0], phase_a, 0
    return FAMILIES, list(range(seeds_per_family)), phase_a, phase_b


def run(*, smoke: bool = False, seeds_per_family: int = 14, phase_a: int = 600, phase_b: int = 600, out: Path = OUT) -> dict:
    t0 = time.time()
    camp = _setup()
    families, seeds, pa, pb = _panel_config(smoke, seeds_per_family, phase_a, phase_b)
    results, frozen = _run_panel(camp, families, seeds, phase_a=pa, phase_b=pb, out=out)
    best = min(results, key=lambda r: r["min_dtz"]) if results else None
    verdict = "FIRST_LEARNED_S1_K6_DELIVERY" if frozen else ("R3C_SMOKE_COMPLETE" if smoke else "R3C_PANEL_EXHAUSTED_NO_K6")
    out.mkdir(parents=True, exist_ok=True)
    summary = {"contract": "COIN_KINETIC_R3C_AUTHORITY_UNLOCK_V1", "smoke": smoke,
               "frontiers_mm": [f.dtz_mm for f in camp.frontiers], "families": dict(families),
               "seeds": seeds, "phase_a": pa, "phase_b": pb, "results": results,
               "best": ({k: best[k] for k in ("family", "seed", "beta", "min_dtz", "final_aux")} if best else None),
               "frozen": frozen, "verdict": verdict, "wall_s": round(time.time() - t0, 1)}
    (out / ("smoke.json" if smoke else "r3c_rl.json")).write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    r = run(smoke=smoke)
    print(f"\nR3-C {'SMOKE ' if smoke else ''}VERDICT: {r['verdict']}  (wall {r['wall_s']}s)")
    if r["best"]:
        b = r["best"]
        print(f"  best: {b['family']} seed {b['seed']} β{b['beta']} min_dtz {b['min_dtz']}mm  "
              f"k6_strict {b['final_aux']['k6_strict']} clean {b['final_aux']['clean']}")
    print(f"  frontiers {r['frontiers_mm']}mm")
