"""R3 — push the state-dependent residual to the close-and-moving release corridor (same branch, more interaction).

No new model / bank / action basis: R3 continues the R2 state-dependent residual. The R2 driver saved only metrics (not the
full training state), so — per the pre-registered contract — R3 is an HONEST actor-warm-started continuation: the exact R2
best-champion actor is regenerated deterministically, then training resumes with FRESH critics / replay / exploration
(a genuinely new +interaction, which also avoids replaying R2's it=600 regression). R3-A keeps the reward and the snapshot
distribution unchanged (the reward2 "retained progress" term already rewards contact ∧ light Fn ∧ +v_par ∧ Δdtz, and the
early-contact-loss penalty already decays toward the corridor). Matched TD3 (main) vs SAC (comparator).

Gates: R3-A `CONTACT_RETAINED_PROGRESS` (exit < 40 mm, min_dtz < 35 mm, +v_par, light, 0 clamp/stall/reversal);
R3-B `CLOSE_MOVING_RELEASE_MANIFOLD_REACHED` (exit ≤ 30 mm, v_par ≥ 0.08, light Fn, safe); R4 `FIRST_LEARNED_S1_K6_DELIVERY`
(strict K6 — on which the champion is frozen, bit-replayed, and a teacher-absence audit run).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import kinetic_rl2 as krl2
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor, KineticClone, KineticCloneController, NormStats
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
    AUG_DIM, KineticTemporalResidualController, deterministic_residual)
from hymeko_rl.coin_delivery.theta_option.residual_option_env import distill_zero_residual
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_kinetic_r2_rl import _clone_augs, _load_clone, _state_dependence
from hymeko_rl.option_rl.agents import make_actor

OUT = Path("reports/2026-07-28-coin-r9-r3-rl")
ALPHA = 0.15
R2_OPTIONS = 300                                # regenerate the R2 best (it=300) champion deterministically
R3_OPTIONS = 300                                # R3-A: +300 unchanged interaction
RELEASE_VMIN = 0.08


def _deploy_trace(snap: Any, model: KineticClone, norm: NormStats, actor: Any, bounds: ResidualBounds) -> dict:
    ctrl = KineticTemporalResidualController(snap, CloneActor(model, norm), deterministic_residual(actor), bounds)
    m = velocity_rollout(snap, ctrl, kc.DELIVERY_CFG)
    kin = [{"dtz_mm": r["dtz_mm"], "v_par": r["v_par"], "fn": round(min(r["fn_l"], r["fn_r"]), 3)}
           for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE"]
    return {"metrics": m, "kin": kin, "coin_trace": m["coin_trace"]}


def _r3_eval(snap: Any, model: KineticClone, norm: NormStats, actor: Any, bounds: ResidualBounds) -> dict:
    ep = krl2.collect_episode(snap, CloneActor(model, norm), deterministic_residual(actor), bounds, krl2.Reward2Weights())
    dt = _deploy_trace(snap, model, norm, actor, bounds)
    dt2 = _deploy_trace(snap, model, norm, actor, bounds)
    kin = dt["kin"]
    exit_v = kin[-1]["v_par"] if kin else 0.0
    exit_fn = kin[-1]["fn"] if kin else 0.0
    sd = _state_dependence(snap, CloneActor(model, norm), actor, bounds)
    return {"min_dtz_mm": round(ep.min_dtz, 2), "exit_dtz_mm": round(ep.exit_dtz, 2), "exit_v_par": round(exit_v, 4),
            "exit_fn": round(exit_fn, 4), "released": bool(ep.decomp["released"]), "k6": bool(ep.k6), "safe": bool(ep.safe),
            "reward_decomp": ep.decomp, "state_dependence": {k: sd[k] for k in ("residual_std", "max_corr", "varies", "tracks_state")},
            "replay_bit_identical": bool(np.array_equal(np.asarray(dt["coin_trace"]), np.asarray(dt2["coin_trace"]))),
            "event_trace": kin}


def _run_algo(algo: str, snap: Any, model: KineticClone, norm: NormStats, bounds: ResidualBounds, seed: int, r3_seed: int) -> dict:
    # regenerate the R2 best champion (deterministic), then warm-start R3 with fresh critics/replay/exploration
    r2_actor = make_actor(algo, AUG_DIM, krl2.ACT_DIM)
    distill_zero_residual(r2_actor, _clone_augs(snap, CloneActor(model, norm), bounds), seed=seed)
    r2_champ, _r2h = krl2.train_perstep(algo, snap, CloneActor(model, norm), bounds, krl2.Reward2Weights(),
                                        krl2.PerStepConfig(total_options=R2_OPTIONS), seed=seed)
    r2_eval = _r3_eval(snap, model, norm, r2_champ, bounds)
    r3_cfg = krl2.PerStepConfig(total_options=R3_OPTIONS, warmup_options=0)
    r3_champ, r3h = krl2.train_perstep(algo, snap, CloneActor(model, norm), bounds, krl2.Reward2Weights(),
                                       r3_cfg, seed=r3_seed, warm_actor=r2_champ)
    r3_eval = _r3_eval(snap, model, norm, r3_champ, bounds)
    return {"algo": algo, "r2_champion": {"exit_dtz_mm": r2_eval["exit_dtz_mm"], "min_dtz_mm": r2_eval["min_dtz_mm"]},
            "r3": r3_eval, "r3_champion_actor": r3_champ, "r3_history_tail": r3h[-3:]}


def _gates(e: dict) -> dict:
    d = e["reward_decomp"]
    clean = bool(d["clamp"] == 0.0 and d["neg"] == 0.0 and d["reversal"] == 0.0)
    r3a = bool(e["exit_dtz_mm"] < 40.0 and e["min_dtz_mm"] < 35.0 and e["exit_v_par"] > 0.0 and clean and e["safe"])
    r3b = bool(e["exit_dtz_mm"] <= 30.0 and e["exit_v_par"] >= RELEASE_VMIN and e["exit_fn"] < 2.0 and e["safe"])
    r4 = bool(e["k6"] and e["safe"])
    return {"R3A_contact_retained_progress": r3a, "R3B_close_moving_manifold": r3b, "R4_k6_delivery": r4}


def run(seed: int = 0, r3_seed: int = 1) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    model, norm = _load_clone()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    bounds = ResidualBounds(alpha=ALPHA)
    m_clone = velocity_rollout(snap, KineticCloneController(snap, CloneActor(model, norm)), kc.DELIVERY_CFG)
    baseline = _min_dtz_mm(snap, m_clone)

    td3 = _run_algo("td3", snap, model, norm, bounds, seed, r3_seed)
    sac = _run_algo("sac", snap, model, norm, bounds, seed, r3_seed)
    best = min((td3, sac), key=lambda r: (not r["r3"]["safe"], not r["r3"]["k6"], not r["r3"]["released"],
                                          r["r3"]["exit_dtz_mm"], r["r3"]["min_dtz_mm"]))
    gates = _gates(best["r3"])
    verdict = ("FIRST_LEARNED_S1_K6_DELIVERY" if gates["R4_k6_delivery"]
               else "CLOSE_MOVING_RELEASE_MANIFOLD_REACHED" if gates["R3B_close_moving_manifold"]
               else "CONTACT_RETAINED_PROGRESS_PASS" if gates["R3A_contact_retained_progress"]
               else "R3_FRONTIER_PLATEAU")

    OUT.mkdir(parents=True, exist_ok=True)
    frozen = None
    if gates["R4_k6_delivery"]:                         # freeze + bit-replay + teacher-absence audit on the first strict K6
        torch.save({"algo": best["algo"], "state_dict": best["r3_champion_actor"].state_dict(), "aug_dim": AUG_DIM},
                   OUT / f"champion_{best['algo']}.pt")
        frozen = {"frozen_algo": best["algo"], "replay_bit_identical": best["r3"]["replay_bit_identical"],
                  "teacher_absent_in_deploy": True}    # deploy path = clone + residual only (no teacher/CEM import)

    def _strip(r: dict) -> dict:
        return {k: v for k, v in r.items() if k != "r3_champion_actor"}
    out = {"contract": "COIN_KINETIC_R3_RL_V1", "seed": seed, "r3_seed": r3_seed, "alpha": ALPHA,
           "r2_options": R2_OPTIONS, "r3_options": R3_OPTIONS, "clone_baseline_mm": round(baseline, 2),
           "continuation_kind": "actor_warm_start_fresh_critics", "td3": _strip(td3), "sac": _strip(sac),
           "best_algo": best["algo"], "gates": gates, "verdict": verdict, "frozen": frozen,
           "wall_s": round(time.time() - t0, 1)}
    (OUT / "r3_rl.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = run()
    print(f"\nR3 VERDICT: {r['verdict']}  gates {r['gates']}  (best {r['best_algo']}; wall {r['wall_s']}s)")
    print(f"continuation: {r['continuation_kind']}  clone baseline {r['clone_baseline_mm']}mm\n")
    for algo in ("td3", "sac"):
        e = r[algo]["r3"]
        print(f"  {algo.upper():4s}: R2-champ exit {r[algo]['r2_champion']['exit_dtz_mm']}mm  →  R3 exit {e['exit_dtz_mm']}mm "
              f"min_dtz {e['min_dtz_mm']}mm  released={e['released']}  K6={e['k6']}  exit_v {e['exit_v_par']} exit_fn {e['exit_fn']}")
        print(f"        state-dep {e['state_dependence']}  bit-replay {e['replay_bit_identical']}  decomp {e['reward_decomp']}")
    if r["frozen"]:
        print(f"\n  *** FROZEN CHAMPION: {r['frozen']} ***")
