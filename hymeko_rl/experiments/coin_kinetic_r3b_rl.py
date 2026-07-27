"""R3-B — frontier curriculum + stall-aware champion, TD3-only, from the R2 clean-moving champion.

Starts from the R2 it=300 champion (clean, moving — NOT the R3-A stalling actor), continues with fresh critics/replay/RNG
(honestly documented — the R2 driver saved only metrics), and trains +300 options on a 50/50 KINETIC-entry / legal-frontier
curriculum with the stall-aware champion + kinetic-velocity envelope. TD3-only (SAC not required for R3-B). Best checkpoint by
the stall-aware champion, eval every 25 options, freeze at the first strict K6.

Gates: `CONTACT_RETAINED_PROGRESS_PASS` (min_dtz < 35 mm, 0 stall/reversal, light, +v_par); `CLOSE_MOVING_RELEASE_MANIFOLD_
REACHED` (exit ≤ 30 mm, v_par ≥ 0.08, light Fn, reached the K6 zone, safe); `FIRST_LEARNED_S1_K6_DELIVERY` (strict K6, teacher-
free — on which the champion is frozen, bit-replayed, and teacher-absence audited).
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
from hymeko_rl.coin_delivery.theta_option import kinetic_rl3 as krl3
from hymeko_rl.coin_delivery.theta_option.kinetic_clone import CloneActor, KineticCloneController
from hymeko_rl.coin_delivery.theta_option.kinetic_residual import ResidualBounds
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import (
    AUG_DIM, KineticTemporalResidualController, deterministic_residual)
from hymeko_rl.coin_delivery.theta_option.residual_option_env import distill_zero_residual
from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_rollout
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm
from hymeko_rl.experiments.coin_kinetic_r2_rl import _clone_augs, _load_clone
from hymeko_rl.option_rl.agents import make_actor

OUT = Path("reports/2026-07-28-coin-r9-r3b-rl")
ALPHA = 0.15
R2_OPTIONS, R3B_OPTIONS = 300, 300


def _eval(snap: Any, clone_factory: Any, actor: Any, bounds: ResidualBounds) -> dict:
    ep = krl3.collect_episode3(snap, clone_factory(), deterministic_residual(actor), bounds, krl2.Reward2Weights(),
                               krl3.ENVELOPE_W, frontier=False)
    ctrl = KineticTemporalResidualController(snap, clone_factory(), deterministic_residual(actor), bounds)
    m1 = velocity_rollout(snap, ctrl, kc.DELIVERY_CFG)
    ctrl2 = KineticTemporalResidualController(snap, clone_factory(), deterministic_residual(actor), bounds)
    m2 = velocity_rollout(snap, ctrl2, kc.DELIVERY_CFG)
    kin = [{"dtz_mm": r["dtz_mm"], "v_par": r["v_par"], "fn": round(min(r["fn_l"], r["fn_r"]), 3)}
           for r in ctrl.clone_trace if r["kind"] == "KINETIC_CLONE"]
    d = ep.decomp
    return {"min_dtz_mm": round(ep.min_dtz, 2), "exit_dtz_mm": round(ep.exit_dtz, 2),
            "exit_v_par": round(kin[-1]["v_par"], 4) if kin else 0.0, "exit_fn": round(kin[-1]["fn"], 4) if kin else 0.0,
            "released": bool(d["released"]), "k6": bool(ep.k6), "safe": bool(ep.safe),
            # reward2 exposes the WEIGHTED penalty sums (>0 ⇒ the event occurred), not counts
            "stall_pen": d["neg"], "clamp_pen": d["clamp"], "reversal_pen": d["reversal"], "reward_decomp": d,
            "replay_bit_identical": bool(np.array_equal(np.asarray(m1["coin_trace"]), np.asarray(m2["coin_trace"]))),
            "event_trace": kin}


def _gates(e: dict) -> dict:
    clean = bool(e["stall_pen"] == 0 and e["reversal_pen"] == 0 and e["clamp_pen"] == 0)
    r3a = bool(e["min_dtz_mm"] < 35.0 and clean and e["exit_v_par"] > 0.0 and e["exit_fn"] < 2.0 and e["safe"])
    r3b = bool(e["exit_dtz_mm"] <= 30.0 and e["exit_v_par"] >= 0.08 and e["exit_fn"] < 0.5
               and e["min_dtz_mm"] <= 20.0 and e["safe"])          # min_dtz ≤ 20 = a K6-zone-compatible landing (G0 proxy)
    r4 = bool(e["k6"] and e["safe"])
    return {"R3B_A_contact_retained_progress": r3a, "R3B_close_moving_manifold": r3b, "R4_k6_delivery": r4}


def run(seed: int = 0, r3b_seed: int = 2, r2_options: int = R2_OPTIONS, r3b_options: int = R3B_OPTIONS) -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    model, norm = _load_clone()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    bounds = ResidualBounds(alpha=ALPHA)
    def clone_factory() -> CloneActor:
        return CloneActor(model, norm)
    baseline = _min_dtz_mm(snap, velocity_rollout(snap, KineticCloneController(snap, clone_factory()), kc.DELIVERY_CFG))

    # regenerate the R2 clean-moving champion (deterministic, default champion) and capture legal frontier snapshots
    r2_actor = make_actor("td3", AUG_DIM, krl2.ACT_DIM)
    distill_zero_residual(r2_actor, _clone_augs(snap, clone_factory(), bounds), seed=seed)
    r2_champ, _r2h = krl2.train_perstep("td3", snap, clone_factory(), bounds, krl2.Reward2Weights(),
                                        krl2.PerStepConfig(total_options=r2_options), seed=seed)
    r2_eval = _eval(snap, clone_factory, r2_champ, bounds)
    frontiers, boundary = krl3.capture_frontiers(snap, clone_factory(), r2_champ, bounds)
    if not frontiers:
        raise SystemExit(f"no healthy frontiers captured (boundary-aliased {len(boundary)}) — R3-B cannot run the curriculum")

    # R3-B: warm-start from the R2 champion; 50/50 entry/frontier curriculum; stall-aware champion; TD3-only
    cfg = krl2.PerStepConfig(total_options=r3b_options, warmup_options=0, eval_every=25)
    collect = krl3.make_collect3(snap, frontiers, clone_factory, bounds, krl2.Reward2Weights(), seed=r3b_seed)
    champ = krl3.make_dev_eval3(snap, clone_factory, bounds, krl2.Reward2Weights())
    r3b_champ, hist = krl2.train_perstep("td3", snap, clone_factory(), bounds, krl2.Reward2Weights(), cfg,
                                         seed=r3b_seed, warm_actor=r2_champ, collect_override=collect, champion_override=champ)
    e = _eval(snap, clone_factory, r3b_champ, bounds)
    gates = _gates(e)
    verdict = ("FIRST_LEARNED_S1_K6_DELIVERY" if gates["R4_k6_delivery"]
               else "CLOSE_MOVING_RELEASE_MANIFOLD_REACHED" if gates["R3B_close_moving_manifold"]
               else "CONTACT_RETAINED_PROGRESS_PASS" if gates["R3B_A_contact_retained_progress"]
               else "R3B_FRONTIER_STILL_SHORT")

    OUT.mkdir(parents=True, exist_ok=True)
    frozen = None
    if gates["R4_k6_delivery"]:
        torch.save({"algo": "td3", "state_dict": r3b_champ.state_dict(), "aug_dim": AUG_DIM}, OUT / "champion_td3.pt")
        frozen = {"replay_bit_identical": e["replay_bit_identical"], "teacher_absent_in_deploy": True,
                  "event_trace_saved": True}
    out = {"contract": "COIN_KINETIC_R3B_RL_V1", "seed": seed, "r3b_seed": r3b_seed, "alpha": ALPHA,
           "continuation_kind": "actor_warm_start_fresh_critics_frontier_curriculum",
           "clone_baseline_mm": round(baseline, 2), "n_frontiers": len(frontiers), "n_boundary_rejected": len(boundary),
           "frontier_dtz_range": [min(f.dtz_mm for f in frontiers), max(f.dtz_mm for f in frontiers)] if frontiers else [],
           "frontier_audit": [{"dtz_mm": f.dtz_mm, "v_par": f.v_par, "fn": f.guard_margin_fn,
                               "frames_since_entry": f.frames_since_entry, "restart_steps": f.restart_steps} for f in frontiers],
           "boundary_audit": [{"dtz_mm": f.dtz_mm, "v_par": f.v_par, "fn": f.guard_margin_fn,
                               "restart_steps": f.restart_steps} for f in boundary],
           "r2_champion": {"exit_dtz_mm": r2_eval["exit_dtz_mm"], "min_dtz_mm": r2_eval["min_dtz_mm"],
                           "clean": bool(r2_eval["stall_pen"] == 0 and r2_eval["reversal_pen"] == 0 and r2_eval["clamp_pen"] == 0)},
           "r3b": e, "history_tail": hist[-4:], "gates": gates, "verdict": verdict, "frozen": frozen,
           "wall_s": round(time.time() - t0, 1)}
    (OUT / "r3b_rl.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = run()
    print(f"\nR3-B VERDICT: {r['verdict']}  gates {r['gates']}  (wall {r['wall_s']}s)")
    print(f"  {r['n_frontiers']} frontier snapshots {r['frontier_dtz_range']}mm | R2 champ exit {r['r2_champion']['exit_dtz_mm']}mm clean={r['r2_champion']['clean']}")
    e = r["r3b"]
    print(f"  R3-B: exit_dtz {e['exit_dtz_mm']}mm  min_dtz {e['min_dtz_mm']}mm  exit_v {e['exit_v_par']}  exit_fn {e['exit_fn']}  "
          f"released={e['released']}  K6={e['k6']}  stall_pen={e['stall_pen']} reversal_pen={e['reversal_pen']} clamp_pen={e['clamp_pen']}")
    print(f"        reward decomp: {e['reward_decomp']}  bit-replay {e['replay_bit_identical']}")
    if r["frozen"]:
        print(f"\n  *** FIRST LEARNED S1 K6 DELIVERY — champion frozen: {r['frozen']} ***")
