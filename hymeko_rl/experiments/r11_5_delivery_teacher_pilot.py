"""R11.5 pilot — target-conditioned delivery teacher vs frozen R2 on 12 certified-grasp DELIVERY_FAILURE states.

For each pilot scenario: regenerate a certified bilateral grasp (grasp-aware capture), then run BOTH arms on the SAME
snap — frozen R2 (baseline, ``characterize_delivery``) and the target-conditioned teacher (``solve_delivery`` with the
full-transport spec, up to 5 restarts, early-exit on K6). The teacher re-optimizes PUSH/BRAKE/RELEASE for the actual
coin→target geometry, replacing the frozen R2 downstream. Energy is measured on the winner only (diagnostic, never in the
objective — R11.4A contract).

Metric panel per scenario: strict K6, final dtz, coin progress, target-entry / peak coin speed, contact-loss, W±, peak
coin KE, safety. Pre-registered pilot gate: >=6/12 recovered to strict K6, >=1 dev, >=1 test, no safety regression, all
energy ledgers complete. Deterministic (fixed seeds). CPU-bound MuJoCo; parallelizable via --offset/--limit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from hymeko_rl.coin_delivery.delivery_teacher.regrasp_characterize import characterize_delivery
from hymeko_rl.coin_delivery.delivery_teacher.solver import (
    full_transport_spec,
    solve_delivery,
    solve_delivery_with_settle,
)
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective, is_certified_grasp
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

DEFAULT_OUT = Path("reports/2026-07-30-r11-5-delivery-pilot/pilot.jsonl")
RESTARTS = 5
PILOTS = [
    ("bank_c0_0", "train"), ("bank_c3_r6_a+45", "train"), ("bank_c1_+0.01_-0.02", "train"),
    ("bank_c2_+0.015_+0.015", "train"),
    ("bank_c0_1", "dev"), ("bank_c1_+0.03_-0.02", "dev"), ("bank_c3_r9_a-45", "dev"), ("bank_c3_r9_a-15", "dev"),
    ("bank_c1_+0.03_+0.00", "test"), ("bank_c2_+0.025_+0.000", "test"), ("bank_c3_r9_a+30", "test"),
    ("bank_c3_r9_a+45", "test"),
]


def _certified_grasp_snap(rig, cfg, conf, obj, scen, capture_seeds=3, seed_offset=0):
    """Regenerate a certified bilateral grasp for the scenario (the shared A/B input); None if none within the budget.
    ``capture_seeds`` sets how many capture regens to try (a larger budget hedges the handoff-quality grasp variance);
    ``seed_offset`` samples a different realization (for the stability re-gate)."""
    for seed in range(seed_offset, seed_offset + capture_seeds):
        home, coin = Z._home_with_coin(rig, scen.coin_xy)
        _r, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg, conf, seed)
        if rc is not None and is_certified_grasp(rc.result.outcome, obj):
            return rc.result.outcome.snapshot, seed
    return None, -1


def _teacher_best(snap, solve_fn, restarts=RESTARTS, seed_offset=0):
    """Up to ``restarts`` restarts on the SAME snap; keep best by (k6, -min_dtz), early-exit on K6. A larger bounded
    restart budget converts CEM-search variance (fixed-grasp K6-rate p) into recovery: P(>=1 K6) = 1-(1-p)^restarts.
    ``seed_offset`` shifts the CEM seeds to sample a different realization (stability re-gate)."""
    best = None
    for s in range(seed_offset, seed_offset + restarts):
        res = solve_fn(snap, s)
        if best is None or (res.k6, -res.min_dtz_mm) > (best.k6, -best.min_dtz_mm):
            best = res
        if res.k6:
            break
    return best


def run_one(rig, cfg, conf, obj, solve_fn, sid: str, split: str, restarts=RESTARTS, capture_seeds=3,
            seed_offset=0) -> dict:
    scen = next(s for s in build_bank_scenarios() if s.scenario_id == sid)
    snap, cap_seed = _certified_grasp_snap(rig, cfg, conf, obj, scen, capture_seeds, seed_offset)
    if snap is None:
        return {"scenario_id": sid, "split": split, "certified": False, "recovered": False, "note": "no certified grasp"}
    base = characterize_delivery(snap, rig["down"])
    t = _teacher_best(snap, solve_fn, restarts, seed_offset)
    m, e = t.measurements, t.energy
    return {
        "scenario_id": sid, "split": split, "certified": True, "capture_seed": cap_seed,
        "base_r2_k6": bool(base.k6), "base_r2_kinetic": bool(base.reaches_kinetic),
        "base_r2_dtz_mm": round(float(base.min_dtz_mm), 2),
        "teacher_k6": bool(t.k6), "teacher_dtz_mm": t.min_dtz_mm, "teacher_safe": bool(t.safe),
        "recovered": bool(t.k6 and not base.k6), "theta": list(t.theta),
        "coin_progress_mm": round(float(m.get("forward", 0.0)) * 1000, 1),
        "gap_closed": round(float(m.get("gap_closed", 0.0)), 3),
        "target_entry_speed": round(float(m.get("terminal_speed", 0.0)), 3),
        "peak_coin_speed": round(float(m.get("peak_coin_speed", 0.0)), 3),
        "contact_lost_steps": int(m.get("contact_lost_steps", m.get("contact_lost", 0))),
        "w_pos": round(float(e.w_actuator_pos), 4), "w_neg": round(float(e.w_actuator_neg), 4),
        "peak_coin_ke": round(float(e.peak_coin_ke), 6), "energy_complete": bool(e.is_complete()),
    }


def _pilot_passed(n_rec: int, rec_split: dict, safety_ok: bool, energy_ok: bool) -> bool:
    """Pre-registered gate: >=6/12 recovered, >=1 dev, >=1 test, no safety regression, energy ledgers complete."""
    return n_rec >= 6 and rec_split["dev"] >= 1 and rec_split["test"] >= 1 and safety_ok and energy_ok


def gate(rows: list) -> dict:
    done = [r for r in rows if r.get("certified")]
    rec = [r for r in done if r.get("recovered")]
    rec_split = {sp: sum(r["split"] == sp for r in rec) for sp in ("train", "dev", "test")}
    safety_ok = all(r.get("teacher_safe", True) for r in done)
    energy_ok = all(r.get("energy_complete", False) for r in done)
    passed = _pilot_passed(len(rec), rec_split, safety_ok, energy_ok)
    return {"n": len(rows), "certified": len(done), "recovered": len(rec), "recovered_by_split": rec_split,
            "safety_ok": safety_ok, "energy_complete_all": energy_ok,
            "verdict": ("R11_5_TARGET_CONDITIONED_DELIVERY_TEACHER_PILOT_PASS" if passed
                        else "R11_5_PIPELINE_PASS_DELIVERY_TEACHER_RECOVERY_INSUFFICIENT")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 12 pilots")
    ap.add_argument("--settle", action="store_true", help="use the settle-coordinate spec (transport + settle_gain)")
    ap.add_argument("--restarts", type=int, default=RESTARTS, help="delivery CEM restarts per scenario (bounded budget R)")
    ap.add_argument("--capture-seeds", type=int, default=3, help="certified-grasp regen attempts (hedge handoff variance)")
    ap.add_argument("--seed-offset", type=int, default=0, help="shift capture+CEM seeds for a different realization (stability re-gate)")
    args = ap.parse_args()
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=3, grasp_objective=obj)
    tspec = full_transport_spec()
    solve_fn = ((lambda snap, s: solve_delivery_with_settle(snap, s)) if args.settle
                else (lambda snap, s: solve_delivery(snap, s, tspec)))
    pilots = PILOTS[args.offset:(args.offset + args.limit if args.limit else len(PILOTS))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w", encoding="utf-8") as fh:
        for i, (sid, split) in enumerate(pilots, 1):
            t0 = time.perf_counter()
            r = run_one(rig, cfg, conf, obj, solve_fn, sid, split, args.restarts, args.capture_seeds, args.seed_offset)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[{i}/{len(pilots)}] {sid:24s} {split:5s} base_r2_k6={r.get('base_r2_k6')} "
                  f"teacher_k6={r.get('teacher_k6')} dtz={r.get('teacher_dtz_mm')} recovered={r.get('recovered')} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print("=== R11.5 PILOT GATE ===", flush=True)
    print(json.dumps(gate(rows), indent=2), flush=True)
    print("R11_5_PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
