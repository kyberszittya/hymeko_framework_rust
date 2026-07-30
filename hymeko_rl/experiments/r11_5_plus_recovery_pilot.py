"""Phase 5 — pre-registered 3-arm residual-recovery pilot over the taxonomy's bounded 12 (4 capture-support, 4 negative-x
CONTACT_LOSS, 4 INSUFFICIENT_TRANSPORT_PROGRESS). Each scenario, on the SAME certified grasp, runs three arms:
frozen R2 (``characterize_delivery``), single-stage (``solve_delivery``), and two-stage (``solve_delivery_two_stage``,
TARGET_RELATIVE_ALIGNMENT_PHASE), each with a bounded restart budget. Isolates whether the two-stage ALIGN adds recovery
beyond the single-stage push. Energy diagnostic-only; every attempt retained.

Gate: >= 6/12 recovered by the improved teacher (single OR two-stage k6 where R2 is not), >= 2 negative-x recovered when
>= 2 are available, 0 safety regressions, 0 nudge-only K6 (every input is a certified grasp), 100% energy/provenance.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from hymeko_rl.coin_delivery.delivery_teacher.regrasp_characterize import characterize_delivery
from hymeko_rl.coin_delivery.delivery_teacher.solver import (
    AlignSearchSpec,
    DeliveryResult,
    full_transport_spec,
    solve_delivery,
    solve_delivery_two_stage,
)
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_5_delivery_teacher_pilot import _certified_grasp_snap

TAXONOMY = Path("reports/2026-07-31-r11-5-plus-residual-taxonomy.json")
DEFAULT_OUT = Path("reports/2026-07-31-r11-5-plus-recovery-pilot.json")
NEG_X = "CONTACT_LOSS_DURING_DELIVERY"


def _best(solve: Any, snap: Any, restarts: int) -> DeliveryResult:
    """Best of ``restarts`` seeds by (k6, -min_dtz); early-exit on the first K6."""
    best: DeliveryResult | None = None
    for s in range(restarts):
        r = solve(snap, s)
        if best is None or (r.k6, -r.min_dtz_mm) > (best.k6, -best.min_dtz_mm):
            best = r
        if r.k6:
            break
    assert best is not None
    return best


def run_scenario(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, sid: str, category: str,
                 split: str, restarts: int) -> "dict[str, Any]":
    scen = next(s for s in build_bank_scenarios() if s.scenario_id == sid)
    snap, seed = _certified_grasp_snap(rig, cfg, conf, obj, scen, capture_seeds=5)  # type: ignore[no-untyped-call]  # untyped MuJoCo glue
    if snap is None:
        return {"scenario_id": sid, "category": category, "split": split, "certified": False, "recovered": False,
                "note": "no certified grasp (capture-support honest-negative)"}
    r2 = characterize_delivery(snap, rig["down"])
    tspec = full_transport_spec()
    single = _best(lambda sn, s: solve_delivery(sn, s, tspec), snap, restarts)
    two = _best(lambda sn, s: solve_delivery_two_stage(sn, s, AlignSearchSpec()), snap, restarts)
    best_teacher = two if (two.k6, -two.min_dtz_mm) >= (single.k6, -single.min_dtz_mm) else single
    e = best_teacher.energy
    return {"scenario_id": sid, "category": category, "split": split, "certified": True, "capture_seed": seed,
            "r2_k6": bool(r2.k6), "r2_dtz_mm": round(float(r2.min_dtz_mm), 2),
            "single_k6": bool(single.k6), "single_dtz_mm": single.min_dtz_mm,
            "two_stage_k6": bool(two.k6), "two_stage_dtz_mm": two.min_dtz_mm,
            "two_stage_stage": two.stage_used, "align_verdict": two.align_verdict,
            "recovered": bool(best_teacher.k6 and not r2.k6),
            "two_stage_adds": bool(two.k6 and not single.k6),
            "safe": bool(best_teacher.safe), "energy_complete": bool(e.is_complete())}


def _tally(rows: "list[dict[str, Any]]") -> "tuple[int, int, int, int, bool, bool, int]":
    """(#certified, #recovered, negx_available, negx_recovered, safety_ok, energy_ok, two_stage_adds)."""
    done = [r for r in rows if r.get("certified")]
    rec = [r for r in done if r.get("recovered")]
    negx_avail = sum(1 for r in rows if r.get("category") == NEG_X)
    negx_rec = sum(1 for r in rec if r.get("category") == NEG_X)
    safety_ok = all(r.get("safe", True) for r in done)
    energy_ok = all(r.get("energy_complete", False) for r in done)
    return len(done), len(rec), negx_avail, negx_rec, safety_ok, energy_ok, sum(r.get("two_stage_adds", False) for r in done)


def gate(rows: "list[dict[str, Any]]") -> "dict[str, Any]":
    n_done, n_rec, negx_avail, negx_rec, safety_ok, energy_ok, adds = _tally(rows)
    passed = n_rec >= 6 and (negx_rec >= 2 or negx_avail < 2) and safety_ok and energy_ok
    return {"n": len(rows), "certified": n_done, "recovered": n_rec,
            "negative_x_recovered": f"{negx_rec}/{negx_avail}", "two_stage_adds": adds,
            "safety_ok": safety_ok, "energy_complete_all": energy_ok,
            "verdict": ("R11_5_PLUS_RESIDUAL_RECOVERY_PILOT_PASS" if passed
                        else "R11_5_PLUS_PIPELINE_PASS_RESIDUAL_RECOVERY_INSUFFICIENT")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--restarts", type=int, default=5)
    args = ap.parse_args()
    pilot = json.loads(args.taxonomy.read_text(encoding="utf-8"))["pilot"]
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=3, grasp_objective=obj)
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for i, p in enumerate(pilot, 1):
            t0 = time.perf_counter()
            r = run_scenario(rig, cfg, conf, obj, p["scenario_id"], p["category"], p["split"], args.restarts)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[{i}/{len(pilot)}] {p['scenario_id']:24s} {p['category'][:18]:18s} recovered={r.get('recovered')} "
                  f"r2_k6={r.get('r2_k6')} single={r.get('single_dtz_mm')} two={r.get('two_stage_dtz_mm')} "
                  f"align={r.get('align_verdict')} ({time.perf_counter() - t0:.0f}s)", flush=True)
    g = gate(rows)
    args.out.write_text(json.dumps({"gate": g, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(g, indent=2), flush=True)
    print("R11_5_PLUS_RECOVERY_PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
