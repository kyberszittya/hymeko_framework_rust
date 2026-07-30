"""R11.5++ Phase — deliverability-ranked capture-grasp selection A/B on the 10 INSUFFICIENT_TRANSPORT_PROGRESS residuals.

Transport is FROZEN (same ``solve_delivery`` / ``full_transport_spec`` / objective / K6 + safety contracts). Only the
capture-candidate SELECTION varies. Per scenario: generate N certified grasps from the capture population (N seeds, all
certified collected — capture controller unchanged), deliver EACH with the frozen teacher, then compare:
  * Arm C (current): the first certified grasp (what ``_certified_grasp_snap`` early-exits on);
  * Arm D (ranked):  the deliverability-ranked best (``select_deliverable_grasp``, a teacher-only oracle).
Logs the pre-delivery handoff descriptor per candidate (to train a handoff->deliverability surrogate later). Energy
diagnostic-only. Provenance: selection_kind=DELIVERY_ORACLE_RANKED_CAPTURE_TEACHER, teacher_only=True. Parallelizable via
--offset/--limit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_teacher.deliverability_ranking import (
    SELECTION_KIND,
    GraspDelivery,
    is_material_dtz_improvement,
    select_deliverable_grasp,
)
from hymeko_rl.coin_delivery.delivery_teacher.solver import DeliveryResult, full_transport_spec, solve_delivery
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective, is_certified_grasp
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

TAXONOMY = Path("reports/2026-07-31-r11-5-plus-residual-taxonomy.json")
DEFAULT_OUT = Path("reports/2026-07-31-r11-5pp-deliverability-pilot.json")
INSUFFICIENT = "INSUFFICIENT_TRANSPORT_PROGRESS"


def _handoff(snap: Any) -> "dict[str, Any]":
    """The pre-delivery handoff descriptor (what a deployment surrogate must predict deliverability from)."""
    rl = snap.branch()
    d = rl.inner.data
    coin_vel = np.asarray(rl.inner._planar_metrics.disk_vel, np.float64)[:2]
    return {"q": [round(float(v), 4) for v in d.qpos[:4]], "qdot": [round(float(v), 4) for v in d.qvel[:4]],
            "prev_tau": [round(float(v), 4) for v in snap.prev_tau],
            "coin_xy": [round(float(v), 4) for v in _coin_xy(rl)],
            "coin_vel": [round(float(v), 4) for v in coin_vel]}


def _candidate(seed: int, rc: Any, d: DeliveryResult) -> "tuple[GraspDelivery, dict[str, Any]]":
    """Build the ranking record + the per-candidate log for one certified grasp's frozen delivery."""
    oc, params, m = rc.result.outcome, rc.result.params, d.measurements
    gd = GraspDelivery(seed=seed, certified=True, safe=bool(d.safe), bilateral_dwell=int(oc.bilateral_dwell),
                       kinetic=bool(m["touched"]), delivered_dtz_mm=float(d.min_dtz_mm),
                       gap_closed=float(m["gap_closed"]), contact_delay=int(oc.left_right_contact_delay), k6=bool(d.k6))
    log = {"seed": seed,
           "capture": {"n": params.n, "s": round(float(params.s), 4), "preload_start": round(float(params.preload_start), 4),
                       "bmax": round(float(params.bmax), 4)},
           "bilateral_dwell": int(oc.bilateral_dwell), "contact_delay": int(oc.left_right_contact_delay),
           "first_contact_relvel": _num(oc.first_contact_relvel), "second_contact_relvel": _num(oc.second_contact_relvel),
           "coin_disp_capture_mm": _num(oc.coin_disp_capture_mm), "terminal_coin_speed": _num(oc.terminal_coin_speed),
           "handoff": _handoff(rc.result.outcome.snapshot),
           "delivery": {"kinetic": bool(m["touched"]), "min_dtz_mm": float(d.min_dtz_mm),
                        "final_dtz_mm": round(float(m["dtz_end"]) * 1000, 2), "gap_closed": round(float(m["gap_closed"]), 3),
                        "forward_mm": round(float(m["forward"]) * 1000, 1), "k6": bool(d.k6), "safe": bool(d.safe)}}
    return gd, log


def _num(v: float) -> "float | None":
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)


def _deliver_best(snap: Any, tspec: Any, restarts: int) -> DeliveryResult:
    """Frozen delivery teacher, best of ``restarts`` seeds by (k6, -min_dtz), early-exit on K6 — the SAME transport
    budget the baseline gives a scenario, now measuring a single grasp's deliverability (transport bit-identical)."""
    best: DeliveryResult | None = None
    for s in range(restarts):
        d = solve_delivery(snap, seed=s, spec=tspec)
        if best is None or (d.k6, -d.min_dtz_mm) > (best.k6, -best.min_dtz_mm):
            best = d
        if d.k6:
            break
    assert best is not None
    return best


def _collect(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, scen: Any,
             n_seeds: int) -> "list[tuple[int, Any]]":
    """Generate up to ``n_seeds`` certified grasps from the capture population (all certified kept; controller frozen)."""
    out = []
    for seed in range(n_seeds):
        home, coin = Z._home_with_coin(rig, scen.coin_xy)
        _r, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg, conf, seed)
        if rc is not None and is_certified_grasp(rc.result.outcome, obj):
            out.append((seed, rc))
    return out


def run_scenario(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, sid: str, split: str,
                 n_seeds: int, delivery_restarts: int) -> "dict[str, Any]":
    scen = next(s for s in build_bank_scenarios() if s.scenario_id == sid)
    grasps = _collect(rig, cfg, conf, obj, scen, n_seeds)
    if not grasps:
        return {"scenario_id": sid, "split": split, "n_grasps": 0, "note": "no certified grasp in the population"}
    tspec = full_transport_spec()                                            # FROZEN transport (bit-identical to baseline)
    cands = [_candidate(seed, rc, _deliver_best(rc.result.outcome.snapshot, tspec, delivery_restarts))
             for seed, rc in grasps]
    graspdels = [c[0] for c in cands]
    current, ranked = graspdels[0], select_deliverable_grasp(graspdels)      # Arm C = first certified; Arm D = ranked
    return {"scenario_id": sid, "split": split, "selection_kind": SELECTION_KIND, "teacher_only": True,
            "n_grasps": len(graspdels), "n_deliverable_k6": sum(g.k6 for g in graspdels),
            "all_certified": all(g.certified for g in graspdels), "all_safe": all(g.safe for g in graspdels),
            "current_seed": current.seed, "current_k6": current.k6, "current_dtz_mm": round(current.delivered_dtz_mm, 2),
            "ranked_seed": ranked.seed, "ranked_k6": ranked.k6, "ranked_dtz_mm": round(ranked.delivered_dtz_mm, 2),
            "recovered_by_ranking": bool(ranked.k6 and not current.k6),
            "material_dtz_improvement": is_material_dtz_improvement(current.delivered_dtz_mm, ranked.delivered_dtz_mm),
            "candidates": [c[1] for c in cands]}


def _tally(rows: "list[dict[str, Any]]") -> "tuple[list[dict[str, Any]], int, int, int, bool, bool, bool]":
    """(done, #material-dtz-improve, #ranked-K6, #recovered-by-ranking, all-safe, all-certified, any-deliverable)."""
    done = [r for r in rows if r.get("n_grasps", 0) > 0]
    return (done, sum(1 for r in done if r.get("material_dtz_improvement")),
            sum(1 for r in done if r.get("ranked_k6")), sum(1 for r in done if r.get("recovered_by_ranking")),
            all(r.get("all_safe", True) for r in done), all(r.get("all_certified", True) for r in done),
            any(r.get("n_deliverable_k6", 0) > 0 for r in done))


def gate(rows: "list[dict[str, Any]]") -> "dict[str, Any]":
    done, material, ranked_k6, recovered, safe, certified, any_deliverable = _tally(rows)
    passed = material >= 6 and ranked_k6 >= 5 and safe and certified
    verdict = ("R11_5_PLUS_DELIVERABILITY_RANKED_GRASP_PILOT_PASS" if passed
               else "CAPTURE_DELIVERABILITY_SUPPORT_INSUFFICIENT" if not any_deliverable
               else "CAPTURE_DELIVERABILITY_RANKING_CONTRACT_GAP")
    return {"n": len(rows), "scenarios_with_grasps": len(done), "material_dtz_improvement": f"{material}/{len(done)}",
            "ranked_k6": f"{ranked_k6}/{len(done)}", "recovered_by_ranking": recovered, "all_safe": safe,
            "all_certified": certified, "any_deliverable": any_deliverable, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-seeds", type=int, default=10, help="capture population size (seeds; all certified kept)")
    ap.add_argument("--delivery-restarts", type=int, default=5, help="frozen-transport restarts per grasp (baseline budget)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 10 INSUFFICIENT")
    args = ap.parse_args()
    tax = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    scen_ids = [(r["scenario_id"], r["split"]) for r in tax["records"] if r["residual_class"] == INSUFFICIENT]
    scen_ids = scen_ids[args.offset:(args.offset + args.limit if args.limit else len(scen_ids))]
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=1, grasp_objective=obj)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for i, (sid, split) in enumerate(scen_ids, 1):
            t0 = time.perf_counter()
            r = run_scenario(rig, cfg, conf, obj, sid, split, args.n_seeds, args.delivery_restarts)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[{i}/{len(scen_ids)}] {sid:22s} grasps={r.get('n_grasps')} deliverable={r.get('n_deliverable_k6')} "
                  f"current={r.get('current_dtz_mm')}(k6={r.get('current_k6')}) ranked={r.get('ranked_dtz_mm')}"
                  f"(k6={r.get('ranked_k6')}) recovered={r.get('recovered_by_ranking')} ({time.perf_counter() - t0:.0f}s)",
                  flush=True)
    g = gate(rows)
    args.out.write_text(json.dumps({"gate": g, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(g, indent=2), flush=True)
    print("R11_5PP_DELIVERABILITY_PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
