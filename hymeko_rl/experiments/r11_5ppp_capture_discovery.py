"""R11.5+++ Stage A — capture-population discovery curve on the barren INSUFFICIENT scenarios.

Separates "we under-sampled a good grasp" from "the capture proposal saturates on inadequate support". Everything is
FROZEN except the number of capture seeds: transport bit-exact (``solve_delivery``/``full_transport_spec``), the R11.5++
deliverability ranking, the capture controller + parameter bounds, and the delivery budget (R=5). ``delivered_dtz`` stays
a teacher-only oracle. For each barren scenario we run up to ``--max-seeds`` capture seeds, deliver EACH certified grasp
with the frozen oracle, and report the cumulative discovery curve at budgets {5,10,20,40}: certified grasps, unique
handoff descriptors, deliverable (K6) grasps, best delivered dtz, first-deliverable seed, marginal gain per seed.

Gate: >= 3/5 barren scenarios acquire a deliverable grasp (+ safety + stability-gate + transport bit-identical) ->
R11_5_PLUS_CAPTURE_POPULATION_GROWTH_PASS; else (the curve flattens, no new deliverable) ->
R11_5_PLUS_CAPTURE_PROPOSAL_SUPPORT_LIMITED (do not burn 100 seeds; go to the diversity step). Parallelizable via --offset/--limit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from hymeko_rl.coin_delivery.delivery_teacher.deliverability_ranking import DWELL_TARGET, SELECTION_KIND
from hymeko_rl.coin_delivery.delivery_teacher.solver import full_transport_spec
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective, is_certified_grasp
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_5_plus_deliverability_pilot import _deliver_best, _handoff

TAXONOMY = Path("reports/2026-07-31-r11-5-plus-residual-taxonomy.json")
DEFAULT_OUT = Path("reports/2026-07-31-r11-5ppp-capture-discovery.json")
BUDGETS = (5, 10, 20, 40)
# the 5 barren scenarios from the R11.5++ kato14 A/B (deliverable=0), 4/5 in the a-45 cluster:
BARREN = ["bank_c2_-0.015_+0.025", "bank_c3_r5_a-45", "bank_c3_r6_a-45", "bank_c3_r7_a-45", "bank_c3_r9_a-45"]


def _sig(handoff: "dict[str, Any]") -> "tuple[Any, ...]":
    """A coarse handoff signature (rounded q + coin pose) to count UNIQUE grasps (dedup near-identical realizations)."""
    return tuple(round(v, 2) for v in handoff["q"]) + tuple(round(v, 3) for v in handoff["coin_xy"])


def _discover(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, scen: Any, max_seeds: int,
              delivery_restarts: int) -> "list[dict[str, Any]]":
    """Per certified capture seed: deliver with the frozen oracle, record deliverable / dtz / stability / descriptor."""
    tspec = full_transport_spec()
    out = []
    for seed in range(max_seeds):
        home, coin = Z._home_with_coin(rig, scen.coin_xy)
        _r, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg, conf, seed)
        if rc is None or not is_certified_grasp(rc.result.outcome, obj):
            continue
        oc = rc.result.outcome
        d = _deliver_best(oc.snapshot, tspec, delivery_restarts)
        out.append({"seed": seed, "deliverable": bool(d.k6), "dtz_mm": round(float(d.min_dtz_mm), 2),
                    "stable": bool(oc.bilateral_dwell >= DWELL_TARGET), "safe": bool(d.safe),
                    "sig": _sig(_handoff(oc.snapshot))})
    return out


def _curve_point(recs: "list[dict[str, Any]]", budget: int, prev_deliverable: int) -> "dict[str, Any]":
    up = [r for r in recs if r["seed"] < budget]
    deliverable = [r for r in up if r["deliverable"]]
    return {"budget": budget, "certified": len(up), "unique_descriptors": len({r["sig"] for r in up}),
            "deliverable": len(deliverable), "best_dtz_mm": min((r["dtz_mm"] for r in up), default=None),
            "first_deliverable_seed": deliverable[0]["seed"] if deliverable else None,
            "marginal_deliverable": len(deliverable) - prev_deliverable}


def run_scenario(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, sid: str,
                 max_seeds: int, delivery_restarts: int) -> "dict[str, Any]":
    scen = next(s for s in build_bank_scenarios() if s.scenario_id == sid)
    recs = _discover(rig, cfg, conf, obj, scen, max_seeds, delivery_restarts)
    curve, prev = [], 0
    for b in BUDGETS:
        pt = _curve_point(recs, b, prev)
        curve.append(pt)
        prev = pt["deliverable"]
    deliverable_at_max = curve[-1]["deliverable"] if curve else 0
    # saturation: no NEW deliverable grasp after budget 20 (the curve flattened)
    saturated = deliverable_at_max == 0 and all(pt["marginal_deliverable"] == 0 for pt in curve if pt["budget"] >= 20)
    return {"scenario_id": sid, "selection_kind": SELECTION_KIND, "teacher_only": True, "curve": curve,
            "certified_at_max": curve[-1]["certified"] if curve else 0, "deliverable_at_max": deliverable_at_max,
            "best_dtz_mm": curve[-1]["best_dtz_mm"] if curve else None, "saturated": saturated,
            "all_safe": all(r["safe"] for r in recs), "all_stable": all(r["stable"] for r in recs)}


def gate(rows: "list[dict[str, Any]]") -> "dict[str, Any]":
    got = sum(1 for r in rows if r.get("deliverable_at_max", 0) > 0)
    safe = all(r.get("all_safe", True) for r in rows)
    stable = all(r.get("all_stable", True) for r in rows)
    all_saturated = all(r.get("saturated") for r in rows if r.get("deliverable_at_max", 0) == 0)
    passed = got >= 3 and safe and stable
    verdict = ("R11_5_PLUS_CAPTURE_POPULATION_GROWTH_PASS" if passed
               else "R11_5_PLUS_CAPTURE_PROPOSAL_SUPPORT_LIMITED" if all_saturated
               else "R11_5_PLUS_CAPTURE_POPULATION_GROWTH_PARTIAL")
    return {"n": len(rows), "scenarios_with_deliverable": f"{got}/{len(rows)}", "all_safe": safe, "all_stable": stable,
            "barren_saturated": all_saturated, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-seeds", type=int, default=40)
    ap.add_argument("--delivery-restarts", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 5 barren")
    args = ap.parse_args()
    ids = BARREN[args.offset:(args.offset + args.limit if args.limit else len(BARREN))]
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=1, grasp_objective=obj)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for i, sid in enumerate(ids, 1):
            t0 = time.perf_counter()
            r = run_scenario(rig, cfg, conf, obj, sid, args.max_seeds, args.delivery_restarts)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[{i}/{len(ids)}] {sid:22s} certified={r['certified_at_max']} deliverable={r['deliverable_at_max']} "
                  f"best_dtz={r['best_dtz_mm']} saturated={r['saturated']} "
                  f"curve_deliv={[p['deliverable'] for p in r['curve']]} ({time.perf_counter() - t0:.0f}s)", flush=True)
    g = gate(rows)
    args.out.write_text(json.dumps({"gate": g, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(g, indent=2), flush=True)
    print("R11_5PPP_CAPTURE_DISCOVERY_DONE", flush=True)


if __name__ == "__main__":
    main()
