"""R11.5+++ re-A/B — frozen 3-arm on the full 10 INSUFFICIENT panel, one shared candidate bank per scenario.

For each scenario, build ONE bank of up to N=40 certified grasps (capture controller frozen), deliver EACH with the frozen
teacher (``solve_delivery`` / ``full_transport_spec``, R restarts). Three arms read the SAME bank (no regeneration, same
physics, same transport) so population depth and selection are cleanly separated:
  * A0 = current selection over the first 10 seeds (first certified grasp) — the deployed pick at the old budget;
  * A1 = current selection over all 40 seeds (first certified grasp) — depth with the SAME selection;
  * A2 = deliverability-ranked over all 40 seeds (teacher-only oracle) — depth + ranking.
A0->A1 = population-depth gain; A1->A2 = selection/ranking gain.

Pre-registered PASS: A2 >= 8/10 strict K6; A2 >= A0 + 3; A2 not worse than A1; 0 safety regression; 0 nudge-only K6; every
selected grasp certified; transport bit-identical. Verdict R11_5_PLUS_DELIVERABILITY_RANKED_ENLARGED_POPULATION_PASS.
Parallelizable via --offset/--limit. Teacher-only oracle (selection_kind=DELIVERY_ORACLE_RANKED_CAPTURE_TEACHER).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from hymeko_rl.coin_delivery.delivery_teacher.deliverability_ranking import (
    SELECTION_KIND,
    GraspDelivery,
    select_deliverable_grasp,
)
from hymeko_rl.coin_delivery.delivery_teacher.solver import full_transport_spec
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective, is_certified_grasp
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_5_plus_deliverability_pilot import _candidate, _deliver_best

TAXONOMY = Path("reports/2026-07-31-r11-5-plus-residual-taxonomy.json")
DEFAULT_OUT = Path("reports/2026-07-31-r11-5ppp-reab.json")
INSUFFICIENT = "INSUFFICIENT_TRANSPORT_PROGRESS"
A0_BUDGET = 10


def _build_bank(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, scen: Any, n_seeds: int,
                delivery_restarts: int) -> "list[tuple[int, GraspDelivery, dict[str, Any]]]":
    """The shared candidate bank: each certified grasp (seed order) with its frozen-teacher delivery + handoff log."""
    tspec = full_transport_spec()
    bank = []
    for seed in range(n_seeds):
        home, coin = Z._home_with_coin(rig, scen.coin_xy)
        _r, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg, conf, seed)
        if rc is None or not is_certified_grasp(rc.result.outcome, obj):
            continue
        gd, log = _candidate(seed, rc, _deliver_best(rc.result.outcome.snapshot, tspec, delivery_restarts))
        bank.append((seed, gd, log))
    return bank


def _current(bank: "list[tuple[int, GraspDelivery, dict[str, Any]]]", seed_cap: int) -> "GraspDelivery | None":
    """Current selection = the first certified grasp among seeds < ``seed_cap`` (the deployed early-exit pick)."""
    return next((gd for seed, gd, _ in bank if seed < seed_cap), None)


def _k6(g: "GraspDelivery | None") -> bool:
    return bool(g is not None and g.k6)


def run_scenario(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, sid: str, split: str,
                 n_seeds: int, delivery_restarts: int) -> "dict[str, Any]":
    scen = next(s for s in build_bank_scenarios() if s.scenario_id == sid)
    bank = _build_bank(rig, cfg, conf, obj, scen, n_seeds, delivery_restarts)
    if not bank:
        return {"scenario_id": sid, "split": split, "n_grasps": 0, "note": "no certified grasp in the bank"}
    a0, a1 = _current(bank, A0_BUDGET), _current(bank, n_seeds)
    a2 = select_deliverable_grasp([gd for _, gd, _ in bank])
    grasps = [gd for _, gd, _ in bank]
    return {"scenario_id": sid, "split": split, "selection_kind": SELECTION_KIND, "teacher_only": True,
            "n_grasps": len(grasps), "n_deliverable_k6": sum(g.k6 for g in grasps),
            "all_certified": all(g.certified for g in grasps), "all_safe": all(g.safe for g in grasps),
            "a0_seed": (a0.seed if a0 else None), "a0_k6": _k6(a0), "a0_dtz_mm": (round(a0.delivered_dtz_mm, 2) if a0 else None),
            "a1_seed": a1.seed if a1 else None, "a1_k6": _k6(a1), "a1_dtz_mm": (round(a1.delivered_dtz_mm, 2) if a1 else None),
            "a2_seed": a2.seed, "a2_k6": _k6(a2), "a2_dtz_mm": round(a2.delivered_dtz_mm, 2),
            "candidates": [log for _, _, log in bank]}


def gate(rows: "list[dict[str, Any]]") -> "dict[str, Any]":
    done = [r for r in rows if r.get("n_grasps", 0) > 0]
    a0 = sum(1 for r in done if r.get("a0_k6"))
    a1 = sum(1 for r in done if r.get("a1_k6"))
    a2 = sum(1 for r in done if r.get("a2_k6"))
    a2_not_worse = all(r.get("a2_k6") or not r.get("a1_k6") for r in done)   # A2 K6 wherever A1 is K6
    safe = all(r.get("all_safe", True) for r in done)
    certified = all(r.get("all_certified", True) for r in done)
    passed = a2 >= 8 and a2 >= a0 + 3 and a2_not_worse and safe and certified
    verdict = ("R11_5_PLUS_DELIVERABILITY_RANKED_ENLARGED_POPULATION_PASS" if passed
               else "R11_5_PLUS_DELIVERABILITY_RANKED_ENLARGED_POPULATION_INSUFFICIENT")
    return {"n": len(rows), "scenarios_with_grasps": len(done),
            "a0_k6": f"{a0}/{len(done)}", "a1_k6": f"{a1}/{len(done)}", "a2_k6": f"{a2}/{len(done)}",
            "population_depth_gain": a1 - a0, "ranking_gain": a2 - a1, "a2_over_a0": a2 - a0,
            "a2_not_worse_than_a1": a2_not_worse, "all_safe": safe, "all_certified": certified, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--delivery-restarts", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 10 INSUFFICIENT")
    args = ap.parse_args()
    tax = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    ids = [(r["scenario_id"], r["split"]) for r in tax["records"] if r["residual_class"] == INSUFFICIENT]
    ids = ids[args.offset:(args.offset + args.limit if args.limit else len(ids))]
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=1, grasp_objective=obj)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for i, (sid, split) in enumerate(ids, 1):
            t0 = time.perf_counter()
            r = run_scenario(rig, cfg, conf, obj, sid, split, args.n_seeds, args.delivery_restarts)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[{i}/{len(ids)}] {sid:22s} grasps={r.get('n_grasps')} A0_k6={r.get('a0_k6')} A1_k6={r.get('a1_k6')} "
                  f"A2_k6={r.get('a2_k6')} A2_dtz={r.get('a2_dtz_mm')}(seed {r.get('a2_seed')}) "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    g = gate(rows)
    args.out.write_text(json.dumps({"gate": g, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(g, indent=2), flush=True)
    print("R11_5PPP_REAB_DONE", flush=True)


if __name__ == "__main__":
    main()
