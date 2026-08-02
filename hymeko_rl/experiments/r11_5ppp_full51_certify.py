"""R11.5+++ official full-51 certification — deliverability-ranked enlarged-population selection over ALL 51
DELIVERY_FAILURE scenarios, at the frozen transport budget R=11. This closes the official coverage number.

Frozen protocol (no new research): capture population N=40; deliverability-ranked certified-grasp selection (teacher-only
oracle); delivery CEM restarts R=11; single-stage target-conditioned transport (``solve_delivery`` / ``full_transport_spec``);
NO ALIGN, no new controller, no new score/coordinate. Per scenario, build a bank of up to N=40 certified grasps, deliver
EACH with the frozen teacher at R=11, select the deliverability-ranked best (the coverage-counting delivery). Every
candidate's ``handoff descriptor -> oracle delivered dtz / K6`` is retained in the shard (the demonstration bank for the
two later learning targets: a coin/target-conditioned delivery policy + a cheap handoff->deliverability surrogate that
replaces the oracle at deployment). Coverage gate (``r11_5_full_coverage.coverage_gate``): overall >= 45/64, C0-C3 each
>= 50 %, dev + test positive, 0 nudge-only K6, 0 safety regression, 100 % energy/provenance ->
R11_5_TARGET_CONDITIONED_DELIVERY_TEACHER_COVERAGE_PASS. Parallelizable via --offset/--limit.
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
from hymeko_rl.coin_delivery.delivery_teacher.solver import DeliveryResult, full_transport_spec
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective, is_certified_grasp
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_5_full_coverage import R11_4A_BANK, bank_facts, coverage_gate
from hymeko_rl.experiments.r11_5_plus_deliverability_pilot import _candidate, _deliver_best

DEFAULT_OUT = Path("reports/2026-08-02-r11-5ppp-full51-certify.json")


def _bank(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, scen: Any, n_seeds: int,
          restarts: int) -> "list[tuple[GraspDelivery, DeliveryResult, dict[str, Any]]]":
    """Up to ``n_seeds`` certified grasps, each delivered by the frozen teacher at R=``restarts`` (keeps the DeliveryResult
    for the selected grasp's energy ledger + the per-candidate handoff log for the demonstration bank)."""
    tspec = full_transport_spec()
    out = []
    for seed in range(n_seeds):
        home, coin = Z._home_with_coin(rig, scen.coin_xy)
        _r, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg, conf, seed)
        if rc is None or not is_certified_grasp(rc.result.outcome, obj):
            continue
        d = _deliver_best(rc.result.outcome.snapshot, tspec, restarts)
        gd, log = _candidate(seed, rc, d)
        out.append((gd, d, log))
    return out


def run_scenario(rig: "dict[str, Any]", cfg: Any, conf: PipelineConfig, obj: GraspObjective, sid: str, split: str,
                 n_seeds: int, restarts: int) -> "dict[str, Any]":
    scen = next(s for s in build_bank_scenarios() if s.scenario_id == sid)
    bank = _bank(rig, cfg, conf, obj, scen, n_seeds, restarts)
    if not bank:
        return {"scenario_id": sid, "split": split, "certified": False, "recovered": False, "teacher_k6": False,
                "teacher_safe": True, "energy_complete": True, "note": "no certified grasp in the N=40 bank"}
    best_gd = select_deliverable_grasp([gd for gd, _, _ in bank])
    sel = next(d for gd, d, _ in bank if gd.seed == best_gd.seed)            # the selected grasp's DeliveryResult
    return {"scenario_id": sid, "split": split, "selection_kind": SELECTION_KIND, "teacher_only": True,
            "certified": True, "teacher_k6": bool(sel.k6), "teacher_dtz_mm": sel.min_dtz_mm,
            "teacher_safe": bool(sel.safe), "recovered": bool(sel.k6), "selected_seed": best_gd.seed,
            "energy_complete": bool(sel.energy.is_complete()), "n_grasps": len(bank),
            "candidates": [log for _, _, log in bank]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=R11_4A_BANK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-seeds", type=int, default=40)
    ap.add_argument("--restarts", type=int, default=11)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 51 DELIVERY_FAILURE scenarios")
    args = ap.parse_args()
    fails, _r2, _n = bank_facts(args.bank)
    fails = fails[args.offset:(args.offset + args.limit if args.limit else len(fails))]
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=1, grasp_objective=obj)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.with_suffix(".jsonl").open("w", encoding="utf-8") as fh:
        for i, (sid, split) in enumerate(fails, 1):
            t0 = time.perf_counter()
            r = run_scenario(rig, cfg, conf, obj, sid, split, args.n_seeds, args.restarts)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[{i}/{len(fails)}] {sid:26s} {split:5s} certified={r.get('certified')} recovered={r.get('recovered')} "
                  f"dtz={r.get('teacher_dtz_mm')} seed={r.get('selected_seed')} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("=== R11.5+++ FULL-51 CERTIFICATION GATE (deliverability-ranked N=40, R=11) ===", flush=True)
    g = coverage_gate(rows, args.bank)
    args.out.write_text(json.dumps({"gate": g, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(g, indent=2), flush=True)
    print("R11_5PPP_CERTIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
