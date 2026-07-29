"""R11.4A bounded capture->delivery re-grasp characterization (measure BEFORE any teacher search).

For each bank scenario: reach + capture (deployed RRT + capture teacher) -> characterize the FROZEN capture->delivery
transition (does it reach KINETIC + re-establish a bilateral delivery-ready grasp?). Writes a per-scenario JSONL + an
addressability summary. This measures the coverage of the current frozen re-grasp controller over the RRT-straddle set
(G_RRT vs B_regrasp) — a coverage gap, never physical infeasibility. No teacher search here.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from hymeko_rl.coin_delivery.delivery_teacher.regrasp_characterize import RegraspClass, characterize_delivery
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.demo_bank.scenario import CoinTargetScenario, curriculum_stage
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

DEFAULT_OUT = Path("reports/2026-07-30-r11-4a-regrasp-characterization")
_CAPTURE = PipelineConfig(teacher_budget=1)   # one capture seed per scenario (a representative delivery-start state)


def _characterize_scenario(rig: dict[str, Any], sc: CoinTargetScenario) -> "dict[str, Any] | None":
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    home, coin = Z._home_with_coin(rig, sc.coin_xy)
    reason, rc = P._do_reach_and_capture(rig, sc, coin, home, cfg, _CAPTURE, 0)
    if rc is None:
        return {"scenario_id": sc.scenario_id, "curriculum_stage": curriculum_stage(sc.kind),
                "split": sc.split.value, "reach_ok": False, "reach_reason": reason,
                "regrasp_class": "reach_or_goal_failure", "addressable": False}
    met = characterize_delivery(rc.result.outcome.snapshot, rig["down"])
    d = dataclasses.asdict(met)
    d.update({"scenario_id": sc.scenario_id, "curriculum_stage": curriculum_stage(sc.kind), "split": sc.split.value,
              "reach_ok": True, "coin_pose": [round(float(x), 5) for x in sc.coin_xy],
              "target_pose": None if sc.target_xy is None else [round(float(x), 5) for x in sc.target_xy],
              "capture_contacts": int(rc.result.outcome.contacts),
              "addressable": met.regrasp_class == RegraspClass.DELIVERY_ADDRESSABLE.value})
    return d


def run(out_dir: Path, limit: int = 0) -> list[dict[str, Any]]:
    rig = _rig()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "regrasp.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    scenarios = build_bank_scenarios()
    if limit > 0:
        scenarios = scenarios[:limit]
    rows: list[dict[str, Any]] = []
    for i, sc in enumerate(scenarios, start=1):
        t0 = time.perf_counter()
        row = _characterize_scenario(rig, sc)
        assert row is not None
        rows.append(row)
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[{i:2d}/{len(scenarios)}] {sc.scenario_id:22s} {row.get('regrasp_class','?'):40s} "
              f"kin={row.get('reaches_kinetic')} grasp_start={row.get('grasp_at_delivery_start')} "
              f"addr={row['addressable']} ({time.perf_counter()-t0:.1f}s)", flush=True)
    return rows


def _by_stage(valid: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in valid:
        st = out.setdefault(r["curriculum_stage"], {"n": 0, "addressable": 0, "reaches_kinetic": 0})
        st["n"] += 1
        st["addressable"] += int(r["addressable"])
        st["reaches_kinetic"] += int(bool(r.get("reaches_kinetic")))
    return out


def _outcome_count(addr: list[dict[str, Any]], label: str) -> int:
    return sum(1 for r in addr if r.get("addressable_outcome") == label)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("reach_ok")]
    addr = [r for r in valid if r["addressable"]]
    return {"n_scenarios": len(rows), "n_valid_reach": len(valid),
            "reaches_kinetic": sum(1 for r in valid if r.get("reaches_kinetic")),
            "delivery_addressable": len(addr),
            "addressable_settle_failures": _outcome_count(addr, "SETTLE_FAILURE_AFTER_VALID_REGRASP"),
            "addressable_delivery_failures": _outcome_count(addr, "DELIVERY_FAILURE_AFTER_VALID_REGRASP"),
            "addressable_successes": _outcome_count(addr, "SUCCESS"),
            "class_counts": dict(Counter(r["regrasp_class"] for r in valid)), "by_stage": _by_stage(valid)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.perf_counter()
    rows = run(args.out, args.limit)
    s = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(s, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n=== ADDRESSABILITY ({time.perf_counter()-t0:.1f}s) ===", flush=True)
    print(json.dumps(s, indent=2, sort_keys=True), flush=True)
    verdict = ("R11_4A_CAPTURE_TO_DELIVERY_INTERFACE_COVERAGE_GAP_IDENTIFIED"
               if (s["addressable_settle_failures"] < 12 or s["addressable_delivery_failures"] < 8)
               else "R11_4A_CAPTURE_TO_DELIVERY_ADDRESSABILITY_CHARACTERIZED")
    print(f"VERDICT={verdict}", flush=True)
    print("REGRASP_CHARACTERIZATION_DONE", flush=True)


if __name__ == "__main__":
    main()
