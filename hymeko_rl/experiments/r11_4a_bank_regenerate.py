"""R11.4A re-measure: the 64-scenario x seed panel under the grasp-aware DEFAULT capture, producing the NEW bank
side-by-side (the R11.3 bank is untouched) + the R11.4A metrics = the R11.5 starting distribution.

Per (scenario, seed): grasp-aware reach+capture (bilateral dwell + delivery-ready certification) then the frozen delivery
(``characterize_delivery`` -> KINETIC entry + K6 + min_dtz). Early-exit per scenario on a certified grasped-K6 (never a
nudge-K6). Checkpointed per attempt to JSONL so a partial run is recoverable.

Metrics: certified-grasp rate, KINETIC-entry rate, valid (grasped) K6 rate, and the 5-class R11.4A0 taxonomy
(K6_WITH_VALID_DELIVERY_MODE / K6_WITHOUT_DELIVERY_MODE_TRANSITION / SETTLE_ or DELIVERY_FAILURE_AFTER_VALID_GRASP /
CAPTURE_FAIL). Deterministic (fixed seeds). CPU-bound MuJoCo (no GPU benefit); runnable on Mac or katolab-CPU.

Run (respect the 16 GB RSS cap):
    systemd-run --user -p MemoryMax=16G -- python -m hymeko_rl.experiments.r11_4a_bank_regenerate
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from hymeko_rl.coin_delivery.delivery_teacher.regrasp_characterize import characterize_delivery
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective, is_certified_grasp
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

DEFAULT_BANK = Path("reports/2026-07-30-r11-4a-bank/bank.jsonl")
SEEDS = (0, 1, 2)
SETTLE_TOL_MM = 20.0            # within CENTER_TOL of the zone but not K6 => settle failure (vs a farther delivery failure)


def classify(certified: bool, reaches_kinetic: bool, k6: bool, min_dtz_mm: float) -> str:
    """5-class R11.4A0 outcome for one attempt (certification gates everything; a nudge-K6 is not a valid delivery)."""
    if not certified:
        return "CAPTURE_FAIL"
    if k6:
        return "K6_WITH_VALID_DELIVERY_MODE" if reaches_kinetic else "K6_WITHOUT_DELIVERY_MODE_TRANSITION"
    if min_dtz_mm <= SETTLE_TOL_MM:
        return "SETTLE_FAILURE_AFTER_VALID_GRASP"
    return "DELIVERY_FAILURE_AFTER_VALID_GRASP"


def _attempt(rig: dict[str, Any], cfg: Any, conf: PipelineConfig, sc: Any, seed: int, obj: GraspObjective) -> dict:
    """One grasp-aware reach+capture+delivery attempt -> a serialisable metrics row."""
    home, coin = Z._home_with_coin(rig, sc.coin_xy)
    reason, rc = P._do_reach_and_capture(rig, sc, coin, home, cfg, conf, seed)
    base = dict(scenario_id=sc.scenario_id, split=sc.split.value, seed=seed)
    if rc is None:
        return {**base, "reach_fail": reason, "certified": False, "cls": "CAPTURE_FAIL"}
    o = rc.result.outcome
    dm = characterize_delivery(o.snapshot, rig["down"])
    cert = bool(is_certified_grasp(o, obj))
    return {**base, "reach_fail": None, "contacts": int(o.contacts), "dwell": int(o.bilateral_dwell), "certified": cert,
            "kinetic": bool(dm.reaches_kinetic), "deliver_k6": bool(dm.k6),
            "deliver_dtz_mm": round(float(dm.min_dtz_mm), 2),
            "cls": classify(cert, dm.reaches_kinetic, dm.k6, dm.min_dtz_mm)}


def measure(rig: dict[str, Any], cfg: Any, scenarios: list, out_path: Path) -> "tuple[list[dict], int]":
    """Run the panel with the grasp-aware default; per scenario stop on the first certified grasped-K6. JSONL-checkpointed."""
    if out_path.exists():
        out_path.unlink()
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=3, grasp_objective=obj)
    rows, attempts = [], 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, sc in enumerate(scenarios, start=1):
            for seed in SEEDS:
                t0 = time.perf_counter()
                row = _attempt(rig, cfg, conf, sc, seed, obj)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                attempts += 1
                print(f"[{i:2d}/{len(scenarios)} s{seed}] {sc.scenario_id:22s} {row['cls']:34s} "
                      f"cert={row['certified']!s:5s} ({time.perf_counter() - t0:.1f}s) [att {attempts}]", flush=True)
                if row.get("deliver_k6") and row.get("kinetic") and row["certified"]:
                    break
    return rows, attempts


def summarize(rows: "list[dict]") -> dict:
    n = len(rows)

    def rate(key: str) -> float:
        return round(sum(1 for r in rows if r.get(key)) / n, 3) if n else 0.0

    valid_k6 = round(sum(1 for r in rows if r.get("deliver_k6") and r.get("certified")) / n, 3) if n else 0.0
    return {"attempts": n, "certified_grasp_rate": rate("certified"), "kinetic_rate": rate("kinetic"),
            "valid_k6_rate": valid_k6, "class_counts": dict(Counter(r["cls"] for r in rows))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N scenarios (0 = all 64)")
    args = ap.parse_args()
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    scenarios = build_bank_scenarios()
    if args.limit > 0:
        scenarios = scenarios[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows, attempts = measure(rig, cfg, scenarios, args.out)
    wall = time.perf_counter() - t0
    print(f"\n=== R11.4A REMEASURE ({wall:.1f}s, {attempts} attempts, {wall / max(1, attempts):.1f}s/attempt) ===", flush=True)
    print(json.dumps(summarize(rows), indent=2), flush=True)
    print("R11_4A_REMEASURE_DONE", flush=True)


if __name__ == "__main__":
    main()
