"""R11.5 full coverage run — the target-conditioned delivery teacher (two-stage transport+settle) over ALL 51
certified-grasp DELIVERY_FAILURE scenarios, on the SAME certified-grasp state as frozen R2.

Reuses the pilot's ``run_one`` (A/B, metric panel, energy diagnostic-only). EVERY attempt is retained (not just
scenario-best) — the near-misses are valuable for later critic/reward calibration. Deterministic; parallelizable via
--offset/--limit.

Coverage gate (over all 64 bank scenarios; the 7 frozen-R2 K6 are already positive): overall positive coverage >= 45/64,
>= 50 % in each of C0-C3, dev + test both have positive teacher trajectories, 0 nudge-only K6 (every input is a certified
grasp), 0 safety regression, energy ledgers complete. Verdict:
R11_5_TARGET_CONDITIONED_DELIVERY_TEACHER_COVERAGE_PASS else ..._COVERAGE_INSUFFICIENT.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path

from hymeko_rl.coin_delivery.delivery_teacher.solver import full_transport_spec, solve_delivery
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_5_delivery_teacher_pilot import run_one

R11_4A_BANK = Path("reports/2026-07-30-r11-4a-bank/bank.jsonl")
DEFAULT_OUT = Path("reports/2026-07-30-r11-5-full-coverage/coverage.jsonl")


def _cclass(sid: str) -> str:
    tok = sid.split("_")[1]
    return tok[:2] if tok[:1] == "c" else tok


def _scenario_kind(sid: str, rs: list) -> "tuple[str, str] | None":
    """('r2', cclass) if this scenario got a certified R2-K6; ('fail', split) if certified but no K6; else None."""
    if any(r.get("deliver_k6") and r.get("certified") for r in rs):
        return "r2", _cclass(sid)
    if any(r.get("certified") for r in rs):
        return "fail", next(r["split"] for r in rs)
    return None


def bank_facts(bank_path: Path) -> "tuple[list[tuple[str, str]], dict, int]":
    """From the R11.4A bank: the DELIVERY_FAILURE scenarios [(sid, split)], the R2-K6 count per C-class, and total."""
    by: dict = defaultdict(list)
    for x in bank_path.open():
        if x.strip():
            r = json.loads(x)
            by[r["scenario_id"]].append(r)
    fails, r2_by_class = [], defaultdict(int)
    for sid, rs in by.items():
        kind = _scenario_kind(sid, rs)
        if kind and kind[0] == "r2":
            r2_by_class[kind[1]] += 1
        elif kind:
            fails.append((sid, kind[1]))
    return fails, dict(r2_by_class), sum(r2_by_class.values())


def _count_by_class(sids) -> dict:
    out: dict = defaultdict(int)
    for sid in sids:
        out[_cclass(sid)] += 1
    return out


def _coverage_passed(overall: int, cov: dict, rec_split: dict, nudge_only: int,
                     safety_ok: bool, energy_ok: bool) -> bool:
    """Pre-registered coverage gate: >=45/64 overall, >=50% in each C-class, dev+test positive, 0 nudge-K6, safe, energy."""
    return (overall >= 45 and all(v >= 0.50 for v in cov.values()) and rec_split["dev"] >= 1
            and rec_split["test"] >= 1 and nudge_only == 0 and safety_ok and energy_ok)


def _gate_metrics(rows: list) -> tuple:
    """(done, recovered, safety_ok, energy_ok, nudge_only_k6) from the attempt rows."""
    done = [r for r in rows if r.get("certified")]
    rec = [r for r in done if r.get("recovered")]
    safety_ok = all(r.get("teacher_safe", True) for r in done)
    energy_ok = all(r.get("energy_complete", False) for r in done)
    nudge_only = sum(1 for r in done if r.get("teacher_k6") and not r.get("certified"))
    return done, rec, safety_ok, energy_ok, nudge_only


def _base_coverage_ok(overall: int, rec_split: dict, nudge: int, safety: bool, energy: bool) -> bool:
    """Everything the global gate needs EXCEPT per-class >=50%: >=45/64, dev+test positive, 0 nudge-K6, safe, energy."""
    return overall >= 45 and rec_split["dev"] >= 1 and rec_split["test"] >= 1 and nudge == 0 and safety and energy


def _coverage_verdict(overall: int, cov: dict, rec_split: dict, nudge: int, safety: bool, energy: bool) -> str:
    if _coverage_passed(overall, cov, rec_split, nudge, safety, energy):
        return "R11_5_TARGET_CONDITIONED_DELIVERY_TEACHER_COVERAGE_PASS"
    if _base_coverage_ok(overall, rec_split, nudge, safety, energy):   # global >=45 but a C-class <50% (likely far C3)
        return "R11_5_GLOBAL_COVERAGE_PASS_WITH_HARD_GEOMETRY_GAP"
    return "R11_5_TEACHER_COVERAGE_INSUFFICIENT"


def coverage_gate(rows: list, bank_path: Path = R11_4A_BANK) -> dict:
    _fails, r2_by_class, r2_total = bank_facts(bank_path)
    class_total = _count_by_class(s.scenario_id for s in build_bank_scenarios())
    done, rec, safety_ok, energy_ok, nudge_only = _gate_metrics(rows)
    rec_by_class = _count_by_class(r["scenario_id"] for r in rec)
    overall = r2_total + len(rec)
    cov = {c: round((r2_by_class.get(c, 0) + rec_by_class.get(c, 0)) / class_total[c], 3) for c in sorted(class_total)}
    rec_split = {sp: sum(1 for r in rec if r["split"] == sp) for sp in ("train", "dev", "test")}
    return {"scenarios": len(rows), "certified": len(done), "teacher_recovered": len(rec),
            "overall_coverage": f"{overall}/64", "coverage_by_class": cov, "recovered_by_split": rec_split,
            "nudge_only_k6": nudge_only, "safety_ok": safety_ok, "energy_complete_all": energy_ok,
            "verdict": _coverage_verdict(overall, cov, rec_split, nudge_only, safety_ok, energy_ok)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bank", type=Path, default=R11_4A_BANK)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all 51 failures")
    ap.add_argument("--restarts", type=int, default=11, help="delivery CEM restarts R (frozen protocol: 11)")
    ap.add_argument("--capture-seeds", type=int, default=5, help="certified-grasp regen attempts (frozen: 5)")
    args = ap.parse_args()
    fails, _r2, _n = bank_facts(args.bank)
    fails = fails[args.offset:(args.offset + args.limit if args.limit else len(fails))]
    rig = _rig()
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=3, grasp_objective=obj)
    tspec = full_transport_spec()                                     # SAME coordinate/objective as the re-gate (transport)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("w", encoding="utf-8") as fh:
        for i, (sid, split) in enumerate(fails, 1):
            t0 = time.perf_counter()
            r = run_one(rig, cfg, conf, obj, lambda snap, s: solve_delivery(snap, s, tspec), sid, split,
                        args.restarts, args.capture_seeds)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[{i}/{len(fails)}] {sid:26s} {split:5s} recovered={r.get('recovered')} "
                  f"dtz={r.get('teacher_dtz_mm')} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("=== R11.5 FULL COVERAGE GATE ===", flush=True)
    print(json.dumps(coverage_gate(rows, args.bank), indent=2), flush=True)
    print("R11_5_COVERAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
