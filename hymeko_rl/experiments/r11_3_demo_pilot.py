"""R11.3 demonstration-bank PILOT runner + Phase-7 gates.

Runs the pilot scenario set through the certified pipeline, appends every rollout to a JSONL bank (checkpointing as it
goes), then evaluates the Phase-7 pilot gates and prints the honest verdict. Generation only — no BC/RL/refinement. The
verdict is ``R11_3_COIN_TARGET_DEMONSTRATION_PIPELINE_PASS`` iff every gate holds; the teacher-handoff coverage (K6 rate,
handoff-invalid count) is reported alongside but does not by itself fail the pilot.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from hymeko_rl.coin_delivery.demo_bank import (
    DemonstrationBank,
    DemonstrationRecord,
    build_pilot_scenarios,
    replay_matches,
    run_scenario,
    split_ids,
)
from hymeko_rl.coin_delivery.demo_bank.scenario import CoinTargetScenario, ScenarioSplit
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

DEFAULT_BANK = Path("reports/2026-07-29-r11-3-coin-target-demo-pilot/pilot_bank.jsonl")
PASS_VERDICT = "R11_3_COIN_TARGET_DEMONSTRATION_PIPELINE_PASS"


def run_pilot(bank_path: Path, scenarios: "list[CoinTargetScenario] | None" = None, seed: int = 0) -> DemonstrationBank:
    """Generate the pilot bank. Each scenario is appended as it completes, so a partial run is recoverable."""
    rig = _rig()
    if bank_path.exists():
        bank_path.unlink()
    bank = DemonstrationBank(bank_path)
    scenarios = scenarios if scenarios is not None else build_pilot_scenarios()
    for i, sc in enumerate(scenarios, start=1):
        t0 = time.perf_counter()
        rec = run_scenario(rig, sc, seed=seed)
        bank.append(rec)
        print(f"[{i:2d}/{len(scenarios)}] {sc.scenario_id:16s} {rec.curriculum_stage:7s} adm={rec.admissible!s:5s} "
              f"-> {rec.outcome_label:26s} k6={rec.k6!s:5s} dtz={rec.min_dtz_mm} "
              f"handoff_adm={rec.handoff_admissible!s:5s} link={rec.k6_provenance_link!s:5s} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
    return bank


def _disjoint(scenarios: "list[CoinTargetScenario]") -> bool:
    by = split_ids(scenarios)
    tr, dv, te = by[ScenarioSplit.TRAIN], by[ScenarioSplit.DEV], by[ScenarioSplit.TEST]
    return tr.isdisjoint(dv) and tr.isdisjoint(te) and dv.isdisjoint(te)


def _start_gates(records: "list[DemonstrationRecord]", invalids: "list[DemonstrationRecord]",
                 scenarios: "list[CoinTargetScenario]") -> dict[str, bool]:
    return {
        "ALL_VALID_ROLLOUTS_EXACT_ZERO_CERTIFICATE": all(r.ic_valid for r in records),
        "INVALID_STARTS_REJECTED_BEFORE_PLANNING": all((not r.admissible) and (not r.reach_found) for r in invalids),
        "TRAIN_DEV_TEST_IDS_DISJOINT": _disjoint(scenarios),
    }


def _reach_gates(accepted: "list[DemonstrationRecord]") -> dict[str, bool]:
    return {
        "ACCEPTED_REACH_COIN_MOTION_LE_1MM":
            all(r.reach is not None and r.reach["coin_moved_before_capture_mm"] <= 1.0 for r in accepted),
        "NO_ACCEPTED_REACH_PREMATURE_CONTACT": all(r.premature_contacts == 0 for r in accepted),
        "HANDOFF_DESCRIPTORS_COMPLETE": all(r.handoff_complete for r in accepted),
    }


def _trace_energy_gates(adm: "list[DemonstrationRecord]", accepted: "list[DemonstrationRecord]") -> dict[str, bool]:
    return {
        "MODE_TRACES_VALID_AND_COMPLETE": all(len(r.mode_trace) >= 3 and r.mode_trace[0] == 0 for r in adm),
        "ENERGY_LEDGERS_COMPLETE": all(r.energy_measurement_complete for r in accepted),
        "ENERGY_RESIDUALS_RECORDED": all("ENERGY_BALANCE_RESIDUAL_RECORDED" in r.energy_verdicts for r in accepted),
    }


def _provenance_gates(records: "list[DemonstrationRecord]", accepted: "list[DemonstrationRecord]",
                      replay_ok: bool) -> dict[str, bool]:
    return {
        "ALL_TEACHER_CALLS_IN_PROVENANCE": all(r.teacher_identity != "none" for r in accepted),
        "SUCCESSFUL_K6_LINKS_TO_PROVENANCE_HASH": all(r.k6_provenance_link for r in records if r.is_success),
        "SERIALIZATION_ROUNDTRIP_PRESERVES_RECORD":
            all(DemonstrationRecord.from_dict(r.to_dict()) == r for r in records),
        "REPLAY_REPRODUCES_SCENARIO_AND_OUTCOME": replay_ok,
        "NO_TEACHER_ROLLOUT_LABELLED_TEACHER_FREE": all(r.teacher_identity != "TEACHER_FREE" for r in records),
    }


def evaluate_gates(rig: Any, records: "list[DemonstrationRecord]",
                   scenarios: "list[CoinTargetScenario]") -> dict[str, bool]:
    """The Phase-7 pilot gates over the generated bank + one deterministic replay (an admissible + the invalid path)."""
    adm = [r for r in records if r.admissible and r.split != ScenarioSplit.INVALID.value]
    accepted = [r for r in adm if r.reach_found and r.reach is not None]
    invalids = [r for r in records if r.split == ScenarioSplit.INVALID.value]
    by_id = {s.scenario_id: s for s in scenarios}
    replay_ok = _replay_gate(rig, records, by_id)
    return {**_start_gates(records, invalids, scenarios), **_reach_gates(accepted),
            **_trace_energy_gates(adm, accepted), **_provenance_gates(records, accepted, replay_ok)}


def _replay_gate(rig: Any, records: "list[DemonstrationRecord]", by_id: dict) -> bool:
    """Re-run one admissible scenario (a K6 success if any — fast, early-exit) + the first invalid, and require the replay
    to reproduce the record (deterministic content hash + outcome class)."""
    checks = []
    admissible = [r for r in records if r.admissible and r.reach_found]
    invalid = [r for r in records if not r.admissible]
    # prefer a success (early-exit -> fast), else any admissible
    admissible.sort(key=lambda r: (not r.is_success, r.scenario_id))
    targets = (admissible[:1]) + invalid[:1]
    for rec in targets:
        rerun = run_scenario(rig, by_id[rec.scenario_id])
        checks.append(replay_matches(rec, rerun))
    return bool(checks) and all(checks)


def _coverage(records: "list[DemonstrationRecord]") -> str:
    adm = [r for r in records if r.admissible and r.split != ScenarioSplit.INVALID.value]
    k6 = [r for r in adm if r.k6]
    handoff_bad = [r for r in adm if r.outcome_label == "precontact_handoff_invalid"]
    rate = 0.0 if not adm else len(k6) / len(adm)
    return f"K6 {len(k6)}/{len(adm)} ({rate:.2f}); handoff_invalid={len(handoff_bad)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--smoke", action="store_true", help="run only the canonical scenario (production-scale smoke)")
    args = ap.parse_args()

    scenarios = build_pilot_scenarios()
    if args.smoke:
        scenarios = [s for s in scenarios if s.curriculum_stage == "C0"]
        args.bank = args.bank.with_name("smoke_bank.jsonl")

    t0 = time.perf_counter()
    bank = run_pilot(args.bank, scenarios)
    records = bank.read()
    s = bank.summarize()
    print(f"\n=== PILOT SUMMARY ({time.perf_counter() - t0:.1f}s) ===", flush=True)
    print(f"admissible={s.admissible} successes={s.successes} success_rate={s.success_rate:.3f}", flush=True)
    print(f"coverage: {_coverage(records)}", flush=True)
    print(f"failures_by_class={s.failures_by_class}", flush=True)
    print(f"rejected={s.rejected} rejection_reasons={s.rejection_reasons}", flush=True)
    if args.smoke:
        print("SMOKE_DONE", flush=True)
        return
    gates = evaluate_gates(_rig(), records, scenarios)
    print(f"gates={json.dumps(gates)}", flush=True)
    verdict = PASS_VERDICT if all(gates.values()) else "R11_3_PILOT_GATE_FAILURE"
    print(f"VERDICT={verdict}", flush=True)
    print("PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
