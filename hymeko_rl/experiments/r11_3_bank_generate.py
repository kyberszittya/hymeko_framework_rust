"""R11.3 bounded coin/target demonstration-bank generator (Phase 8) — runs ONLY after the pilot passes.

Generates the bounded overnight bank: 64 candidate scenarios (C0/C1/C2/C3, geometric-cell 70/15/15 splits), at most 3
teacher seeds per scenario with early-exit on the first strict-K6 (<=192 teacher attempts total). Every attempt is retained
with its true label; the first K6 per scenario is the preferred positive (derived at analysis time as the min-seed K6, not
overwritten). Deployed RRT reach + CEM training teacher (labelled in provenance); no BC/RL/refinement. Checkpointed per
attempt to JSONL so a partial run is recoverable.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from hymeko_rl.coin_delivery.demo_bank import DemonstrationBank, PipelineConfig, build_bank_scenarios, run_scenario
from hymeko_rl.coin_delivery.demo_bank.scenario import CoinTargetScenario
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

DEFAULT_BANK = Path("reports/2026-07-30-r11-3-coin-target-demo-bank/bank.jsonl")
SEEDS = (0, 1, 2)
_SINGLE_SEED = PipelineConfig(teacher_budget=1)   # one CEM seed per attempt so each retained record is one teacher seed


def generate_bank(bank_path: Path, scenarios: "list[CoinTargetScenario] | None" = None) -> DemonstrationBank:
    """Run the bank: per scenario, up to 3 teacher seeds, early-exit on the first K6, retaining every attempt."""
    rig = _rig()
    if bank_path.exists():
        bank_path.unlink()
    bank = DemonstrationBank(bank_path)
    scenarios = scenarios if scenarios is not None else build_bank_scenarios()
    attempts = 0
    for i, sc in enumerate(scenarios, start=1):
        for seed in SEEDS:
            t0 = time.perf_counter()
            rec = run_scenario(rig, sc, seed=seed, config=_SINGLE_SEED)
            bank.append(rec)
            attempts += 1
            k6 = rec.k6
            print(f"[{i:2d}/{len(scenarios)} s{seed}] {sc.scenario_id:22s} {rec.curriculum_stage:7s} "
                  f"{rec.outcome_label:26s} k6={k6!s:5s} dtz={rec.min_dtz_mm} ({time.perf_counter() - t0:.1f}s) "
                  f"[att {attempts}]", flush=True)
            if not rec.admissible or k6:                          # early-exit: first K6 (or an inadmissible scenario)
                break
    print(f"\nBANK_DONE attempts={attempts}", flush=True)
    return bank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N scenarios only (0 = all 64)")
    args = ap.parse_args()
    scenarios = build_bank_scenarios()
    if args.limit > 0:
        scenarios = scenarios[:args.limit]
    t0 = time.perf_counter()
    bank = generate_bank(args.bank, scenarios)
    s = bank.summarize()
    print(f"=== BANK SUMMARY ({time.perf_counter() - t0:.1f}s) ===", flush=True)
    print(f"records={len(bank.read())} admissible_denominator={s.admissible} successes={s.successes} "
          f"success_rate={s.success_rate:.3f}", flush=True)
    print(f"failures_by_class={s.failures_by_class}", flush=True)
    print(f"rejected={s.rejected} rejection_reasons={s.rejection_reasons}", flush=True)


if __name__ == "__main__":
    main()
