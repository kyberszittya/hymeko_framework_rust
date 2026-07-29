"""R11.4A0 derived reclassification of the R11.3 demonstration bank under the corrected capture->delivery contract.

Reads the existing bank (does NOT rewrite it) and emits a versioned derived label per attempt + per scenario-best, using
the measured proxy ``contacts == 2 <=> grasped handoff <=> delivery mode`` (verified on the exact-replay set). Recomputes
how many of the former 8/64 K6 were valid-delivery-mode vs nudge-only (K6_WITHOUT_DELIVERY_MODE_TRANSITION).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from hymeko_rl.coin_delivery.delivery_teacher.delivery_contract import (
    DeliveryOutcomeClass,
    reclassify_from_bank_fields,
)

DERIVED_VERSION = "r11.4a0-derived-v1"
DEFAULT_BANK = Path("reports/2026-07-30-r11-3-coin-target-demo-bank/bank.jsonl")


def _label(r: dict[str, Any]) -> str:
    return reclassify_from_bank_fields(contacts=int(r["contacts"]), k6=bool(r["k6"]),
                                       min_dtz_mm=r["min_dtz_mm"]).value


def _best_per_scenario(admissible: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in admissible:
        by.setdefault(r["scenario_id"], []).append(r)
    best = []
    for rs in by.values():
        k6s = [r for r in rs if r["k6"]]
        best.append(min(k6s, key=lambda r: r["teacher_seed"]) if k6s
                    else min(rs, key=lambda r: (r["min_dtz_mm"] if r["min_dtz_mm"] is not None else 1e9)))
    return best


def _k6_breakdown(best: list[dict[str, Any]]) -> "tuple[int, int]":
    """(former K6 scenarios, of which valid-delivery-mode)."""
    k6 = [r for r in best if r["k6"]]
    valid = sum(1 for r in k6 if _label(r) == DeliveryOutcomeClass.K6_WITH_VALID_DELIVERY_MODE.value)
    return len(k6), valid


def reclassify(bank_path: Path) -> dict[str, Any]:
    recs = [json.loads(line) for line in bank_path.open() if line.strip()]
    adm = [r for r in recs if r["admissible"] and r["split"] != "invalid"]
    best = _best_per_scenario(adm)
    attempt_counts = Counter(_label(r) for r in adm)
    best_counts = Counter(_label(r) for r in best)
    n_k6, k6_valid = _k6_breakdown(best)
    return {
        "derived_version": DERIVED_VERSION,
        "proxy": "contacts==2 <=> grasped handoff <=> delivery-mode (verified on exact-replay set)",
        "note": "derived labels only; the R11.3 bank is NOT rewritten",
        "n_attempts": len(adm), "n_scenarios": len(best),
        "attempt_level_counts": dict(attempt_counts), "scenario_best_counts": dict(best_counts),
        "former_k6_scenarios": n_k6,
        "k6_with_valid_delivery_mode": k6_valid,
        "k6_without_delivery_mode_transition_nudge": n_k6 - k6_valid,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/2026-07-30-r11-4a0-capture-to-delivery-audit/derived_reclassification.json"))
    args = ap.parse_args()
    result = reclassify(args.bank)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print("RECLASSIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
