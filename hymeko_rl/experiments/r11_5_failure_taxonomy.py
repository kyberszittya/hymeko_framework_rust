"""R11.5 complete failure taxonomy — assign every uncovered scenario exactly one mutually-exclusive category.

The full-51 coverage run left 24/64 scenarios uncovered. The coverage report highlighted only 9 of them (capture +
far-tail + zone-near); the R11.5+ recovery design must not be built on a partial taxonomy, so this module classifies
ALL 24 from their measured delivery traces (not guesses). Categories are trace-derived and mutually exclusive.

The 24 uncovered = the 6 baseline CAPTURE_FAIL (never certified in the R11.4A re-measure, never attempted) + the 18
non-recovered attempts (4 that failed to re-certify a grasp in 5 seeds + 14 certified-but-not-delivered).

Discriminators (all present in the coverage ledger):
  * ``certified`` — a certified bilateral grasp was (re)formed for the delivery teacher at all.
  * ``gap_closed`` — fraction of the entry coin→target gap the delivery closed; **sign** separates "moved toward
    target" (>=0) from "driven away / off-axis" (<0).
  * ``teacher_dtz_mm`` — final coin→target distance vs the K6 zone tolerance (``CENTER_TOL`` = 20 mm).
  * ``target_entry_speed`` vs ``SETTLE_VEL`` (0.06) — whether a zone entry was too fast to settle.
  * ``coin_progress_mm`` — net forward displacement; ~0 with ~0 gap change means the kinetic handoff never moved the coin.
"""
from __future__ import annotations

import dataclasses
import enum
import json
from collections import defaultdict
from pathlib import Path

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL, SETTLE_VEL
from hymeko_rl.experiments.r11_5_full_coverage import _cclass, _scenario_kind

ZONE_TOL_MM = CENTER_TOL * 1000.0                    # 20 mm — K6 dtz gate
_STILL_MM = 5.0                                      # |progress| below this with ~0 gap change = coin never moved
_STILL_GAP = 0.05


class FailureCategory(enum.Enum):
    """Mutually-exclusive R11.5 failure categories. Ordering is the classifier's precedence."""
    CAPTURE_SUPPORT_FAILURE = "CAPTURE_SUPPORT_FAILURE"          # no certified grasp to hand to the delivery teacher
    HANDOFF_TO_KINETIC_FAILURE = "HANDOFF_TO_KINETIC_FAILURE"    # certified, but the coin never got moving
    DIRECTIONAL_BIAS = "DIRECTIONAL_BIAS"                        # driven, but net AWAY from target (gap_closed < 0)
    ZONE_NEAR_SPEED_FAILURE = "ZONE_NEAR_SPEED_FAILURE"          # reached the zone but too fast to settle
    ZONE_ENTRY_WITHOUT_DWELL = "ZONE_ENTRY_WITHOUT_DWELL"        # entered the zone, slow, but no held dwell
    INSUFFICIENT_PROGRESS = "INSUFFICIENT_PROGRESS"             # moved toward target, closed part of the gap, stalled short
    CONTACT_LOSS = "CONTACT_LOSS"                                # premature grasp loss as the sole measured cause
    OTHER_MEASURED = "OTHER_MEASURED"                            # only with a concrete trace reason


def classify(certified: bool, gap_closed: float, dtz_mm: float,
             entry_speed: float, coin_progress_mm: float) -> FailureCategory:
    """Assign one category from measured delivery outcome.

    Preconditions: ``dtz_mm >= 0``; ``gap_closed``/``entry_speed``/``coin_progress_mm`` finite. A grasp that never
    certified is CAPTURE_SUPPORT_FAILURE regardless of the (absent) delivery metrics.
    Postcondition: returns exactly one ``FailureCategory``; ``CONTACT_LOSS``/``OTHER_MEASURED`` are reachable only by an
    explicit trace override, never inferred here (``contact_lost_steps`` is kinetic-normal, not discriminative).
    """
    assert dtz_mm >= 0.0, "dtz must be non-negative"
    if not certified:
        return FailureCategory.CAPTURE_SUPPORT_FAILURE
    if abs(coin_progress_mm) < _STILL_MM and abs(gap_closed) < _STILL_GAP:
        return FailureCategory.HANDOFF_TO_KINETIC_FAILURE
    if gap_closed < 0.0:
        return FailureCategory.DIRECTIONAL_BIAS
    if dtz_mm <= ZONE_TOL_MM:
        return (FailureCategory.ZONE_NEAR_SPEED_FAILURE if entry_speed > SETTLE_VEL
                else FailureCategory.ZONE_ENTRY_WITHOUT_DWELL)
    return FailureCategory.INSUFFICIENT_PROGRESS


@dataclasses.dataclass(frozen=True)
class FailureRecord:
    scenario_id: str
    split: str
    cclass: str
    category: FailureCategory
    subtype: str                    # capture: "systematic_pp" | "stochastic_regen"; else ""
    dtz_mm: float | None
    gap_closed: float | None
    coin_progress_mm: float | None


def _bank_class(bank_path: Path) -> "tuple[set[str], set[str], dict[str, str]]":
    """(delivery-fail ids, baseline-capture-fail ids, split-by-id) from the R11.4A bank."""
    by: dict[str, list[dict[str, object]]] = defaultdict(list)
    for x in bank_path.open():
        if x.strip():
            r = json.loads(x)
            by[r["scenario_id"]].append(r)
    fails: set[str] = set()
    cap_fail: set[str] = set()
    split: dict[str, str] = {}
    for sid, rs in by.items():
        split[sid] = str(next((r.get("split") for r in rs if r.get("split")), "?"))
        kind = _scenario_kind(sid, rs)
        if kind and kind[0] == "fail":
            fails.add(sid)
        elif kind is None:                      # no certified record at all = baseline capture failure
            cap_fail.add(sid)
    return fails, cap_fail, split


def build_taxonomy(coverage_path: Path, bank_path: Path) -> list[FailureRecord]:
    """The complete 24-scenario taxonomy. Postcondition: every uncovered scenario appears exactly once."""
    cov = {json.loads(ln)["scenario_id"]: json.loads(ln) for ln in coverage_path.open() if ln.strip()}
    _fails, cap_fail, split = _bank_class(bank_path)
    out: list[FailureRecord] = []
    for sid in sorted(cap_fail):                # A) baseline capture failures — systematic (all +/+), never attempted
        out.append(FailureRecord(sid, split.get(sid, "?"), _cclass(sid),
                                 FailureCategory.CAPTURE_SUPPORT_FAILURE, "systematic_pp", None, None, None))
    for sid, r in sorted(cov.items()):
        if r.get("recovered"):
            continue
        if r.get("certified") is False:         # B) certified in re-measure, failed to re-certify in 5 seeds
            out.append(FailureRecord(sid, r["split"], _cclass(sid),
                                     FailureCategory.CAPTURE_SUPPORT_FAILURE, "stochastic_regen", None, None, None))
        elif r.get("certified"):                # C) certified-but-not-delivered — classify from the trace
            cat = classify(True, r["gap_closed"], r["teacher_dtz_mm"], r["target_entry_speed"], r["coin_progress_mm"])
            out.append(FailureRecord(sid, r["split"], _cclass(sid), cat, "",
                                     r["teacher_dtz_mm"], r["gap_closed"], r["coin_progress_mm"]))
    return out


def _capture_pick(records: list[FailureRecord]) -> list[FailureRecord]:
    """2 systematic-+/+ (hardest, never certified) + 2 stochastic-regen (likeliest to recover) capture-support cases."""
    cap = [r for r in records if r.category is FailureCategory.CAPTURE_SUPPORT_FAILURE]
    syst = sorted((r for r in cap if r.subtype == "systematic_pp"), key=lambda r: r.scenario_id)
    stoch = sorted((r for r in cap if r.subtype == "stochastic_regen"), key=lambda r: r.scenario_id)
    return syst[:2] + stoch[:2]


def _dedup(records: list[FailureRecord]) -> list[FailureRecord]:
    seen: set[str] = set()
    out: list[FailureRecord] = []
    for r in records:
        if r.scenario_id not in seen:
            seen.add(r.scenario_id)
            out.append(r)
    return out


def _progress_pick(records: list[FailureRecord]) -> list[FailureRecord]:
    """4 INSUFFICIENT_PROGRESS cases. Held-out first (one dev + one test if the group has them — the transport fix must
    be validated off-train, since the DIRECTIONAL_BIAS tail is structurally all-train), then the dtz range (nearest +
    farthest) to span the stall-distance spectrum."""
    by_dtz = sorted((r for r in records if r.category is FailureCategory.INSUFFICIENT_PROGRESS),
                    key=lambda r: (r.dtz_mm if r.dtz_mm is not None else 0.0))
    heldout = [c for want in ("dev", "test")
               if (c := next((r for r in by_dtz if r.split == want), None)) is not None]
    return _dedup(heldout + [by_dtz[0], by_dtz[-1]] + by_dtz)[:4]


def select_pilot(records: list[FailureRecord]) -> list[FailureRecord]:
    """Bounded 12-scenario recovery pilot: 4 capture-support (systematic-+/+ and stochastic-regen), 4 DIRECTIONAL_BIAS
    (the whole negative-x tail), 4 representative INSUFFICIENT_PROGRESS. Postcondition: exactly 12 records, collectively
    including >=1 dev and >=1 test scenario."""
    capture = _capture_pick(records)
    directional = sorted((r for r in records if r.category is FailureCategory.DIRECTIONAL_BIAS),
                         key=lambda r: r.scenario_id)[:4]
    progress = _progress_pick(records)
    return capture + directional + progress


def _summarize(records: list[FailureRecord]) -> "dict[str, list[str]]":
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in records:
        by_cat[r.category.value].append(r.scenario_id)
    return {k: sorted(v) for k, v in sorted(by_cat.items())}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=Path, default=Path("reports/2026-07-30-r11-5-coverage/coverage.jsonl"))
    ap.add_argument("--bank", type=Path, default=Path("reports/2026-07-30-r11-4a-bank/bank.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports/2026-07-30-r11-5-coverage/taxonomy.json"))
    args = ap.parse_args()
    recs = build_taxonomy(args.coverage, args.bank)
    pilot = select_pilot(recs)
    summary = _summarize(recs)
    print(f"=== R11.5 COMPLETE FAILURE TAXONOMY ({len(recs)} uncovered) ===")
    for cat, ids in summary.items():
        print(f"  {cat} ({len(ids)}): {', '.join(ids)}")
    print(f"\n=== BOUNDED PILOT ({len(pilot)}) ===")
    for r in pilot:
        print(f"  {r.scenario_id:26s} {r.split:5s} {r.category.value:26s} {r.subtype}")
    payload = {"uncovered": len(recs), "by_category": summary,
               "records": [dataclasses.asdict(r) | {"category": r.category.value} for r in recs],
               "pilot": [{"scenario_id": r.scenario_id, "split": r.split, "category": r.category.value,
                          "subtype": r.subtype} for r in pilot]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
