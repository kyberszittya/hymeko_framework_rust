"""R11.6C — exact-zero retrieval composition run. For each panel scenario: run the full exact-zero -> RRT reach ->
capture -> LIVE descriptor chain ONCE, then deliver with BOTH frozen configs (primary std_weighted3, control
std_nearest) reusing the single handoff. Classify the full chain, emit the first ExactZeroCoinDeliveryCertificate, and
apply the composability-first gate. The TEST split is sealed (not evaluated here). Parallel via --offset/--limit.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, fresh_rig, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.retrieval import RetrievalConfig, load_frozen
from hymeko_rl.coin_delivery.exact_zero_composition import (
    SUCCESS,
    CompositionOutcomeClass,
    deliver_record,
    reach_capture_descriptor,
)
from hymeko_rl.experiments.r11_4b_conditioned_bc import _load_dataset

FROZEN = Path("reports/2026-08-06-r11-5r-retrieval/frozen_policy.json")
B1_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
DEFAULT_OUT = Path("reports/2026-08-06-r11-6c-composition")
PARKED = ["bank_c2_+0.015_+0.015", "bank_c3_r9_a-45"]        # the 2 parked-stress scenarios (no stored theta)
_NUDGE = CompositionOutcomeClass.NUDGE_NOT_VALID_DELIVERY.value
_SAFETY = CompositionOutcomeClass.SAFETY_FAILURE.value


def _panel(dataset_dir: Path) -> "list[tuple[str, str, int, np.ndarray | None]]":
    """(group, scenario_id, seed, stored_x). train-like + dev from the robust bank (stored x for the drift check); the 2
    parked-stress scenarios at seed 0 with no stored x. The TEST split is excluded (sealed)."""
    samples = _load_dataset(dataset_dir)
    panel: list[tuple[str, str, int, np.ndarray | None]] = [
        ("train-like" if s.split == "train" else "dev", s.scenario_id, int(s.seed), np.asarray(s.x, np.float64))
        for s in samples if s.split in ("train", "dev")]
    panel += [("parked", pid, 0, None) for pid in PARKED]
    return panel


def _slice(seq: list, offset: int, limit: int) -> list:
    return seq[offset:(offset + limit if limit else len(seq))]


def _run_eval(args: argparse.Namespace) -> None:
    fp = json.loads(args.frozen.read_text())
    primary = load_frozen(fp["table"])                                           # primary config stored in the table
    control = load_frozen(fp["table"], config=RetrievalConfig.from_json(fp["control_config"]))
    radius = round(primary.table_coverage_radius(95.0), 4)
    cfg, conf, obj = bc_context()
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for group, sid, seed, x in _slice(_panel(args.dataset_dir), args.offset, args.limit):
        scen = scenario_by_id(sid)
        h = reach_capture_descriptor(fresh_rig(), scen, seed, cfg, conf, obj, x)
        if h.record is not None:                                                  # failed before delivery -> same for both
            prim = ctrl = h.record
        else:
            prim = deliver_record(scen, seed, h.snap, h.x, primary, radius)
            ctrl = deliver_record(scen, seed, h.snap, h.x, control, radius)
        rows.append({"group": group, "radius": radius, "primary": prim.__dict__, "control": ctrl.__dict__})
        print(f"{group:10s} {sid:26s} primary={prim.outcome_class:28s} control={ctrl.outcome_class}", flush=True)
    with (args.out / f"comp_{args.offset:03d}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


_GROUPS = ("train-like", "dev", "parked")


def _k6_rate(recs: "list[dict]") -> float:
    return round(sum(1 for p in recs if p["k6"]) / len(recs), 3) if recs else 0.0


def _success_rate(recs: "list[dict]") -> float:
    return round(sum(1 for p in recs if p["outcome_class"] == SUCCESS.value) / len(recs), 3) if recs else 0.0


def _count(recs: "list[dict]", cls: str) -> int:
    return sum(1 for p in recs if p["outcome_class"] == cls)


def _taxonomy(recs: "list[dict]") -> "dict[str, int]":
    return dict(Counter(p["outcome_class"] for p in recs))


def _map(by_group: "dict[str, list]", fn: Any) -> "dict[str, Any]":
    return {g: fn(rs) for g, rs in by_group.items()}


def _arm_gate(rows: "list[dict]", arm: str) -> "dict[str, Any]":
    recs = [r[arm] for r in rows]
    by_group = {g: [r[arm] for r in rows if r["group"] == g] for g in _GROUPS}
    certs = [p["certificate"] for p in recs if p["certificate"]]
    return {"success_rate": _map(by_group, _success_rate), "k6_rate": _map(by_group, _k6_rate),
            "taxonomy": _map(by_group, _taxonomy), "n_certificates": len(certs),
            "n_nudge": _count(recs, _NUDGE), "n_safety_failure": _count(recs, _SAFETY),
            "first_certificate": certs[0] if certs else None}


def _verdict(g: "dict[str, Any]") -> str:
    ok = (g["success_rate"]["train-like"] >= 0.50 and g["success_rate"]["dev"] > 0.0
          and g["n_certificates"] >= 1 and g["n_safety_failure"] == 0)
    return "R11_6C_EXACT_ZERO_RETRIEVAL_COMPOSITION_PASS" if ok else "R11_6C_COMPOSITION_INSUFFICIENT"


def gate(rows: "list[dict]") -> "dict[str, Any]":
    """Composability-first gate on the PRIMARY arm; the control arm is reported alongside."""
    prim, ctrl = _arm_gate(rows, "primary"), _arm_gate(rows, "control")
    return {"n": len(rows), "radius": rows[0]["radius"] if rows else None,
            "primary": {**prim, "verdict": _verdict(prim)}, "control": ctrl,
            "verdict": _verdict(prim)}


def _run_merge(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(args.out / "comp_*.jsonl"))):
        rows += [json.loads(line) for line in Path(f).open() if line.strip()]
    g = gate(rows)
    (args.out / "composition.json").write_text(json.dumps({"gate": g, "rows": rows}, indent=2), encoding="utf-8")
    print("=== R11.6C EXACT-ZERO COMPOSITION GATE ===", flush=True)
    print(json.dumps({"verdict": g["verdict"], "primary_success": g["primary"]["success_rate"],
                      "primary_k6": g["primary"]["k6_rate"], "n_cert": g["primary"]["n_certificates"],
                      "n_safety": g["primary"]["n_safety_failure"], "taxonomy": g["primary"]["taxonomy"]}, indent=2),
          flush=True)
    print("R11_6C_COMPOSITION_DONE", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("eval", "merge"), required=True)
    ap.add_argument("--frozen", type=Path, default=FROZEN)
    ap.add_argument("--dataset-dir", type=Path, default=B1_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"eval": _run_eval, "merge": _run_merge}[args.phase](args)


if __name__ == "__main__":
    main()
