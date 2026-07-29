"""Phase 1 — complete R11.5+ residual taxonomy over all 24 uncovered scenarios.

The 24 = 6 baseline CAPTURE_FAIL (never certified in the R11.4A re-measure) + 4 that failed to re-certify a grasp in
5 seeds + 14 certified-but-not-delivered. The certified-not-delivered are enriched by DETERMINISTIC REPLAY (recorded
capture_seed + theta) to recover trajectory features the coverage ledger lacks (trajectory-min dtz, max K6 dwell, zone
entry, pre-release contact loss), because ``teacher_dtz_mm`` is ``dtz_end`` (final), not the minimum. Emits the report +
JSON and the gate ``R11_5_PLUS_RESIDUAL_TAXONOMY_COMPLETE`` (counts sum to exactly 24; mutually exclusive).

Deterministic; CPU-bound MuJoCo (~70-90 s per certified-not-delivered replay). Provenance: energy diagnostic, no state edit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.coin_rl_env import CENTER_TOL
from hymeko_rl.coin_delivery.delivery_teacher.solver import _config, full_transport_spec
from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
from hymeko_rl.coin_delivery.demo_bank import pipeline as P
from hymeko_rl.coin_delivery.contact_velocity import primary_fingertip_contacts
from hymeko_rl.coin_delivery.forward_displacement import _coin_speed, rollout_primitive
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.moving_precapture import GraspObjective, is_certified_grasp
from hymeko_rl.experiments import coin_zero_home_reach as Z
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
from hymeko_rl.experiments.r11_5_failure_taxonomy import (
    DeliveryTrace,
    FailureCategory,
    FailureRecord,
    cclass,
    classify,
    select_pilot,
    summarize,
)
from hymeko_rl.experiments.r11_5_full_coverage import _scenario_kind

R11_4A_BANK = Path("reports/2026-07-30-r11-4a-bank/bank.jsonl")
COVERAGE = Path("reports/2026-07-30-r11-5-coverage/coverage.jsonl")
DEFAULT_OUT = Path("reports/2026-07-31-r11-5-plus-residual-taxonomy.json")


def _bank_by_id(bank: Path) -> "dict[str, list[dict[str, Any]]]":
    by: dict[str, list[dict[str, Any]]] = {}
    for x in bank.open():
        if x.strip():
            r = json.loads(x)
            by.setdefault(r["scenario_id"], []).append(r)
    return by


def _split_of(rs: "list[dict[str, Any]]") -> str:
    return str(next((r.get("split") for r in rs if r.get("split")), "?"))


def _uncovered(coverage: Path, bank: Path) -> "tuple[list[tuple[str, str]], list[dict[str, Any]]]":
    """(baseline-capture-fail [(sid, split)], non-recovered coverage rows) — the 24 uncovered scenarios."""
    by = _bank_by_id(bank)
    cap_fail = [(sid, _split_of(rs)) for sid, rs in by.items() if _scenario_kind(sid, rs) is None]
    rows = [json.loads(x) for x in coverage.open() if x.strip()]
    return sorted(cap_fail), [r for r in rows if not r.get("recovered")]


def _replay(rig: "dict[str, Any]", cfg_tc: Any, obj: GraspObjective, conf: PipelineConfig, scen: Any,
            capture_seed: int, theta: np.ndarray) -> "dict[str, Any] | None":
    """Deterministic replay of a certified-not-delivered scenario -> trajectory features; None if the grasp won't re-form."""
    home, coin = Z._home_with_coin(rig, scen.coin_xy)
    _r, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg_tc, conf, capture_seed)
    if rc is None or not is_certified_grasp(rc.result.outcome, obj):
        return None
    traj: list[tuple[float, float, bool]] = []

    def hook(rl: Any, _t: int, _traj: "list[tuple[float, float, bool]]" = traj) -> None:
        _u, dtz = rl.inner.direction_to_zone()
        con = primary_fingertip_contacts(rl)
        _traj.append((float(dtz), float(_coin_speed(rl)), con["left"] is not None and con["right"] is not None))

    m = rollout_primitive(rc.result.outcome.snapshot, theta, _config(full_transport_spec()), frame_hook=hook)
    dtzs = [d for d, _, _ in traj]
    imin = int(np.argmin(dtzs))
    entry = next((i for i, (d, _, _) in enumerate(traj) if d <= CENTER_TOL), None)
    return {"final_dtz_mm": round(m["dtz_end"] * 1000, 2), "min_dtz_mm": round(dtzs[imin] * 1000, 2),
            "max_coin_progress_mm": round(m["forward"] * 1000, 1), "gap_closed_final": round(m["gap_closed"], 3),
            "kinetic_reached": bool(m["touched"]), "max_k6_dwell": int(m["k6_max_dwell"]),
            "zone_entry_step": entry, "entry_speed": round(traj[entry][1] if entry is not None else traj[imin][1], 4),
            "lost_before_release": int(m["lost_before_release"]), "release_step": int(round(float(theta[4])))}


def _rel_vec(scen: Any) -> "list[float]":
    return [round(float(scen.target_xy[0] - scen.coin_xy[0]), 4), round(float(scen.target_xy[1] - scen.coin_xy[1]), 4)]


def _panel(scen: Any, split: str, certified: bool, existing: str, feat: "dict[str, Any] | None") -> "dict[str, Any]":
    """The full per-scenario field panel (poses, relative target, measured trajectory, existing class)."""
    base = {"scenario_id": scen.scenario_id, "curriculum_stage": cclass(scen.scenario_id), "split": split,
            "coin_pose": [round(float(v), 4) for v in scen.coin_xy], "target_pose": [round(float(v), 4) for v in scen.target_xy],
            "relative_target_vector": _rel_vec(scen), "certified_grasp": certified, "first_successful_restart": None,
            "existing_failure_class": existing}
    return base | (feat or {"kinetic_reached": None, "min_dtz_mm": None, "final_dtz_mm": None,
                            "max_coin_progress_mm": None, "gap_closed_final": None, "zone_entry_step": None,
                            "max_k6_dwell": None, "entry_speed": None})


def build_records(coverage: Path, bank: Path, rig: "dict[str, Any]", cfg_tc: Any, obj: GraspObjective,
                  conf: PipelineConfig) -> "tuple[list[FailureRecord], list[dict[str, Any]]]":
    scens = {s.scenario_id: s for s in build_bank_scenarios()}
    cap_fail, non_rec = _uncovered(coverage, bank)
    records: list[FailureRecord] = []
    panels: list[dict[str, Any]] = []
    for sid, split in cap_fail:                              # A) systematic +/+ baseline capture failures
        scen = scens[sid]
        rec = FailureRecord(sid, split, cclass(sid), FailureCategory.CAPTURE_SUPPORT_FAILURE, "systematic_pp", None)
        records.append(rec)
        panels.append(_panel(scen, split, False, "CAPTURE_FAIL", None) | {"residual_class": rec.category.value,
                                                                          "capture_subtype": "systematic_pp"})
    for r in sorted(non_rec, key=lambda z: z["scenario_id"]):
        sid, split, scen = r["scenario_id"], r["split"], scens[r["scenario_id"]]
        if r.get("certified") is False:                     # B) stochastic-regen capture failures
            rec = FailureRecord(sid, split, cclass(sid), FailureCategory.CAPTURE_SUPPORT_FAILURE, "stochastic_regen", None)
            records.append(rec)
            panels.append(_panel(scen, split, False, "DELIVERY_FAILURE", None)
                          | {"residual_class": rec.category.value, "capture_subtype": "stochastic_regen"})
            continue
        feat = _replay(rig, cfg_tc, obj, conf, scen, int(r["capture_seed"]), np.array(r["theta"], np.float64))  # C)
        trace = (DeliveryTrace(True, feat["max_coin_progress_mm"], feat["gap_closed_final"], feat["min_dtz_mm"],
                               feat["max_k6_dwell"], feat["entry_speed"], feat["lost_before_release"], feat["release_step"])
                 if feat else DeliveryTrace(False))
        cat = classify(trace) if feat else FailureCategory.CAPTURE_SUPPORT_FAILURE
        rec = FailureRecord(sid, split, cclass(sid), cat, "", feat["min_dtz_mm"] if feat else None)
        records.append(rec)
        panels.append(_panel(scen, split, bool(feat), "DELIVERY_FAILURE_AFTER_VALID_GRASP", feat)
                      | {"residual_class": cat.value, "capture_subtype": ""})
    return records, panels


def _records_from_payload(payload: "dict[str, Any]") -> list[FailureRecord]:
    """Rebuild FailureRecords from a saved taxonomy payload (to re-select the pilot / re-render without replaying)."""
    return [FailureRecord(p["scenario_id"], p["split"], p["curriculum_stage"], FailureCategory(p["residual_class"]),
                          p.get("capture_subtype", ""), p.get("min_dtz_mm")) for p in payload["records"]]


def _fmt(v: Any) -> str:
    return "—" if v is None else str(v)


def _render_md(payload: "dict[str, Any]") -> str:
    """Markdown report from a taxonomy payload (deterministic, no compute)."""
    g = payload["gate"]
    lines = ["# R11.5+ Phase 1 — Complete Residual Failure Taxonomy (all 24)", "",
             f"**Gate:** `{g['verdict']}` — {g['uncovered']} uncovered, {g['unique']} unique.", "",
             "## By category", "", "| category | n | scenarios |", "|---|---|---|"]
    for cat, ids in payload["by_category"].items():
        lines.append(f"| `{cat}` | {len(ids)} | {', '.join(ids)} |")
    lines += ["", "## Per-scenario panel", "",
              "| scenario | stage | split | rel-target | certified | kinetic | min→final dtz (mm) | "
              "max prog (mm) | lostBR/rel | dwell | zone-entry | residual class |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(payload["records"], key=lambda z: (z["residual_class"], z["scenario_id"])):
        lines.append(f"| {p['scenario_id']} | {p['curriculum_stage']} | {p['split']} | "
                     f"{p['relative_target_vector']} | {_fmt(p['certified_grasp'])} | {_fmt(p.get('kinetic_reached'))} | "
                     f"{_fmt(p.get('min_dtz_mm'))}→{_fmt(p.get('final_dtz_mm'))} | {_fmt(p.get('max_coin_progress_mm'))} | "
                     f"{_fmt(p.get('lost_before_release'))}/{_fmt(p.get('release_step'))} | {_fmt(p.get('max_k6_dwell'))} | "
                     f"{_fmt(p.get('zone_entry_step'))} | `{p['residual_class']}` |")
    lines += ["", "## Bounded 12-scenario recovery pilot", "", "| scenario | split | category | subtype |",
              "|---|---|---|---|"]
    for r in payload["pilot"]:
        lines.append(f"| {r['scenario_id']} | {r['split']} | `{r['category']}` | {r['subtype']} |")
    lines += ["", "## Notes (trajectory ground truth)", "",
              "- Classification is trajectory-derived (deterministic replay), not final-state: the coverage ledger's "
              "`dtz` is `dtz_end`, which cannot separate zone-entry from a short stall.",
              "- **Re-diagnosis:** the 4 negative-x cases (all rel-target `[-0.070, +0.032]`) are "
              "`CONTACT_LOSS_DURING_DELIVERY`, not directional drift — the grasp loses bilateral contact for >80% of the "
              "pre-release window (`lostBR` 39–44/47) and the coin flies off to 126–161 mm. The final-state guess was "
              "`DELIVERY_DIRECTIONAL_BIAS`; the mechanism is the coin squirting out when pushed off-axis, so the ALIGN "
              "fix (which *preserves the grasp* while correcting direction) still applies.",
              "- The 10 `INSUFFICIENT_TRANSPORT_PROGRESS` cases by contrast *hold* the grasp (`lostBR` ~0), move toward "
              "target, and stall 11–50 mm short — a transport horizon/magnitude lever.",
              "- Empty categories (also findings): `HANDOFF_TO_KINETIC_FAILURE`=0 (every certified case moved the coin), "
              "`TARGET_ENTRY_SPEED_FAILURE`/`ZONE_ENTRY_WITHOUT_DWELL`=0 (nothing reached the 20 mm zone; closest 22.4 mm).",
              "- Capture-support audit (candidate-hook, systematic +/+ × 2 seeds): bilateral contact **never forms** "
              "(best class SINGLE, 0 BITRANS/CERT) — an honest-negative geometry bound, not a rank-then-reject bug, so "
              "elite-diversity cannot help the systematic +/+ six.", ""]
    return "\n".join(lines) + "\n"


def _gate(records: list[FailureRecord]) -> "dict[str, Any]":
    summary = {k: len(v) for k, v in summarize(records).items()}
    total = sum(summary.values())
    complete = total == 24 and len({r.scenario_id for r in records}) == 24
    return {"uncovered": total, "unique": len({r.scenario_id for r in records}), "by_category": summary,
            "verdict": "R11_5_PLUS_RESIDUAL_TAXONOMY_COMPLETE" if complete else "R11_5_PLUS_RESIDUAL_TAXONOMY_INCOMPLETE"}


def _pilot_payload(records: list[FailureRecord]) -> "list[dict[str, str]]":
    return [{"scenario_id": r.scenario_id, "split": r.split, "category": r.category.value, "subtype": r.subtype}
            for r in select_pilot(records)]


def _write(payload: "dict[str, Any]", out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_render_md(payload), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", type=Path, default=COVERAGE)
    ap.add_argument("--bank", type=Path, default=R11_4A_BANK)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--render-only", action="store_true", help="re-select pilot + re-render md from an existing --out json (no replay)")
    args = ap.parse_args()
    if args.render_only:                                        # re-render + re-select pilot from the saved json (fast)
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        payload["pilot"] = _pilot_payload(_records_from_payload(payload))
        _write(payload, args.out)
        print(json.dumps(payload["gate"], indent=2), flush=True)
        print(f"pilot: {[r['scenario_id'] for r in payload['pilot']]}", flush=True)
        print("R11_5_PLUS_TAXONOMY_RENDERED", flush=True)
        return
    rig = _rig()
    cfg_tc = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    obj = GraspObjective()
    conf = PipelineConfig(teacher_budget=3, grasp_objective=obj)
    t0 = time.perf_counter()
    records, panels = build_records(args.coverage, args.bank, rig, cfg_tc, obj, conf)
    payload = {"gate": _gate(records), "by_category": summarize(records),
               "records": panels, "pilot": _pilot_payload(records)}
    _write(payload, args.out)
    print(json.dumps(payload["gate"], indent=2), flush=True)
    print(f"pilot: {[r['scenario_id'] for r in payload['pilot']]}", flush=True)
    print(f"wrote {args.out} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print("R11_5_PLUS_TAXONOMY_DONE", flush=True)


if __name__ == "__main__":
    main()
