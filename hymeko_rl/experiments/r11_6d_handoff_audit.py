"""R11.6D Phase 1+2 — causal handoff audit + counterfactual ablation on the 3 c3 far-angle dev failures.

Per dev failure: pair it with its retrieved (nearest) bank demo, reconstruct both handoffs, and roll the SAME retrieved
theta from (a) the bank handoff [control, must strict-K6], (b) the dev handoff [must reproduce the ~30mm miss], and
(c) the dev handoff with ONE component swapped to the bank value, for every counterfactual axis. The component whose
restoration recovers strict-K6 is the control-critical variable. Diagnostic only — no optimization, no canonicalizer yet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import bc_context, fresh_rig, reconstruct_capture, scenario_by_id
from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer
from hymeko_rl.coin_delivery.handoff_audit import SwapComponent, read_audit, roll, swap_component
from hymeko_rl.experiments.r11_4b_conditioned_bc import _load_dataset

FROZEN = Path("reports/2026-08-06-r11-5r-retrieval/frozen_policy.json")
B1_DATASET = Path("reports/2026-08-05-r11-5r-robust-teacher/dataset_b1")
DEFAULT_OUT = Path("reports/2026-08-06-r11-6d-audit")
DEV_FAILURES = ["bank_c3_r7_a+45", "bank_c3_r9_a-30", "bank_c3_r9_a-15"]


def _nearest_bank(dev_x: np.ndarray, Xs: np.ndarray, ids: "list[str]", std: Standardizer) -> "tuple[str, float]":
    d = np.linalg.norm(Xs - std.transform(dev_x), axis=1)
    i = int(np.argmin(d))
    return ids[i], round(float(d[i]), 3)


def _component_diff(a_dev: dict, a_bank: dict, c: SwapComponent) -> float:
    """Pre-swap magnitude of the dev-vs-bank difference for a component (to flag a no-op swap)."""
    def vd(key: str) -> float:
        return float(np.linalg.norm(np.asarray(a_dev[key]) - np.asarray(a_bank[key])))
    if c is SwapComponent.COIN_POS:
        return round(vd("coin_xy") * 1000, 2)                                # mm
    if c is SwapComponent.COIN_YAW:
        return round(abs(a_dev["coin_yaw"] - a_bank["coin_yaw"]), 4)
    if c is SwapComponent.COIN_LINVEL:
        return round(abs(a_dev["coin_speed"] - a_bank["coin_speed"]), 4)
    if c is SwapComponent.COIN_SPIN:
        return round(abs(a_dev["coin_spin"] - a_bank["coin_spin"]), 4)
    if c is SwapComponent.ARM_QPOS:
        return round(vd("arm_qpos"), 4)
    if c is SwapComponent.ARM_QVEL:
        return round(vd("arm_qvel"), 4)
    if c is SwapComponent.PREV_TAU:
        return round(vd("prev_tau"), 4)
    return round(vd("zone") * 1000, 2)                                       # ZONE, mm


def audit_pair(cfg: Any, conf: Any, obj: Any, Xs: np.ndarray, ids: "list[str]", std: Standardizer,
               table_theta: "list[list[float]]", smps: dict, dev_sid: str) -> "dict[str, Any]":
    dev = smps[dev_sid]
    bank_sid, dist = _nearest_bank(np.asarray(dev.x, np.float64), Xs, ids, std)
    theta = np.asarray(table_theta[ids.index(bank_sid)], np.float64)          # the retrieved (nearest) theta
    bank = smps[bank_sid]
    rc_dev = reconstruct_capture(fresh_rig(), cfg, conf, obj, scenario_by_id(dev_sid), dev.seed)
    rc_bank = reconstruct_capture(fresh_rig(), cfg, conf, obj, scenario_by_id(bank_sid), bank.seed)
    if rc_dev is None or rc_bank is None:
        return {"dev": dev_sid, "bank": bank_sid, "error": "reconstruction_failed"}
    snap_dev, snap_bank = rc_dev.result.outcome.snapshot, rc_bank.result.outcome.snapshot
    a_dev, a_bank = read_audit(snap_dev), read_audit(snap_bank)
    control, baseline = roll(snap_bank, theta), roll(snap_dev, theta)
    cf: dict[str, Any] = {}
    for c in SwapComponent:
        r = roll(swap_component(snap_dev, snap_bank, c), theta)
        cf[c.value] = {**r, "pre_swap_diff": _component_diff(a_dev, a_bank, c),
                       "gap_gain": round(r["gap_closed"] - baseline["gap_closed"], 3)}
    recovering = [c for c, v in cf.items() if v["k6"] and not baseline["k6"]]
    print(f"[{dev_sid}] bank={bank_sid} dist={dist} | control_k6={control['k6']} baseline_k6={baseline['k6']} "
          f"({baseline['dtz_mm']}mm) -> recovering: {recovering or 'NONE'}", flush=True)
    return {"dev": dev_sid, "bank": bank_sid, "descriptor_dist": dist, "theta_source": bank_sid,
            "control": control, "baseline": baseline, "counterfactual": cf, "recovering": recovering,
            "audit_dev": a_dev, "audit_bank": a_bank}


def _axis_recovery(diag: "list[dict]", axis: str) -> "tuple[int, float]":
    n = sum(1 for r in diag if axis in r["recovering"])
    gain = float(np.mean([r["counterfactual"][axis]["gap_gain"] for r in diag])) if diag else 0.0
    return n, round(gain, 3)


def summarize(rows: "list[dict]") -> "dict[str, Any]":
    """Which component recovers strict-K6 across the DIAGNOSABLE pairs (baseline failed with the nearest theta). Pairs
    whose baseline already delivers are a retrieval-blend artifact (the single nearest theta transports; only the
    weighted blend failed) and are reported separately, not diagnosed."""
    good = [r for r in rows if "error" not in r]
    diag = [r for r in good if not r["baseline"]["k6"]]                       # baseline fails -> a real transport gap
    artifact = [r["dev"] for r in good if r["baseline"]["k6"]]                # nearest delivers -> blend-only failure
    stats = {c.value: _axis_recovery(diag, c.value) for c in SwapComponent}
    ranked = sorted(stats, key=lambda c: stats[c], reverse=True)
    return {"n_pairs": len(good), "n_diagnosable": len(diag), "blend_artifact_dev": artifact,
            "controls_all_k6": all(r["control"]["k6"] for r in good),
            "recover_count": _project(stats, 0), "mean_gap_gain": _project(stats, 1),
            "ranked_by_recovery": ranked, "dominant": ranked[0] if diag else None}


def _project(stats: "dict[str, tuple]", i: int) -> "dict[str, Any]":
    return {c: stats[c][i] for c in stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", type=Path, default=FROZEN)
    ap.add_argument("--dataset-dir", type=Path, default=B1_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    fp = json.loads(args.frozen.read_text())
    ids, table_theta = fp["table"]["scenario_ids"], fp["table"]["theta"]
    Xtr = np.asarray(fp["table"]["X"], np.float64)
    std = Standardizer.fit(Xtr)
    Xs = std.transform(Xtr)
    smps = {s.scenario_id: s for s in _load_dataset(args.dataset_dir)}
    cfg, conf, obj = bc_context()
    targets = DEV_FAILURES[args.offset:(args.offset + args.limit if args.limit else len(DEV_FAILURES))]
    rows = [audit_pair(cfg, conf, obj, Xs, ids, std, table_theta, smps, sid) for sid in targets]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f"audit_{args.offset:03d}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    if not args.limit:                                                        # full run: summarize + verdict
        summary = summarize(rows)
        (args.out / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print("=== R11.6D CAUSAL AUDIT SUMMARY ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        print("R11_6D_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
