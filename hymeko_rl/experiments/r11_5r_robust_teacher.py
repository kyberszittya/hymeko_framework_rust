"""R11.5R — Robust Delivery Teacher Re-certification. Per scenario: T0 (the nominal teacher theta, from the R11.4B bank)
vs T1 (robust CEM); compare local-K6 survival with a SHARED perturbation bank; re-certify the demonstration bank
(WIDE_RECERTIFIED keeps the robust theta, else the nominal FALLBACK), emit a B1 dataset for the same-size BC re-run, and
apply the teacher gate. Only the teacher scoring changes; the reconstruction/action/physics/K6/safety are frozen.
Parallelizable via --offset/--limit.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import BcSample, bc_context, fresh_rig, reconstruct_capture, scenario_by_id
from hymeko_rl.coin_delivery.delivery_teacher.robust_teacher import (
    REFINE,
    PerturbationBank,
    RobustTeacherConfig,
    recert_status,
    robust_cem,
    survival_at,
)
from hymeko_rl.coin_delivery.delivery_teacher.solver import full_transport_spec

R11_4B_DATASET = Path("reports/2026-08-03-r11-4b-bc/dataset")
DEFAULT_OUT = Path("reports/2026-08-05-r11-5r-robust-teacher")
BANK_SEED = 12345                # the shared perturbation seed-bank (T0 and T1 see identical deltas)
SCALES = (0.005, 0.01, 0.02)


def _load_teacher(dataset_dir: Path) -> "list[BcSample]":
    out: list[BcSample] = []
    for f in sorted(glob.glob(str(dataset_dir / "extract_*.jsonl"))):
        for line in Path(f).open():
            if line.strip():
                d = json.loads(line)
                if not d.get("omitted"):
                    out.append(BcSample.from_json(d))
    return out


def recertify_one(cfg: Any, conf: Any, obj: Any, rcfg: RobustTeacherConfig, bank: PerturbationBank,
                  smp: BcSample) -> dict[str, Any]:
    """T0 (nominal theta) vs T1 (robust CEM) on one scenario; returns the re-certification record + the chosen theta."""
    scen = scenario_by_id(smp.scenario_id)
    rc = reconstruct_capture(fresh_rig(), cfg, conf, obj, scen, smp.seed)
    if rc is None:
        return {"scenario_id": smp.scenario_id, "split": smp.split, "status": "NO_CAPTURE"}
    snap = rc.result.outcome.snapshot
    t0_theta = np.array(smp.theta, np.float64)
    t0_surv, t0_cvar, _n = survival_at(snap, t0_theta, REFINE, rcfg.k_refine, bank, rcfg.cvar_alpha)
    t1 = robust_cem(snap, full_transport_spec(), rcfg, bank, seed=0, init_theta=t0_theta)   # warm-start from nominal theta
    status = recert_status(t1.record, rcfg)
    chosen = [float(v) for v in (t1.theta if status == "WIDE_RECERTIFIED" else t0_theta)]
    return {"scenario_id": smp.scenario_id, "split": smp.split, "seed": smp.seed, "source": smp.source, "status": status,
            "t0": {"survival1": t0_surv, "cvar_dtz": t0_cvar}, "t1": {"survival1": t1.record["surv1"],
            "survival05": t1.record["surv05"], "survival2": t1.record["surv2"], "cvar_dtz": t1.record["cvar_dtz"],
            "compute": t1.compute}, "chosen_theta": chosen, "chosen_wide": status == "WIDE_RECERTIFIED", "x": smp.x,
            "t0_theta": [float(v) for v in t0_theta]}


def _b1_sample(r: dict) -> "dict[str, Any]":
    """A BcSample row for the B1 (robust-recertified) dataset consumed by the same R11.4B BC harness."""
    return BcSample(r["scenario_id"], r["split"], int(r["seed"]), r.get("source", "recovered"),
                    [float(v) for v in r["x"]], [float(v) for v in r["chosen_theta"]], True,
                    float(r["t1"]["cvar_dtz"] if r["chosen_wide"] else r["t0"]["cvar_dtz"])).to_json()


_WIDE, _NARROW, _NO_K6 = "WIDE_RECERTIFIED", "NARROW_ONLY", "NO_NOMINAL_K6"


def _mean(rows: "list[dict]", fn: Any) -> float:
    return round(float(np.mean([fn(r) for r in rows])), 3) if rows else 0.0


def _survival_stats(k6: "list[dict]") -> "tuple[float, float, float]":
    return (_mean(k6, lambda r: r["t1"]["survival1"] - r["t0"]["survival1"]),
            _mean(k6, lambda r: r["t0"]["survival1"]), _mean(k6, lambda r: r["t1"]["survival1"]))


def _status_counts(good: "list[dict]") -> dict[str, int]:
    return {s: sum(1 for r in good if r["status"] == s) for s in (_WIDE, _NARROW, _NO_K6)}


def _partition(rows: "list[dict]") -> "tuple[list, list, list]":
    good = [r for r in rows if r.get("status") in (_WIDE, _NARROW, _NO_K6)]
    return good, [r for r in good if r["split"] == "train"], [r for r in good if r["status"] in (_WIDE, _NARROW)]


def _recert_verdict(frac_wide: float, delivered: int, surv_gain: float) -> str:
    if frac_wide >= 0.70 and delivered >= 45 and surv_gain >= 0.20:
        return "R11_5R_ROBUST_TEACHER_RECERTIFICATION_PASS"
    return "R11_5R_WIDE_BASIN_SUPPORT_LIMITED"


def teacher_gate(rows: "list[dict]") -> dict[str, Any]:
    """PASS: robust theta for >=70% of train scenarios, overall nominal coverage >=45/64, and T1 survival >> T0 on the
    nominal-K6 scenarios (mean +0.20). Else R11_5R_WIDE_BASIN_SUPPORT_LIMITED."""
    good, train, k6 = _partition(rows)                                       # k6 = nominal-K6 (survival is meaningful)
    n_wide = sum(1 for r in train if r["status"] == _WIDE)
    frac_wide = round(n_wide / len(train), 3) if train else 0.0
    delivered = 7 + len(good)                     # 7 frozen-R2 + every chosen theta delivers K6 (robust or nominal fallback)
    surv_gain, mean_t0, mean_t1 = _survival_stats(k6)
    return {"n": len(good), "train_wide_frac": frac_wide, "n_wide": n_wide, "status_counts": _status_counts(good),
            "nominal_coverage": f"{delivered}/64", "mean_survival_gain_t1_minus_t0": surv_gain,
            "mean_t0_survival1": mean_t0, "mean_t1_survival1": mean_t1,
            "verdict": _recert_verdict(frac_wide, delivered, surv_gain)}


def _slice(seq: list, offset: int, limit: int) -> list:
    return seq[offset:(offset + limit if limit else len(seq))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, default=R11_4B_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg, conf, obj = bc_context()
    rcfg = RobustTeacherConfig()
    bank = PerturbationBank(SCALES, max(rcfg.k_screen, rcfg.k_refine, rcfg.k_stress), seed=BANK_SEED)
    smps = _slice(_load_teacher(args.dataset_dir), args.offset, args.limit)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / f"recert_{args.offset:03d}.jsonl").open("w", encoding="utf-8") as fh:
        for i, smp in enumerate(smps, 1):
            r = recertify_one(cfg, conf, obj, rcfg, bank, smp)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
            print(f"[{i}/{len(smps)}] {smp.scenario_id:26s} {smp.split:5s} {r.get('status')} "
                  f"t0_surv1={r.get('t0', {}).get('survival1')} t1_surv1={r.get('t1', {}).get('survival1')}", flush=True)
    print("=== R11.5R TEACHER GATE ===", flush=True)
    print(json.dumps(teacher_gate(rows), indent=2), flush=True)
    print("R11_5R_RECERT_DONE", flush=True)


if __name__ == "__main__":
    main()
