"""R11.7A U6B — bounded teacher-free retrieval pilot, BOX-FIRST (O4-S).

The flagship non-circular family (O4-S, best capture in U6A at 5/6) end-to-end: per-object teacher-theta bank
(N capture-population seeds × R delivery restarts, top-1 certified K6 theta per train scenario) → an
object-family-keyed standardized nearest retrieval (top-1 full theta, no blend/CEM/oracle/teacher) → exact-zero
evaluation on the DEV scenarios with the 8-class failure taxonomy. The sealed pilot-TEST is held until the dev
freeze. O0's frozen retrieval is the control (not re-run here).

Scenarios are chosen from the CERTIFYING band (coin displacement 0.076–0.101 from the zone; center excluded,
where U6A + the cost-probe showed no object certifies) with diverse (coin, target) configs so retrieval has a
real interpolation question. Two phases so the long bank-gen runs detached:

  python -m hymeko_rl.experiments.r11_7a_u6b_box_pilot bank   # ~1.5–2 h, writes bank.json
  python -m hymeko_rl.experiments.r11_7a_u6b_box_pilot eval   # fast: retrieval + dev eval + gate
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import (
    bc_context, best_theta_full, descriptor, full_transport_spec, reconstruct_capture, scenario_by_id)
from hymeko_rl.coin_delivery.delivery_bc.retrieval import RetrievalConfig, RetrievalDeliveryPolicy, SelectRule
from hymeko_rl.coin_delivery.exact_zero_composition import compose_one
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

_VARIANT = "O4-S"
_OUT = Path("reports/2026-08-06-r11-7a-u6b-box-pilot")
_BANK = _OUT / "bank.json"

# The box's OWN 8/2/2 split (independent object). Certifying band, diverse (coin,target); center excluded.
BOX_TRAIN = (
    "bank_c0_1", "bank_c1_+0.01_+0.00", "bank_c1_+0.01_+0.03", "bank_c2_+0.015_+0.000",
    "bank_c2_+0.015_-0.015", "bank_c2_+0.025_+0.000", "bank_c3_r6_a-30", "bank_c3_r7_a-15")
BOX_DEV = ("bank_c2_+0.015_+0.015", "bank_c3_r6_a+15")
BOX_TEST = ("bank_c2_+0.025_+0.015", "bank_c1_+0.01_+0.02")     # SEALED until dev-freeze

_N_CAPTURE = 20        # capture population (seeds) per train scenario
_R_DELIVERY = 5        # teacher delivery-search restarts
_SUPPORT_RADIUS = 6.14  # descriptor-space support radius (R11.6C), for RETRIEVAL_OUT_OF_SUPPORT


# ---- bank generation -----------------------------------------------------------------------------
def _gen_train_bank(rig: dict, cfg: Any, conf: Any, obj: Any) -> dict[str, Any]:
    """Per train scenario, sweep N capture seeds; for each certified grasp run R teacher restarts; keep the
    best K6 (x, theta). One bank entry per scenario that yields any K6."""
    samples: list[dict[str, Any]] = []
    per_scen: dict[str, Any] = {}
    for sid in BOX_TRAIN:
        scen = scenario_by_id(sid)
        best: "tuple[float, list[float], list[float], int] | None" = None
        counts = {"certified_capture": 0, "teacher_k6": 0}
        t0 = time.perf_counter()
        for seed in range(_N_CAPTURE):
            rc = reconstruct_capture(rig, cfg, conf, obj, scen, seed)
            if rc is None:
                continue
            counts["certified_capture"] += 1
            snap = rc.result.outcome.snapshot
            k6, dtz, theta = best_theta_full(snap, full_transport_spec(), _R_DELIVERY)
            if not k6:
                continue
            counts["teacher_k6"] += 1
            if best is None or dtz < best[0]:
                x = descriptor(scen, rc, snap)
                best = (float(dtz), [float(v) for v in x], [float(t) for t in theta], seed)
        dt = time.perf_counter() - t0
        per_scen[sid] = {**counts, "has_sample": best is not None,
                         "best_dtz_mm": (round(best[0], 2) if best else None), "wall_s": round(dt, 1)}
        if best is not None:
            samples.append({"scenario_id": sid, "seed": best[3], "x": best[1], "theta": best[2],
                            "k6": True, "dtz_mm": round(best[0], 2)})
        print(f"[{sid:24s}] certified={counts['certified_capture']}/{_N_CAPTURE} k6={counts['teacher_k6']} "
              f"sample={'Y' if best else 'N'} best_dtz={per_scen[sid]['best_dtz_mm']} ({dt:.0f}s)", flush=True)
    return {"samples": samples, "per_scenario": per_scen}


def run_bank() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    cfg, conf, obj = bc_context()
    t0 = time.perf_counter()
    rig = _rig(object_spec=variant(_VARIANT).object_spec)
    print(f"box rig built in {time.perf_counter() - t0:.0f}s; generating bank "
          f"({len(BOX_TRAIN)} train × N={_N_CAPTURE} × R={_R_DELIVERY})…", flush=True)
    bank = _gen_train_bank(rig, cfg, conf, obj)
    bank["meta"] = {"variant": _VARIANT, "n_capture": _N_CAPTURE, "r_delivery": _R_DELIVERY,
                    "train": list(BOX_TRAIN), "dev": list(BOX_DEV), "test": list(BOX_TEST),
                    "total_wall_s": round(time.perf_counter() - t0, 1)}
    _BANK.write_text(json.dumps(bank, indent=1))
    n = len(bank["samples"])
    print(f"\nBANK_DONE: {n}/{len(BOX_TRAIN)} train scenarios yielded a K6 teacher sample "
          f"({bank['meta']['total_wall_s'] / 60:.1f} min). wrote {_BANK}", flush=True)
    return 0 if n > 0 else 1


# ---- retrieval + evaluation ----------------------------------------------------------------------
_TAXON = {
    "SUCCESS": "SUCCESS",
    "INVALID_INITIAL_CONDITION": "MODEL_OR_CONTRACT_FAILURE",
    "REACH_FAILURE": "REACH_GEOMETRY_FAILURE",
    "PRECONTACT_HANDOFF_INVALID": "CAPTURE_PROPOSAL_TRANSFER_FAILURE",
    "CAPTURE_NO_CERTIFIED_GRASP": "CERTIFICATE_GEOMETRY_FAILURE",
    "DESCRIPTOR_DRIFT": "CAPTURE_PROPOSAL_TRANSFER_FAILURE",
    "RETRIEVAL_OUT_OF_SUPPORT": "RETRIEVAL_OUT_OF_SUPPORT",
    "NUDGE": "CONTACT_RETENTION_FAILURE",
    "SAFETY": "CONTACT_RETENTION_FAILURE",
    "FAILURE": "DELIVERY_PROGRAM_TRANSFER_FAILURE",   # delivered, no K6 (θ mis-transports)
}


def _build_policy(bank: dict) -> "RetrievalDeliveryPolicy | None":
    samples = bank["samples"]
    if not samples:
        return None
    X = np.asarray([s["x"] for s in samples], np.float64)
    Theta = np.asarray([s["theta"] for s in samples], np.float64)
    survival = np.ones(len(samples), np.float64)                 # top-1 nearest; survival uninformative here
    cfg = RetrievalConfig(standardize=True, k=1, select=SelectRule.NEAREST)
    return RetrievalDeliveryPolicy.fit(X, Theta, survival, cfg)


def run_eval() -> int:
    if not _BANK.exists():
        print(f"no bank at {_BANK}; run the `bank` phase first", flush=True)
        return 2
    bank = json.loads(_BANK.read_text())
    policy = _build_policy(bank)
    if policy is None:
        print("EVAL ABORT: empty bank (no train scenario produced a K6 teacher) — see per_scenario", flush=True)
        return 1
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(_VARIANT).object_spec)
    records = []
    for sid in BOX_DEV:                                          # sealed TEST held until dev-freeze
        for seed in (0, 1):
            rec = compose_one(rig, scenario_by_id(sid), seed, cfg, conf, obj, policy, _SUPPORT_RADIUS)
            taxon = _TAXON.get(rec.outcome_class, rec.outcome_class)
            records.append({"scenario_id": sid, "seed": seed, "outcome": rec.outcome_class, "taxon": taxon,
                            "k6": bool(rec.k6), "safe": bool(getattr(rec, "safe", True)),
                            "dtz_mm": getattr(rec, "dtz_mm", None), "support": getattr(rec, "support_dist", None)})
            print(f"[{sid:24s} s{seed}] {rec.outcome_class:26s} taxon={taxon} k6={rec.k6}", flush=True)
    res = _summarize_eval(bank, records)
    (_OUT / "eval.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"\n{res['verdict']} | dev K6={res['dev_k6']}/{len(records)} | taxonomy={res['taxonomy']}")
    print(f"wrote {_OUT / 'eval.json'}")
    return 0 if res["gate_pass"] else 1


def _summarize_eval(bank: dict, records: list[dict]) -> dict[str, Any]:
    n_bank = len(bank["samples"])
    certified_capture = sum(v["certified_capture"] for v in bank["per_scenario"].values())
    teacher_k6 = sum(v["teacher_k6"] for v in bank["per_scenario"].values())
    model_contract = [r for r in records if r["taxon"] == "MODEL_OR_CONTRACT_FAILURE"]
    dev_k6 = sum(1 for r in records if r["k6"])
    tax: dict[str, int] = {}
    for r in records:
        tax[r["taxon"]] = tax.get(r["taxon"], 0) + 1
    # Box-family single-object gate slice: bank non-empty, capture certified >0, teacher motion >0, no
    # model/contract fault in eval, ≥1 dev exact-zero strict-K6 (the flagship non-circular delivery).
    gate_pass = bool(n_bank > 0 and certified_capture > 0 and teacher_k6 > 0
                     and not model_contract and dev_k6 >= 1)
    return {
        "verdict": "R11_7A_BOX_RETRIEVAL_PILOT_PASS" if gate_pass else "R11_7A_BOX_RETRIEVAL_PILOT_FAIL",
        "gate_pass": gate_pass, "variant": _VARIANT,
        "bank_samples": n_bank, "bank_certified_captures": certified_capture, "bank_teacher_k6": teacher_k6,
        "dev_k6": dev_k6, "n_model_contract_failures": len(model_contract), "taxonomy": tax,
        "records": records, "per_scenario_bank": bank["per_scenario"],
    }


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else "bank"
    if phase == "bank":
        return run_bank()
    if phase == "eval":
        return run_eval()
    print(f"unknown phase {phase!r}; use 'bank' or 'eval'", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
