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
from hymeko_rl.coin_delivery.exact_zero_composition import compose_one, deliver_record, reach_capture_descriptor
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

_BANK_DENSE = _OUT / "bank_dense.json"   # R11.7B: keep ALL certified-K6 θ per scenario (coverage densification)
_N_CAPTURE = 20        # capture population (seeds) per train scenario
_R_DELIVERY = 5        # teacher delivery-search restarts
_SUPPORT_RADIUS = 6.14  # descriptor-space support radius (R11.6C), for RETRIEVAL_OUT_OF_SUPPORT


# ---- bank generation -----------------------------------------------------------------------------
def _gen_train_bank(rig: dict, cfg: Any, conf: Any, obj: Any, keep_all: bool = False) -> dict[str, Any]:
    """Per train scenario, sweep N capture seeds; for each certified grasp run R teacher restarts. ``keep_all``
    stores EVERY certified-K6 (x, theta) (the coverage-densification bank, R11.7B); otherwise the best-of-1 per
    scenario (the original sparse bank)."""
    samples: list[dict[str, Any]] = []
    per_scen: dict[str, Any] = {}
    for sid in BOX_TRAIN:
        scen = scenario_by_id(sid)
        scen_samples: list[dict[str, Any]] = []
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
            x = descriptor(scen, rc, snap)
            scen_samples.append({"scenario_id": sid, "seed": seed, "x": [float(v) for v in x],
                                 "theta": [float(t) for t in theta], "k6": True, "dtz_mm": round(float(dtz), 2)})
        dt = time.perf_counter() - t0
        kept = scen_samples if keep_all else (
            [min(scen_samples, key=lambda s: s["dtz_mm"])] if scen_samples else [])
        samples.extend(kept)
        best_dtz = min((s["dtz_mm"] for s in scen_samples), default=None)
        per_scen[sid] = {**counts, "n_kept": len(kept), "best_dtz_mm": best_dtz, "wall_s": round(dt, 1)}
        print(f"[{sid:24s}] certified={counts['certified_capture']}/{_N_CAPTURE} k6={counts['teacher_k6']} "
              f"kept={len(kept)} best_dtz={best_dtz} ({dt:.0f}s)", flush=True)
    return {"samples": samples, "per_scenario": per_scen}


def run_bank(keep_all: bool = False) -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    out_path = _BANK_DENSE if keep_all else _BANK
    cfg, conf, obj = bc_context()
    t0 = time.perf_counter()
    rig = _rig(object_spec=variant(_VARIANT).object_spec)
    print(f"box rig built in {time.perf_counter() - t0:.0f}s; generating {'DENSE ' if keep_all else ''}bank "
          f"({len(BOX_TRAIN)} train × N={_N_CAPTURE} × R={_R_DELIVERY}, keep_all={keep_all})…", flush=True)
    bank = _gen_train_bank(rig, cfg, conf, obj, keep_all=keep_all)
    bank["meta"] = {"variant": _VARIANT, "n_capture": _N_CAPTURE, "r_delivery": _R_DELIVERY, "keep_all": keep_all,
                    "train": list(BOX_TRAIN), "dev": list(BOX_DEV), "test": list(BOX_TEST),
                    "total_wall_s": round(time.perf_counter() - t0, 1)}
    out_path.write_text(json.dumps(bank, indent=1))
    n, scen_hit = len(bank["samples"]), sum(1 for v in bank["per_scenario"].values() if v["n_kept"] > 0)
    print(f"\nBANK_DONE: {n} samples from {scen_hit}/{len(BOX_TRAIN)} train scenarios "
          f"({bank['meta']['total_wall_s'] / 60:.1f} min). wrote {out_path}", flush=True)
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


# ---- fairer eval: more seeds + retrieval-config diagnostic (bank frozen) --------------------------
def _policy_of(bank: dict, k: int, rule: SelectRule) -> "RetrievalDeliveryPolicy | None":
    samples = bank["samples"]
    if len(samples) < k:
        return None
    X = np.asarray([s["x"] for s in samples], np.float64)
    Theta = np.asarray([s["theta"] for s in samples], np.float64)
    survival = np.ones(len(samples), np.float64)
    return RetrievalDeliveryPolicy.fit(X, Theta, survival, RetrievalConfig(standardize=True, k=k, select=rule))


def run_fair_eval(n_seeds: int = 5) -> int:
    """Fairer retrieval test: N_seeds/dev-scenario, and BOTH the top-1 deploy baseline and a k=3 distance-weighted
    diagnostic applied to the SAME reached snapshot (reach+capture run once per (scenario,seed) — retrieval-config-
    independent). Characterizes whether the box can clear a dev K6 at all, vs the capture-seed inconsistency."""
    if not _BANK.exists():
        print(f"no bank at {_BANK}; run `bank` first", flush=True)
        return 2
    bank = json.loads(_BANK.read_text())
    policies = {"top1_nearest": _policy_of(bank, 1, SelectRule.NEAREST),
                "k3_weighted": _policy_of(bank, 3, SelectRule.DIST_WEIGHTED)}
    policies = {k: v for k, v in policies.items() if v is not None}
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(_VARIANT).object_spec)
    records: list[dict[str, Any]] = []
    for sid in BOX_DEV:
        for seed in range(n_seeds):
            h = reach_capture_descriptor(rig, scenario_by_id(sid), seed, cfg, conf, obj)
            if h.record is not None:                             # reach/capture failed — policy-independent
                taxon = _TAXON.get(h.record.outcome_class, h.record.outcome_class)
                records.append({"scenario_id": sid, "seed": seed, "policy": "-", "reached_delivery": False,
                                "outcome": h.record.outcome_class, "taxon": taxon, "k6": False})
                print(f"[{sid:22s} s{seed}] PRE-DELIVERY {h.record.outcome_class:26s} ({taxon})", flush=True)
                continue
            for pname, pol in policies.items():
                rec = deliver_record(scenario_by_id(sid), seed, h.snap, h.x, pol, _SUPPORT_RADIUS)
                records.append({"scenario_id": sid, "seed": seed, "policy": pname, "reached_delivery": True,
                                "outcome": rec.outcome_class, "taxon": _TAXON.get(rec.outcome_class, rec.outcome_class),
                                "k6": bool(rec.k6), "dtz_mm": rec.dtz_mm, "support": rec.support_dist})
                print(f"[{sid:22s} s{seed}] {pname:13s} {rec.outcome_class:26s} k6={rec.k6} dtz={rec.dtz_mm}", flush=True)

    reached = [r for r in records if r["reached_delivery"]]
    by_policy = {p: {"k6": sum(1 for r in reached if r["policy"] == p and r["k6"]),
                     "n": sum(1 for r in reached if r["policy"] == p)} for p in policies}
    n_pairs = len({(r["scenario_id"], r["seed"]) for r in records})
    n_reached = len({(r["scenario_id"], r["seed"]) for r in reached})
    res = {"variant": _VARIANT, "n_seeds": n_seeds, "n_rollouts": n_pairs, "n_reached_delivery": n_reached,
           "by_policy": by_policy, "any_dev_k6": any(v["k6"] > 0 for v in by_policy.values()), "records": records}
    (_OUT / "fair_eval.json").write_text(json.dumps(res, indent=1, default=str))
    summary = {p: f"{v['k6']}/{v['n']}" for p, v in by_policy.items()}
    print(f"\nFAIR EVAL: {n_reached}/{n_pairs} rollouts reached delivery. "
          f"per-policy dev K6: {summary} | any_dev_K6={res['any_dev_k6']}")
    print(f"wrote {_OUT / 'fair_eval.json'}")
    return 0 if res["any_dev_k6"] else 1


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else "bank"
    if phase == "bank":
        return run_bank()
    if phase == "bank_dense":
        return run_bank(keep_all=True)
    if phase == "eval":
        return run_eval()
    if phase == "fair_eval":
        return run_fair_eval(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    print(f"unknown phase {phase!r}; use 'bank' | 'bank_dense' | 'eval' | 'fair_eval'", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
