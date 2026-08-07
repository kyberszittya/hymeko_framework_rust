"""R11.7B — multi-object characterization pilot (unified benchmark across object families).

Runs ONE object family through the same exact-zero ladder on 3 functionally-matched scenarios × 5 seeds and
emits a phase-diagram row + the unified failure taxonomy, so O0 (coin) / O4 (box) / O1-L (size) / O2-M (dynamics)
are directly comparable. The scientific question is NOT "solve O1/O2" but "does the same failure structure appear
when only size, or only dynamics, changes?"

Ladder per (scenario, seed):  exact-zero reach → capture → structured-teacher feasibility (best_theta_full) →
frozen-bank teacher-free retrieval (LOO) → strict K6. The teacher runs on every capture, so the bank + the
θ×handoff coverage/rank come for free — the SPARSE audit (best-θ/scenario) and the DENSE audit (all K6 θ) are both
computed in one pass (no densification run needed; we only *report* both, we do not tune).

Metrics (phase diagram): reach rate · certified-capture rate · teacher-K6|capture (physical solvability) ·
retrieval-K6|capture (deployable generalization, LOO) · bank-contains-delivering-θ (coverage) · delivering-θ
median rank (selection difficulty).

Run:  python -m hymeko_rl.experiments.r11_7b_multiobject_characterization <variant_id> [n_seeds]
      (variant_id ∈ {O0, O1-L, O2-M, O4-S})
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.delivery_bc.dataset import (
    bc_context, best_theta_full, full_transport_spec, scenario_by_id)
from hymeko_rl.coin_delivery.delivery_bc.models import Standardizer, clip_theta
from hymeko_rl.coin_delivery.exact_zero_composition import _delivery_signals, reach_capture_descriptor
from hymeko_rl.coin_delivery.object_curriculum import variant
from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

_OUT = Path("reports/2026-08-07-r11-7b-multiobject")
_R_DELIVERY = 5
# Functionally-matched scenarios, fixed across families for comparability (certifying band, center excluded):
_SCENARIOS = (("S0_short", "bank_c2_+0.015_+0.000"), ("S1_offcenter", "bank_c2_+0.025_+0.000"),
              ("S2_far_angular", "bank_c3_r6_a+15"))

_TAXON_PRE = {"REACH_FAILURE": "REACH_FAILURE", "INVALID_INITIAL_CONDITION": "REACH_FAILURE",
              "PRECONTACT_HANDOFF_INVALID": "CAPTURE_SUPPORT_FAILURE",
              "CAPTURE_NO_CERTIFIED_GRASP": "CAPTURE_SUPPORT_FAILURE", "DESCRIPTOR_DRIFT": "CAPTURE_SUPPORT_FAILURE"}


def _capture_and_teacher(rig: dict, cfg: Any, conf: Any, obj: Any) -> list[dict[str, Any]]:
    """Reach → capture → teacher for each (scenario, seed). Records reach/capture status, teacher-K6, and (on
    teacher K6) the descriptor + θ + snapshot for the retrieval/coverage stage."""
    recs: list[dict[str, Any]] = []
    for label, sid in _SCENARIOS:
        scen = scenario_by_id(sid)
        for seed in range(_N_SEEDS):
            h = reach_capture_descriptor(rig, scen, seed, cfg, conf, obj)
            if h.record is not None:
                taxon = _TAXON_PRE.get(h.record.outcome_class, h.record.outcome_class)
                reached = taxon != "REACH_FAILURE"
                recs.append({"scenario": label, "sid": sid, "seed": seed, "reached": reached, "captured": False,
                             "taxon": taxon})
                print(f"[{label:14s} s{seed}] {h.record.outcome_class:26s} → {taxon}", flush=True)
                continue
            k6, dtz, theta = best_theta_full(h.snap, full_transport_spec(), _R_DELIVERY)
            x = np.asarray(h.x, np.float64)                  # h.x is the 30-D descriptor from reach_capture_descriptor
            recs.append({"scenario": label, "sid": sid, "seed": seed, "reached": True, "captured": True,
                         "teacher_k6": bool(k6), "teacher_dtz": dtz, "x": [float(v) for v in x],
                         "theta": [float(t) for t in theta], "_snap": h.snap})
            print(f"[{label:14s} s{seed}] captured; teacher_k6={k6} dtz={dtz}", flush=True)
    return recs


def _rank(dist: np.ndarray, delivering: list[int]) -> "int | None":
    order = list(np.argsort(dist))
    ranks = sorted(order.index(i) + 1 for i in delivering)
    return ranks[0] if ranks else None


def _retrieval_and_coverage(recs: list[dict]) -> dict[str, Any]:
    """LOO teacher-free retrieval + θ×handoff coverage/rank on the captured snapshots, for both the SPARSE bank
    (best-θ per scenario) and the DENSE bank (all K6 θ)."""
    cap = [r for r in recs if r.get("captured")]
    k6s = [r for r in cap if r.get("teacher_k6")]
    if len(k6s) < 2:
        return {"note": "insufficient teacher-K6 samples for retrieval", "n_teacher_k6": len(k6s)}
    X = np.asarray([r["x"] for r in k6s], np.float64)
    thetas = [np.asarray(r["theta"], np.float64) for r in k6s]
    scen_of = [r["scenario"] for r in k6s]
    # sparse bank indices: the min-teacher-dtz sample per scenario
    best_by_scen: dict[str, int] = {}
    for i, r in enumerate(k6s):
        if r["scenario"] not in best_by_scen or r["teacher_dtz"] < k6s[best_by_scen[r["scenario"]]]["teacher_dtz"]:
            best_by_scen[r["scenario"]] = i
    sparse_idx = set(best_by_scen.values())

    def audit(bank_idx: list[int], tag: str) -> dict[str, Any]:
        if len(bank_idx) < 2:
            return {"n_bank": len(bank_idx), "note": "bank too small"}
        std = Standardizer.fit(X[bank_idx])
        retr_k6 = cover = 0
        ranks: list[int] = []
        for j, r in enumerate(cap):
            snap = r["_snap"]
            loo = [i for i in bank_idx if scen_of[i] != r["scenario"]]   # held-out: exclude same-scenario θ
            if not loo:
                continue
            xj = np.asarray(r["x"], np.float64)
            # retrieval: standardized nearest over the LOO bank
            d = np.linalg.norm(std.transform(X[loo]) - std.transform(xj[None, :])[0], axis=1)
            pick = thetas[loo[int(np.argmin(d))]]
            retr_k6 += int(_delivery_signals(snap, clip_theta(pick)).k6)
            # coverage/rank: which LOO θ deliver on this snapshot
            deliver = [k for k, i in enumerate(loo) if _delivery_signals(snap, clip_theta(thetas[i])).k6]
            if deliver:
                cover += 1
                ranks.append(_rank(np.linalg.norm(std.transform(X[loo]) - std.transform(xj[None, :])[0], axis=1),
                                   deliver) or 0)
        n = len(cap)
        return {"tag": tag, "n_bank": len(bank_idx), "n_eval": n,
                "retrieval_k6": retr_k6, "coverage": cover,
                "delivering_median_rank": (float(np.median(ranks)) if ranks else None),
                "bank_size_for_rank": len(bank_idx)}

    return {"sparse": audit(sorted(sparse_idx), "sparse"), "dense": audit(list(range(len(k6s))), "dense"),
            "n_teacher_k6": len(k6s)}


def _phase_row(variant_id: str, recs: list[dict], retr: dict) -> dict[str, Any]:
    n = len(recs)
    reached = sum(1 for r in recs if r["reached"])
    captured = sum(1 for r in recs if r.get("captured"))
    tk6 = sum(1 for r in recs if r.get("teacher_k6"))
    sp, de = retr.get("sparse", {}), retr.get("dense", {})
    return {
        "variant": variant_id, "n_rollouts": n,
        "reach_rate": round(reached / n, 3),
        "certified_capture_rate": round(captured / reached, 3) if reached else None,
        "teacher_k6_given_capture": round(tk6 / captured, 3) if captured else None,
        "retrieval_k6_given_capture_sparse": (round(sp.get("retrieval_k6", 0) / captured, 3) if captured and sp else None),
        "retrieval_k6_given_capture_dense": (round(de.get("retrieval_k6", 0) / captured, 3) if captured and de else None),
        "coverage_sparse": sp.get("coverage"), "coverage_dense": de.get("coverage"),
        "delivering_median_rank_dense": de.get("delivering_median_rank"),
        "n_teacher_k6": tk6, "n_captured": captured, "n_reached": reached,
    }


def main() -> int:
    variant_id = sys.argv[1] if len(sys.argv) > 1 else "O1-L"
    global _N_SEEDS
    _N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    _OUT.mkdir(parents=True, exist_ok=True)
    print(f"characterize {variant_id}: {len(_SCENARIOS)} scenarios × {_N_SEEDS} seeds", flush=True)
    cfg, conf, obj = bc_context()
    rig = _rig(object_spec=variant(variant_id).object_spec)
    recs = _capture_and_teacher(rig, cfg, conf, obj)
    retr = _retrieval_and_coverage(recs)
    row = _phase_row(variant_id, recs, retr)
    out = {"phase_row": row, "retrieval": retr,
           "records": [{k: v for k, v in r.items() if k != "_snap"} for r in recs]}
    (_OUT / f"characterization_{variant_id}.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"\n=== {variant_id} phase row ===")
    for k, v in row.items():
        print(f"  {k}: {v}")
    print(f"wrote {_OUT / f'characterization_{variant_id}.json'}")
    return 0


_N_SEEDS = 5

if __name__ == "__main__":
    sys.exit(main())
