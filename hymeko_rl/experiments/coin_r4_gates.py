"""R4 closed-loop basin-aware intent-correction gates C0/C1/C2 — one harness (§6.5 #13: modes, not v-files).

Runs, in a single panel build, the three frozen R4 gates against the coast-in closed-loop controller
(`theta_option/closed_loop_{state,intent,rollout}.py`) over the frozen R3 decoder:

  * C0 — teacher no-regression: the closed-loop on the four teacher intents must keep K6 = 4/4 (a correct feedback law is a
    no-op on an already-delivering plan).
  * C1 — development update-0: the R3 predictor's intent as the initial proposal + the R4 correction; dev (s1,s3) = 2/2.
  * C2 — one frozen panel (s1,s3,s4,s7): the hard gate 4/4 incl. held-out 2/2.

For every state the closed-loop (CL) result is reported alongside the open-loop R3 baseline (OL, `decode → fixed_search`)
and the oracle (teacher intent + the same open-loop search) — the honesty controls: the CL is *load-bearing* only if it
strictly beats OL, and a held-out miss is attributable to the correction (not the search/physics) only if the oracle
delivers. Frozen dev-selected params below; `R4_PARAMS` (json env) overrides for a re-sweep. Reuses the R3 predictor path
(`coin_theta_rl_benchmark._r2_dev_data / _r3_dev_dataset`) and the closed-loop search — no re-implementation.
"""
from __future__ import annotations

import json
import os
import resource
import time
from typing import Any

import numpy as np

import hymeko_rl.experiments.coin_theta_rl_benchmark as B
from hymeko_rl.coin_delivery.theta_option.authority_decoder import decode_intent
from hymeko_rl.coin_delivery.theta_option.canonical_frame import canonicalise, flatten_r1, r1_grouped_features
from hymeko_rl.coin_delivery.theta_option.closed_loop_intent import CorrectionParams, IntentCorrector
from hymeko_rl.coin_delivery.theta_option.closed_loop_rollout import closed_loop_search_select
from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
from hymeko_rl.coin_delivery.theta_option.intent_predictor import fit_intent_predictor
from hymeko_rl.coin_delivery.theta_option.physical_intent import extract_teacher_intent
from hymeko_rl.coin_delivery.theta_option.search import SEARCH_STD, fixed_search_select
from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness

OUT = "reports/2026-07-27-coin-r4-closed-loop-intent"
REPORT_DIR = "reports/2026-07-27-coin-teacher-to-rl"
# FROZEN dev-selected controller params (best dev K6 in the dev-only sweep; passthrough OFF — no authority regime split).
FROZEN_PARAMS = {"a_friction": 0.55, "k_forward_deficit": 2.5}
BUDGET = 8
SEED = 90000


def _cl_deploy(snap: Any, intent: Any, params: CorrectionParams) -> dict:
    """Closed-loop deploy on ONE cradle: budget-8 centre-inclusive search over the coast-in closed-loop option."""
    return closed_loop_search_select(snap, intent, IntentCorrector(params), params, np.random.default_rng(SEED),
                                     budget=BUDGET, cfg=DELIVERY_CFG)


def _ol_deploy(snap: Any, intent: Any) -> dict:
    """Open-loop R3 baseline: decode → fixed_search (the honesty control)."""
    dec = decode_intent(intent, snap, DELIVERY_CFG)
    prov = fixed_search_select(snap, np.asarray(dec.physical_theta, np.float64), np.random.default_rng(SEED),
                               budget=BUDGET, cfg=DELIVERY_CFG, std=SEARCH_STD)
    o = prov.outcome
    return {"delivery_success": bool(o["delivery_success"]), "dtz_end_mm": round(o["dtz_end"] * 1000, 2)}


def _pred_intent(predictor: Any, snap: Any) -> Any:
    g, _ = canonicalise(r1_grouped_features(snap))
    return predictor.predict(flatten_r1(g))


def _teacher_intent(snap: Any, theta: Any) -> Any:
    return extract_teacher_intent(snap, np.asarray(theta, np.float64))[0]


def _split_counts(per: dict) -> "tuple[int, int, int]":
    dev = sum(1 for r in per.values() if r["split"] == "development" and r["delivery_success"])
    hel = sum(1 for r in per.values() if r["split"] == "held_out" and r["delivery_success"])
    return dev, hel, dev + hel


def run_gates(params: CorrectionParams) -> dict:
    """Build the R3 predictor + the frozen panel once, then run C0/C1/C2 and write the artifacts. # Postconditions: emits
    c0/c1/c2/response_traces/controller_parameters json; returns the C2 summary with the honest verdict."""
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    bank = json.load(open(f"{REPORT_DIR}/teacher_bank.json"))
    dp = json.load(open(f"{REPORT_DIR}/cradle_delivery_pass.json"))
    dev_tag, dev_seeds, dev_canon = B._r2_dev_data(bank, dp)
    harness = load_harness()
    _snaps, feats, intents, _prov = B._r3_dev_dataset(harness, dev_tag, dev_seeds, dev_canon)
    predictor = fit_intent_predictor(feats, intents, bandwidth=3.0)
    panel = build_panel(harness, bank)
    print(f"built {time.time() - t0:.1f}s", flush=True)

    # ── C0 — teacher no-regression ──
    c0 = {}
    for ps in panel:
        r = _cl_deploy(ps.snap, _teacher_intent(ps.snap, ps.teacher_theta), params)
        c0[ps.tag] = {"split": ps.split, "delivery_success": bool(r["delivery_success"]),
                      "k6_max_dwell": r["k6_max_dwell"], "dtz_end_mm": r["dtz_end_mm"], "peak_qdot": r["peak_qdot"],
                      "peak_coin_speed": r["peak_coin_speed"], "n_corrections": r["n_corrections"]}
    c0dev, c0hel, c0tot = _split_counts(c0)
    c0_ok = c0tot == 4
    json.dump({"contract": "COIN_R4_C0_TEACHER_NO_REGRESSION", "params": params.as_dict(), "per_state": c0,
               "dev_k6": c0dev, "held_out_k6": c0hel, "total_k6": c0tot, "passed": c0_ok,
               "verdict": "CLOSED_LOOP_TEACHER_NO_REGRESSION_PASS" if c0_ok else "CORRECTION_LAW_DESTABILISES_TEACHER"},
              open(f"{OUT}/c0_teacher_no_regression.json", "w"), indent=1, default=float)
    print(f"C0 teacher: {c0tot}/4 -> {'PASS' if c0_ok else 'FAIL'}", flush=True)

    # ── C1 (dev) + C2 (panel) — predictor initial + closed-loop; OL + oracle honesty controls ──
    per = {}
    for ps in panel:
        pi, ti = _pred_intent(predictor, ps.snap), _teacher_intent(ps.snap, ps.teacher_theta)
        cl, ol, orc = _cl_deploy(ps.snap, pi, params), _ol_deploy(ps.snap, pi), _ol_deploy(ps.snap, ti)
        per[ps.tag] = {"split": ps.split, "delivery_success": bool(cl["delivery_success"]),
                       "k6_max_dwell": cl["k6_max_dwell"], "dtz_start_mm": cl["dtz_start_mm"], "dtz_end_mm": cl["dtz_end_mm"],
                       "peak_qdot": cl["peak_qdot"], "peak_coin_speed": cl["peak_coin_speed"],
                       "n_corrections": cl["n_corrections"], "release_step": cl["release_step"], "theta0": cl["theta0"],
                       "theta_exec": cl["theta_exec"], "search_displacement_norm": cl["search_displacement_norm"],
                       "budget_total": cl["budget_total"], "base_intent": cl["base_intent"], "final_intent": cl["final_intent"],
                       "open_loop_delivery": ol["delivery_success"], "open_loop_dtz_end_mm": ol["dtz_end_mm"],
                       "oracle_delivery": orc["delivery_success"], "corrections": cl["corrections"], "responses": cl["responses"]}
        print(f"   {ps.tag}[{ps.split[:3]}] CL={per[ps.tag]['delivery_success']}(dtz{per[ps.tag]['dtz_end_mm']:.0f}) "
              f"| OL={ol['delivery_success']}(dtz{ol['dtz_end_mm']:.0f}) | ORC={orc['delivery_success']}", flush=True)

    dev, hel, tot = _split_counts(per)
    ol_dev = sum(1 for r in per.values() if r["split"] == "development" and r["open_loop_delivery"])
    ol_hel = sum(1 for r in per.values() if r["split"] == "held_out" and r["open_loop_delivery"])
    orc_ok = all(r["oracle_delivery"] for r in per.values())
    motion_ok = all(r["peak_qdot"] <= 3.0 and r["peak_coin_speed"] <= 1.5 for r in per.values())
    budget_ok = all(r["budget_total"] <= BUDGET for r in per.values())
    load_bearing = tot > (ol_dev + ol_hel)
    if tot == 4 and dev == 2 and hel == 2 and motion_ok and budget_ok and c0_ok:
        verdict = "CLOSED_LOOP_INTENT_CORRECTION_LOAD_BEARING"
    elif tot == 3 or hel == 1:
        verdict = "CLOSED_LOOP_FEEDBACK_IMPROVES_GENERALISATION_BUT_GATE_OPEN"
    else:
        verdict = "CURRENT_DETERMINISTIC_FEEDBACK_LAW_INSUFFICIENT"
    authorises = bool(verdict == "CLOSED_LOOP_INTENT_CORRECTION_LOAD_BEARING")

    json.dump({"contract": "COIN_R4_C1_DEVELOPMENT", "params": params.as_dict(),
               "dev_states": {t: per[t] for t in per if per[t]["split"] == "development"},
               "dev_k6": dev, "passed": dev == 2}, open(f"{OUT}/c1_development.json", "w"), indent=1, default=float)
    c2 = {"contract": "COIN_R4_C2_FROZEN_PANEL", "params": params.as_dict(), "per_state": per,
          "closed_loop": {"dev_k6": dev, "held_out_k6": hel, "total_k6": tot},
          "open_loop_baseline": {"dev_k6": ol_dev, "held_out_k6": ol_hel, "total_k6": ol_dev + ol_hel},
          "oracle_validates_search_and_physics": orc_ok, "closed_loop_load_bearing": load_bearing, "c0_pass": c0_ok,
          "motion_ok": motion_ok, "budget_ok": budget_ok, "verdict": verdict, "authorises_sac_td3": authorises}
    json.dump(c2, open(f"{OUT}/c2_frozen_panel.json", "w"), indent=1, default=float)
    json.dump({t: {"corrections": per[t]["corrections"], "responses": per[t]["responses"]} for t in per},
              open(f"{OUT}/response_traces.json", "w"), indent=1, default=float)
    json.dump(params.as_dict(), open(f"{OUT}/controller_parameters.json", "w"), indent=1, default=float)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"\n== R4 GATES ==\n  C0 {c0tot}/4 ({'PASS' if c0_ok else 'FAIL'}) | C1 dev {dev}/2 | "
          f"C2 CL {tot}/4 (held {hel}/2) | OL {ol_dev + ol_hel}/4 | oracle_ok={orc_ok} | load_bearing={load_bearing}\n"
          f"  VERDICT: {verdict} | authorises_sac_td3={authorises}\n  peak RSS {rss:.2f} GB, wall {time.time() - t0:.1f}s\n"
          f"R4_GATES_DONE", flush=True)
    return c2


def main() -> None:
    params = CorrectionParams(**{**FROZEN_PARAMS, **json.loads(os.environ.get("R4_PARAMS", "{}"))})
    run_gates(params)


if __name__ == "__main__":
    main()
