"""K1 cost-audit — 16 representative relabels over the KINETIC-entry neighbourhood (NO 32-bank generated).

Before sizing any feedback bank, measure the receding-horizon relabel on a small, deliberately-diverse neighbourhood of the
frozen KINETIC entry (4 easy / 8 medium / 4 edge, all legal perturbed-control branches, admissibility-gated). Same contract as
K0: warm-start = the squeeze≈0 entry-delivering θ; search = a box-wide legal CEM; label = first executed action only; actor
feature = the canonical 41-D observation only; s4/s7 untouched; f1–f4 sealed. The teacher/CEM is used OFFLINE to label; nothing
is deployed. This is a measurement + a go/no-go on K1-A (32 labels) — it does NOT start BC.

Pre-registered brakes (evaluated in `_verdict`):
  * p90 relabel-time > 2 × the calibrated (entry) baseline  ⇒ STOP, do not start K1-A.
  * a large fraction of replans fail to progress             ⇒ STOP, localise perturbation-admissibility / warm-start first.
  * successful, but the first actions collapse to ~a constant ⇒ REVIEW: verify the bank carries feedback information.
Green only if cost is acceptable, replanning is stable, and the first-action labels genuinely vary with the state.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_k1_cost_audit`` (under /usr/bin/time -l for RSS).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.theta_option import kinetic_bank as kb
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

OUT = Path("reports/2026-07-27-coin-r9-learned-s1-kinetic-delivery-k0/k1_cost_audit.json")
# box-wide legal window around the (squeeze≈0) entry-delivering θ — effectively the full DELIVERY_CFG box, warm-started.
WIDE_WINDOW = (0.25, 0.30, 0.12, 16.0, 25.0, 2.5)
POP, ITERS = 32, 6                                          # the K0-calibrated full-CEM budget (~17 s/label anchor)
PROGRESS_MM = 40.0                                         # min_dtz below this ⇒ "successful replan" (past the ~50 mm scaffold wall)


def _relabel(tsnap: Any) -> "tuple[Any, float]":
    """One box-wide warm-started relabel; returns (RelabelResult, wall_seconds)."""
    t = time.perf_counter()
    r = kc.receding_horizon_relabel(tsnap, warm_theta=kc.S1_CANONICAL_THETA, budget=1,
                                    window=WIDE_WINDOW, pop=POP, iters=ITERS)
    return r, time.perf_counter() - t


def _termination(delivers: bool, min_dtz: float, admissible: bool) -> str:
    if not admissible:
        return "inadmissible"
    if delivers:
        return "delivered"
    return "progress_no_delivery" if min_dtz < PROGRESS_MM else "no_progress"


def _pairwise_diversity(actions: np.ndarray) -> float:
    """Mean pairwise L2 distance between the (normalised) first-action vectors — ~0 ⇒ the labels collapsed to a constant."""
    n = len(actions)
    if n < 2:
        return 0.0
    d = [float(np.linalg.norm(actions[i] - actions[j])) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(d))


def _spread(descriptors: list[dict], key: str) -> dict:
    v = np.array([d[key] for d in descriptors], np.float64)
    return {"min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "std": round(float(v.std()), 4)}


def _verdict(baseline_s: float, walls: list[float], n_success: int, n_k6: int, n: int, fa_diversity: float,
             fa_scale: float) -> dict:
    """Pre-registered go/no-go. Diversity is judged relative to the action scale (mean |first-action| across the set)."""
    p90 = float(np.percentile(walls, 90)) if walls else 0.0
    cost_ok = bool(p90 <= 2.0 * baseline_s)
    replan_ok = bool(n_success >= (n + 1) // 2)             # at least half progress past the scaffold wall
    collapsed = bool(fa_diversity < 0.05 * max(fa_scale, 1e-6))   # first actions ~a single constant
    if not cost_ok:
        gate = "STOP_COST_EXCEEDED"
    elif not replan_ok:
        gate = "STOP_REPLANNING_UNSTABLE"
    elif collapsed:
        gate = "REVIEW_FIRST_ACTION_COLLAPSE"
    else:
        gate = "K1_RELABEL_COST_AND_DIVERSITY_GATE_PASS"
    return {"gate": gate, "p90_s": round(p90, 2), "baseline_s": round(baseline_s, 2),
            "p90_over_2x_baseline": bool(p90 > 2.0 * baseline_s), "cost_ok": cost_ok, "replan_ok": replan_ok,
            "first_action_collapsed": collapsed, "n_success": n_success, "n_k6": n_k6, "n": n,
            "first_action_pairwise_diversity": round(fa_diversity, 5), "first_action_scale": round(fa_scale, 5)}


def run() -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    entry = kc.freeze_kinetic_entry(snap)

    _r0, baseline_s = _relabel(entry.tsnap)                 # calibrated baseline: relabel the entry itself
    accepted, rejected = kb.generate_neighbourhood(entry.tsnap)

    rows: list[dict] = []
    walls: list[float] = []
    first_actions: list[np.ndarray] = []
    for a in accepted:
        r, wall = _relabel(a["tsnap"])
        min_dtz = _min_dtz_mm(a["tsnap"], r.metrics)
        walls.append(wall)
        first_actions.append(np.asarray(r.first_action_norm, np.float64))
        rows.append({**a["provenance"], "descriptor": a["descriptor"], "wall_s": round(wall, 2),
                     "delivers_k6": r.delivers_k6, "min_dtz_mm": round(min_dtz, 2),
                     "termination": _termination(r.delivers_k6, min_dtz, a["provenance"]["admissible"]),
                     "theta": r.theta, "first_action_norm": [round(float(x), 5) for x in r.first_action_norm]})

    # deterministic repeat: 2 selected labels must bit-replay
    repeats = []
    for i in (0, len(accepted) // 2):
        if i < len(accepted):
            r_a, _ = _relabel(accepted[i]["tsnap"])
            r_b, _ = _relabel(accepted[i]["tsnap"])
            repeats.append({"label": accepted[i]["provenance"]["label"],
                            "max_abs_first_action_diff": float(np.max(np.abs(r_a.first_action - r_b.first_action)))})

    fa = np.array(first_actions) if first_actions else np.zeros((0, 4))
    fa_diversity = _pairwise_diversity(fa)
    fa_scale = float(np.mean(np.abs(fa))) if fa.size else 0.0
    n = len(rows)
    n_k6 = sum(1 for x in rows if x["delivers_k6"])
    n_success = sum(1 for x in rows if x["termination"] in ("delivered", "progress_no_delivery"))
    descriptors = [x["descriptor"] for x in rows]
    verdict = _verdict(baseline_s, walls, n_success, n_k6, n, fa_diversity, fa_scale)

    out = {"contract": "COIN_KINETIC_K1_COST_AUDIT_V1", "seed": kc.S1_SEED, "entry": entry.summary(),
           "n_accepted": n, "n_rejected": len(rejected), "rejected": rejected,
           "wall_per_label_s": {"median": round(median(walls), 2) if walls else 0.0,
                                "p90": round(float(np.percentile(walls, 90)), 2) if walls else 0.0,
                                "max": round(max(walls), 2) if walls else 0.0, "baseline": round(baseline_s, 2)},
           "successful_replans": n_success, "k6_delivering_replans": n_k6,
           "termination_reasons": {k: sum(1 for x in rows if x["termination"] == k)
                                   for k in ("delivered", "progress_no_delivery", "no_progress", "inadmissible")},
           "state_spread": {k: _spread(descriptors, k) for k in ("v_par", "slip", "fn_min", "fn_imbalance", "dtz_mm")},
           "first_action": {"pairwise_diversity": round(fa_diversity, 5), "scale": round(fa_scale, 5),
                            "mean": [round(float(x), 5) for x in fa.mean(0)] if fa.size else [],
                            "std": [round(float(x), 5) for x in fa.std(0)] if fa.size else [],
                            "range": [round(float(fa[:, j].max() - fa[:, j].min()), 5) for j in range(4)] if fa.size else []},
           "deterministic_repeat": repeats, "verdict": verdict, "rows": rows, "wall_s": round(time.time() - t0, 1)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = run()
    v = res["verdict"]
    w = res["wall_per_label_s"]
    print(f"\nGATE: {v['gate']}  (wall/label med {w['median']}s p90 {w['p90']}s max {w['max']}s vs baseline {w['baseline']}s; "
          f"total {res['wall_s']}s)\n")
    print(f"  successful replans {res['successful_replans']}/{res['n_accepted']}  |  K6-delivering "
          f"{res['k6_delivering_replans']}/{res['n_accepted']}  |  rejected {res['n_rejected']}")
    print(f"  termination: {res['termination_reasons']}")
    print(f"  first-action diversity {res['first_action']['pairwise_diversity']} (scale {res['first_action']['scale']})  "
          f"range {res['first_action']['range']}")
    print(f"  state spread v_par {res['state_spread']['v_par']}  slip {res['state_spread']['slip']}  "
          f"fn_min {res['state_spread']['fn_min']}  imbal {res['state_spread']['fn_imbalance']}")
    print(f"  det-repeat max|Δfirst_action|: {[round(r['max_abs_first_action_diff'],2) for r in res['deterministic_repeat']]}")
    for r in res["rows"]:
        print(f"    {r['category']:6s} {r['label']:14s} {r['wall_s']:5.1f}s K6={r['delivers_k6']!s:5s} "
              f"min_dtz={r['min_dtz_mm']:6.1f} {r['termination']:20s} fa={r['first_action_norm']}")
