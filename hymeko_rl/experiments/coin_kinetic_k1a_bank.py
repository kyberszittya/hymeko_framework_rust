"""K1-A — build the first KINETIC feedback bank: 32 accepted first-action labels, ≤ 48 attempts.

Green-lit by the K1 cost/diversity gate (`K1_RELABEL_COST_AND_DIVERSITY_GATE_PASS`). Each attempt is a legal perturbed-control
branch of the frozen KINETIC entry (no state edit). Curation is by the TEACHER-REPLAN RESULT, not a manual v_par threshold:

  * admissible ∧ successful causal replan (progress/deliver) ∧ valid first action  → a FEEDBACK LABEL (feedback_labels)
  * admissible ∧ no progressing/delivering continuation                            → a TERMINAL-FAILURE record (kept, separate)
  * inadmissible branch                                                            → a search failure (logged)

The near-zero "do nothing" first actions of stalled states are NEVER emitted as feedback labels (they would teach the coin to
stop); the terminal-failure states are retained separately as negative information (stiction-risk guard / avoidance classifier /
DAgger early-stop / drift measurement). The replan's terminal result is used for dataset curation ONLY and never enters the
actor's 41-D input. Contract: warm-start = the entry-delivering θ; search = box-wide legal CEM; label = first executed action
only; s4/s7 untouched; f1–f4 sealed. Stops with a coverage report if 48 attempts do not yield 32 usable labels — no silent
dilution of admissibility.

Run: ``python -m hymeko_rl.experiments.coin_kinetic_k1a_bank`` (writes feedback_labels/terminal_failure_states/manifest).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from hymeko_rl.coin_delivery.theta_option import kinetic_bank as kb
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.experiments.coin_kinetic_k1_cost_audit import _pairwise_diversity, _relabel, _spread, _termination
from hymeko_rl.experiments.coin_kinetic_positive_control import _min_dtz_mm

BANK_DIR = Path("reports/2026-07-28-coin-r9-k1a-bank")
TARGET_ACCEPTED = 32
MAX_ATTEMPTS = 48
_DESC_KEYS = ("v_par", "slip", "fn_min", "fn_imbalance", "dtz_mm")   # the state axes for the distinct-state / coverage check


def _distinct_states(descriptors: list[dict]) -> dict:
    """Distinct-state audit: pairwise L2 over standardised descriptors — min/mean pairwise + a near-duplicate count. A low min
    pairwise / high near-dup count ⇒ the bank is dominated by a few near-identical (e.g. teacher-trace) states."""
    if len(descriptors) < 2:
        return {"min_pairwise": 0.0, "mean_pairwise": 0.0, "near_duplicates": 0}
    x = np.array([[d[k] for k in _DESC_KEYS] for d in descriptors], np.float64)
    z = (x - x.mean(0)) / (x.std(0) + 1e-9)
    d = [float(np.linalg.norm(z[i] - z[j])) for i in range(len(z)) for j in range(i + 1, len(z))]
    return {"min_pairwise": round(min(d), 4), "mean_pairwise": round(float(np.mean(d)), 4),
            "near_duplicates": int(sum(1 for v in d if v < 0.25))}


def _bank_hash(feedback: list[dict]) -> str:
    """Content hash of the bank (observation + first-action label per record) — reproducibility anchor."""
    payload = json.dumps([[r["obs"], r["first_action_norm"]] for r in feedback], sort_keys=True).encode()
    return hashlib.md5(payload).hexdigest()[:16]


def _verdict(n_feedback: int, fa: np.ndarray, distinct: dict) -> str:
    if n_feedback < TARGET_ACCEPTED:
        return "K1A_COVERAGE_SHORTFALL"
    diversity = _pairwise_diversity(fa)
    scale = float(np.mean(np.abs(fa))) if fa.size else 0.0
    if diversity < 0.05 * max(scale, 1e-6) or distinct["near_duplicates"] > n_feedback:
        return "K1A_REVIEW_DIVERSITY"
    return "K1A_BANK_READY"


def _feedback_record(rec: dict, r: Any, wall: float, min_dtz: float) -> dict:
    return {**rec["provenance"], "descriptor": rec["descriptor"], "obs": [round(float(x), 6) for x in rec["obs"]],
            "first_action": [round(float(x), 6) for x in r.first_action],
            "first_action_norm": [round(float(x), 6) for x in r.first_action_norm],
            "delivers_k6": bool(r.delivers_k6), "min_dtz_mm": round(min_dtz, 2), "replan_theta": r.theta,
            "wall_s": round(wall, 2)}


def _terminal_record(rec: dict, r: Any, wall: float, min_dtz: float, term: str) -> dict:
    return {**rec["provenance"], "descriptor": rec["descriptor"], "oracle_min_dtz_mm": round(min_dtz, 2),
            "oracle_delivers_k6": bool(r.delivers_k6), "failure_reason": term,
            "first_action_norm_provenance": [round(float(x), 6) for x in r.first_action_norm], "wall_s": round(wall, 2)}


def run() -> dict:
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    t0 = time.time()
    harness = load_harness()
    snap, meta = acquire_snapshot(harness, kc.S1_SEED)
    if snap is None:
        raise SystemExit(f"could not acquire s1 straddle: {meta}")
    entry = kc.freeze_kinetic_entry(snap)
    specs = kb.sample_specs(MAX_ATTEMPTS)

    feedback: list[dict] = []
    terminal: list[dict] = []
    inadmissible: list[dict] = []
    fb_states: list[dict] = []                                  # keep the TransportSnapshots for the bit-replay check
    walls: list[float] = []
    attempts = 0
    for i, spec in enumerate(specs):
        if len(feedback) >= TARGET_ACCEPTED:
            break
        attempts += 1
        rec, prov = kb.labeled_state(entry.tsnap, spec, seed=20260728 + 101 * i)
        if rec is None:
            inadmissible.append(prov)
            continue
        r, wall = _relabel(rec["tsnap"])
        walls.append(wall)
        min_dtz = _min_dtz_mm(rec["tsnap"], r.metrics)
        term = _termination(r.delivers_k6, min_dtz, True)
        if term in ("delivered", "progress_no_delivery") and bool(np.all(np.isfinite(r.first_action))):
            feedback.append(_feedback_record(rec, r, wall, min_dtz))
            fb_states.append({"tsnap": rec["tsnap"], "label": spec.label})
        else:
            terminal.append(_terminal_record(rec, r, wall, min_dtz, term))

    # deterministic repeat: 2 accepted labels must bit-replay in the final bank
    repeats = []
    for j in (0, len(fb_states) // 2):
        if j < len(fb_states):
            ra, _ = _relabel(fb_states[j]["tsnap"])
            rb, _ = _relabel(fb_states[j]["tsnap"])
            repeats.append({"label": fb_states[j]["label"],
                            "max_abs_first_action_diff": float(np.max(np.abs(ra.first_action - rb.first_action)))})

    fa = np.array([r["first_action_norm"] for r in feedback], np.float64) if feedback else np.zeros((0, 4))
    descriptors = [r["descriptor"] for r in feedback]
    distinct = _distinct_states(descriptors)
    n_fb = len(feedback)
    cat_counts = {c: sum(1 for r in feedback if r["category"] == c) for c in ("easy", "medium", "edge")}
    manifest = {
        "contract": "COIN_KINETIC_K1A_BANK_V1", "seed": kc.S1_SEED, "entry": entry.summary(),
        "target_accepted": TARGET_ACCEPTED, "max_attempts": MAX_ATTEMPTS, "attempts_used": attempts,
        "n_feedback": n_fb, "n_terminal_failure": len(terminal), "n_inadmissible": len(inadmissible),
        "category_counts": cat_counts,
        "k6_delivering_labels": sum(1 for r in feedback if r["delivers_k6"]),
        "cost_s": {"median": round(median(walls), 2) if walls else 0.0,
                   "p90": round(float(np.percentile(walls, 90)), 2) if walls else 0.0,
                   "max": round(max(walls), 2) if walls else 0.0},
        "state_spread": {k: _spread(descriptors, k) for k in _DESC_KEYS} if descriptors else {},
        "distinct_states": distinct,
        "first_action": {"pairwise_diversity": round(_pairwise_diversity(fa), 5),
                         "scale": round(float(np.mean(np.abs(fa))), 5) if fa.size else 0.0,
                         "mean": [round(float(x), 5) for x in fa.mean(0)] if fa.size else [],
                         "std": [round(float(x), 5) for x in fa.std(0)] if fa.size else [],
                         "range": [round(float(fa[:, k].max() - fa[:, k].min()), 5) for k in range(4)] if fa.size else []},
        "bank_hash": _bank_hash(feedback), "deterministic_repeat": repeats,
        "verdict": _verdict(n_fb, fa, distinct), "wall_s": round(time.time() - t0, 1)}

    BANK_DIR.mkdir(parents=True, exist_ok=True)
    (BANK_DIR / "feedback_labels.json").write_text(json.dumps(feedback, indent=1))
    (BANK_DIR / "terminal_failure_states.json").write_text(json.dumps(terminal, indent=1))
    (BANK_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    m = run()
    print(f"\nVERDICT: {m['verdict']}  (feedback {m['n_feedback']}/{m['target_accepted']} in {m['attempts_used']} attempts; "
          f"terminal {m['n_terminal_failure']}; inadmissible {m['n_inadmissible']}; total {m['wall_s']}s)\n")
    print(f"  category mix {m['category_counts']}  |  K6-delivering labels {m['k6_delivering_labels']}/{m['n_feedback']}")
    print(f"  cost/label median {m['cost_s']['median']}s p90 {m['cost_s']['p90']}s max {m['cost_s']['max']}s")
    print(f"  first-action diversity {m['first_action']['pairwise_diversity']} (scale {m['first_action']['scale']})  "
          f"range {m['first_action']['range']}")
    print(f"  distinct-states {m['distinct_states']}  |  state_spread v_par {m['state_spread'].get('v_par')}")
    print(f"  bank_hash {m['bank_hash']}  det-repeat {[round(r['max_abs_first_action_diff'],2) for r in m['deterministic_repeat']]}")
