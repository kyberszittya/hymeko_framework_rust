"""Coverage-only causal curve — tests for the PURE coverage logic (fast, no physics) + the reproduce_state dev-override.

The pure functions decide the experiment's scientific validity: the N=4 selection must be deterministic and NON-outcome
(geometry only), the dev sets must be nested (so N is the sole independent variable), the actor→teacher distance must be
the mechanistic probe, and the verdict must follow the frozen decision tree (gate reached → authorise; N=6 misses →
coverage-alone-insufficient; monotone held-out improvement short of the gate → the positive mechanistic finding).
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option.coverage_curve import (
    FROZEN_DEV_SEEDS, coverage_record, coverage_verdict, nested_dev_sets, select_n4_additions, theta_distance_summary)
from hymeko_rl.coin_delivery.theta_option.semantics import DIM, ThetaBox


# ───────────────────────────── N=4 selection (non-outcome, deterministic) ─────────────────────────────
def _fps(**kv):
    return {s: np.asarray(v, np.float64) for s, v in kv.items()}


def test_select_n4_additions_picks_the_geometrically_farthest_two():
    # frozen dev at origin-ish; candidates at increasing distance — FPS must pick the two FARTHEST from the frozen set
    fps = _fps(**{"14250": [0.0, 0.0], "14750": [0.1, 0.0],
                  "16500": [0.3, 0.0], "17750": [5.0, 0.0], "19500": [9.0, 0.0], "24000": [0.35, 0.0]})
    fps = {int(k): v for k, v in fps.items()}
    sel = select_n4_additions([16500, 17750, 19500, 24000], fps, FROZEN_DEV_SEEDS, k=2)
    assert sel["non_outcome_based"] is True
    assert sel["selected"] == [19500, 17750]                 # farthest first (greedy), then next-farthest from the set
    # rule reads ONLY fingerprints — there is no outcome/K6 argument in the signature
    assert "delivery" not in sel["rule"] and "k6" not in sel["rule"]


def test_select_n4_additions_is_deterministic_and_tie_breaks_to_lower_seed():
    fps = {14250: np.array([0.0, 0.0]), 14750: np.array([0.0, 0.0]),
           16500: np.array([1.0, 0.0]), 17750: np.array([-1.0, 0.0]),   # equidistant (‖·‖=1) from frozen dev
           19500: np.array([0.2, 0.0]), 24000: np.array([0.25, 0.0])}
    a = select_n4_additions([16500, 17750, 19500, 24000], fps, FROZEN_DEV_SEEDS, k=1)
    b = select_n4_additions([16500, 17750, 19500, 24000], fps, FROZEN_DEV_SEEDS, k=1)
    assert a["selected"] == b["selected"]                    # deterministic
    assert a["selected"] == [16500]                          # tie (16500 vs 17750 both dist 1.0) → lower seed


def test_nested_dev_sets_are_strictly_nested():
    sets = nested_dev_sets([16500, 17750, 19500, 24000], [19500, 17750], FROZEN_DEV_SEEDS)
    assert sets[2] == [14250, 14750]
    assert set(sets[2]) <= set(sets[4]) <= set(sets[6])      # N=2 ⊂ N=4 ⊂ N=6
    assert sets[4] == [14250, 14750, 19500, 17750]
    assert sorted(sets[6]) == sorted([14250, 14750, 16500, 17750, 19500, 24000])
    assert len(sets[2]) == 2 and len(sets[4]) == 4 and len(sets[6]) == 6


# ───────────────────────────── actor→teacher θ distance ─────────────────────────────
def test_theta_distance_summary_matches_manual_l2_and_splits():
    box = ThetaBox()
    teacher = box.denorm(np.zeros(DIM))                      # box centre
    proposed = teacher.copy()
    rows = [{"tag": "s1", "split": "development", "proposed": proposed, "teacher": teacher},
            {"tag": "s4", "split": "held_out", "proposed": proposed, "teacher": teacher}]
    d0 = theta_distance_summary(rows, box)
    assert d0["per_state"]["s1"]["theta_l2_norm"] == 0.0     # identical θ ⇒ zero distance
    assert d0["mean_theta_l2_norm_by_split"]["held_out"] == 0.0
    # perturb the held-out proposal in normalised space by a known amount on one axis
    z = np.zeros(DIM)
    z[0] = 0.5
    rows[1]["proposed"] = box.denorm(z)
    d1 = theta_distance_summary(rows, box)
    assert d1["per_state"]["s4"]["theta_l2_norm"] == pytest.approx(0.5, abs=1e-4)
    assert d1["mean_theta_l2_norm_by_split"]["development"] == 0.0


# ───────────────────────────── per-N record extraction ─────────────────────────────
def _fake_state(split, k6, dtz_end_mm=10.0, dwell=6, peak_qdot=1.0, peak_coin_speed=0.4):
    return {"split": split, "delivery_success": bool(k6), "dtz_end_mm": dtz_end_mm, "k6_max_dwell": dwell,
            "peak_qdot": peak_qdot, "peak_coin_speed": peak_coin_speed, "theta_exec": [0.1] * DIM}


def _fake_out(dev_k6, held_k6_states, *, budget0_total, oracle=4, passed=False, authorises=False):
    # held_k6_states: dict tag->k6 for s4,s7 ; dev is s1,s3 both = dev_k6==2 case. A K6=0 held state has sub-dwell (<6),
    # since delivery requires the full HELD_DWELL hold — dwell 6 would contradict K6=0.
    per8 = {"s1": _fake_state("development", 1), "s3": _fake_state("development", 1),
            "s4": _fake_state("held_out", held_k6_states["s4"], dtz_end_mm=33.0,
                              dwell=6 if held_k6_states["s4"] else 2),
            "s7": _fake_state("held_out", held_k6_states["s7"], dtz_end_mm=50.0,
                              dwell=6 if held_k6_states["s7"] else 2)}
    held8 = sum(held_k6_states.values())
    per0 = {t: _fake_state(s["split"], 0) for t, s in per8.items()}
    return {"deploy_budget": 8,
            "informed_sweep": {8: {"per_state": per8, "dev_k6": dev_k6, "held_out_k6": held8,
                                   "total_k6": dev_k6 + held8, "n_states": 4},
                               0: {"per_state": per0, "dev_k6": 0, "held_out_k6": 0, "total_k6": budget0_total, "n_states": 4}},
            "uninformed_sweep": {8: {"total_k6": 0}},
            "oracle_gate_diagnostic": {"total_k6": oracle},
            "gate": {"passed": passed, "actor_load_bearing": True},
            "authorises_rl": authorises, "verdict": "X", "diagnosed_blocker": None}


def test_coverage_record_has_all_required_fields_and_failure_mode():
    box = ThetaBox()
    out = _fake_out(2, {"s4": 0, "s7": 0}, budget0_total=1)
    dist = theta_distance_summary([{"tag": "s4", "split": "held_out", "proposed": box.denorm(np.zeros(DIM)),
                                    "teacher": box.denorm(np.zeros(DIM))}], box)
    rec = coverage_record(4, [14250, 14750, 19500, 17750], out, dist)
    for key in ("N", "train_cradles", "gate_informed_dev_k6", "gate_informed_held_out_k6", "gate_informed_total_k6",
                "proposal_only_total_k6", "search8_total_k6", "held_out_theta_l2_norm", "held_out_failure_mode",
                "motion_contract_ok_per_state", "gate_passed", "oracle_total_k6"):
        assert key in rec, key
    assert rec["train_cradles"] == [14250, 14750, 19500, 17750]
    assert rec["gate_informed_held_out_k6"] == 0 and rec["gate_informed_dev_k6"] == 2
    assert rec["proposal_only_total_k6"] == 1                # budget-0 (proposal-only) < budget-8
    assert rec["held_out_failure_mode"]["s4"] in ("NEVER_REACHED_ZONE", "REACHED_BUT_NO_SETTLE")  # dtz 33mm, no K6
    assert all(rec["motion_contract_ok_per_state"].values())  # within motion bounds


# ───────────────────────────── decision-tree verdict ─────────────────────────────
def _rec(n, dev, held, dist, passed):
    return {"N": n, "gate_informed_dev_k6": dev, "gate_informed_held_out_k6": held,
            "gate_informed_total_k6": dev + held, "held_out_theta_l2_norm": dist, "gate_passed": passed}


def test_verdict_authorises_when_a_gate_is_reached():
    recs = [_rec(2, 2, 0, 0.9, False), _rec(4, 2, 2, 0.3, True), _rec(6, 2, 2, 0.2, True)]
    v = coverage_verdict(recs)
    assert v["verdict"] == "COVERAGE_REACHES_UPDATE_ZERO_GATE"
    assert v["authorise_sac_td3"] is True and v["first_passing_N"] == 4 and v["stop"] is False


def test_verdict_insufficient_but_generalisation_improves():
    # no N passes, but held-out K6 rises 0→1→1 AND θ-distance shrinks monotonically → the positive mechanistic finding
    recs = [_rec(2, 2, 0, 0.90, False), _rec(4, 2, 1, 0.60, False), _rec(6, 2, 1, 0.40, False)]
    v = coverage_verdict(recs)
    assert v["verdict"] == "COVERAGE_ALONE_INSUFFICIENT"
    assert v["authorise_sac_td3"] is False and v["stop"] is True
    assert v["generalisation_improves_with_coverage"] is True
    assert "mechanistic_finding" in v


def test_verdict_flat_insufficient_no_mechanistic_finding():
    recs = [_rec(2, 2, 0, 0.90, False), _rec(4, 2, 0, 0.91, False), _rec(6, 2, 0, 0.92, False)]
    v = coverage_verdict(recs)
    assert v["verdict"] == "COVERAGE_ALONE_INSUFFICIENT"
    assert v["generalisation_improves_with_coverage"] is False
    assert "mechanistic_finding" not in v


# ───────────────────────────── reproduce_state dev-override (physics, slow) ─────────────────────────────
@pytest.mark.slow
def test_reproduce_state_dev_override_yields_a_dev_entry_for_a_new_seed():
    """The tag/split override lets a NEW seed be reproduced as a development cradle (canonical θ + basin augmentation) with
    the SAME frozen machinery — the coverage curve depends on this. Defaults (no override) still derive tag/split from idx."""
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness, reproduce_state
    e = reproduce_state(load_harness(), idx=16500, seed=16500, augment=True, tag="c1", split="development", basin_seed=16500)
    assert e["tag"] == "c1" and e["split"] == "development"
    assert "canonical_theta_vec" in e and len(e["canonical_theta_vec"]) == DIM
    assert e.get("k6_delivered") is True                     # seed 16500 is a certified K6-deliverable dev cradle
    assert "basin_candidates" in e and e.get("n_basin_delivering", 0) >= 1   # dev augmentation harvested
