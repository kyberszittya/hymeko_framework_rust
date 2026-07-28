"""R11.3 demonstration-bank pilot gates.

Fast (no rig): disjoint train/dev/test splits, the full failure taxonomy (every class + SUCCESS reachable via the
classifier), the record serialize/hash/replay round-trip, and the bank's success-denominator / rejection-panel separation.
Slow (physics): a canonical scenario runs end-to-end from a valid exact-zero certificate to a labelled outcome with a
complete measured energy ledger + mode trace (K6 linked to provenance on success), a re-run reproduces it
(replay_matches), and an invalid scenario is rejected before planning and kept out of the denominator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.coin_delivery.demo_bank import (
    DemonstrationBank,
    DemonstrationRecord,
    RolloutSignals,
    build_pilot_scenarios,
    classify,
    replay_matches,
    split_ids,
)
from hymeko_rl.coin_delivery.demo_bank.failure_class import ClassifyThresholds, FailureClass
from hymeko_rl.coin_delivery.demo_bank.scenario import CANONICAL_COIN, CoinTargetScenario, ScenarioSplit

THR = ClassifyThresholds()


# ── DISJOINT_TRAIN_DEV_TEST_SCENARIO_IDS ─────────────────────────────────────────────────────────────────────────────
def test_scenario_splits_disjoint_and_ids_unique():
    scenarios = build_pilot_scenarios()
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))                                         # unique
    by = split_ids(scenarios)
    train, dev, test, invalid = (by[ScenarioSplit.TRAIN], by[ScenarioSplit.DEV], by[ScenarioSplit.TEST],
                                 by[ScenarioSplit.INVALID])
    assert train and dev and test and invalid                               # every split populated
    assert train.isdisjoint(dev) and train.isdisjoint(test) and dev.isdisjoint(test)   # pairwise disjoint
    assert len([s for s in scenarios if s.split != ScenarioSplit.INVALID]) >= 10        # ~12 admissible


def test_relative_goal_vector():
    zone = np.array([0.10, 0.10])
    sc = CoinTargetScenario("t", CANONICAL_COIN, None, ScenarioSplit.TRAIN, "canonical")
    assert np.allclose(sc.relative_goal(zone), zone - CANONICAL_COIN)        # None target -> canonical zone
    sc2 = CoinTargetScenario("t2", CANONICAL_COIN, np.array([0.2, 0.2]), ScenarioSplit.TRAIN, "changed_target")
    assert np.allclose(sc2.relative_goal(zone), np.array([0.2, 0.2]) - CANONICAL_COIN)  # explicit target


# ── the full 14-class failure taxonomy (never one generic label) ─────────────────────────────────────────────────────
def _sig(**over: object) -> RolloutSignals:
    base = dict(goal_set_empty=False, reach_found=True, reach_qerr_rad=0.04, precontact_coin_motion_mm=0.0,
                premature_contacts=0, safe=True, handoff_admissible=True, k6=True, coin_progress_mm=60.0, min_dtz_mm=9.0)
    base.update(over)
    return RolloutSignals(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("sig,expected", [
    (_sig(goal_set_empty=True), FailureClass.GOAL_SET_EMPTY),
    (_sig(reach_found=False), FailureClass.RRT_PLANNING_FAILURE),
    (_sig(reach_qerr_rad=0.5), FailureClass.RRT_EXECUTION_FAILURE),
    (_sig(precontact_coin_motion_mm=2.0), FailureClass.PREMATURE_COIN_MOTION),
    (_sig(premature_contacts=1), FailureClass.PREMATURE_CONTACT),
    (_sig(safe=False), FailureClass.SAFETY_FAILURE),
    # non-K6, no coin progress: handoff-admissibility splits HANDOFF_INVALID vs CAPTURE_TEACHER_FAILURE
    (_sig(k6=False, coin_progress_mm=1.0, handoff_admissible=False), FailureClass.PRECONTACT_HANDOFF_INVALID),
    (_sig(k6=False, coin_progress_mm=1.0, handoff_admissible=True), FailureClass.CAPTURE_TEACHER_FAILURE),
    # engaged capture that did not arrive: delivery bands
    (_sig(k6=False, coin_progress_mm=30.0, min_dtz_mm=50.0), FailureClass.DELIVERY_TEACHER_FAILURE),
    (_sig(k6=False, coin_progress_mm=60.0, min_dtz_mm=10.0), FailureClass.TARGET_ENTRY_OVERSHOOT),
    (_sig(k6=False, coin_progress_mm=40.0, min_dtz_mm=25.0), FailureClass.SETTLE_K6_FAILURE),
])
def test_classify_each_failure_class(sig, expected):
    assert classify(sig, THR) is expected


def test_classify_success_before_contacts_and_handoff():
    # K6 with terminal release (no coin progress signal, inadmissible-looking handoff) is still SUCCESS
    assert classify(_sig(k6=True, coin_progress_mm=0.0, handoff_admissible=False), THR) is None


def test_classify_stage_order_reach_before_capture():
    # a failed reach outranks any downstream capture/delivery signal
    assert classify(_sig(reach_found=False, k6=False, coin_progress_mm=0.0), THR) is FailureClass.RRT_PLANNING_FAILURE


# ── record serialize / hash / replay round-trip ──────────────────────────────────────────────────────────────────────
def _rec(**over: object) -> DemonstrationRecord:
    base = dict(
        schema_version="r11.3-v2", scenario_id="canonical", split="train", kind="canonical", curriculum_stage="C0",
        seed=0, sample_id="canonical", ic_valid=True, ic_violations=(), admissible=True, rejection_reason="ADMISSIBLE",
        coin_pose=(0.076, 0.143), target_pose=None, relative_goal=(0.0, -0.1), zone_pose=(0.076, 0.043),
        target_relocation_honored=True, reach_found=True, goal_set_empty=False, n_goals=44, planning_time_s=0.5,
        reach={"reach_qerr": 0.01}, min_link_clr_mm=30.0, premature_contacts=0, handoff_complete=True,
        handoff_admissible=True, mode_trace=(0, 1, 2, 3, 4, 5, 6, 7), teacher_identity="CEM_phase_shape_capture",
        teacher_seed=0, teacher_params={"n": 3.0}, entry_obs={"entry_dtz_mm": 74.0}, contact_sequence=(0, 2),
        contacts=2, cos_dir=0.9, vel_scale=0.5, dtau=0.1, k6=True, entry_dtz_mm=74.0, min_dtz_mm=11.54,
        coin_progress_mm=62.46, safe=True, outcome_label="SUCCESS", energy_ledger={"robot_ke": 0.0},
        energy_verdicts=("ENERGY_LEDGER_COMPLETE",), energy_measurement_complete=True, git_sha="abc",
        provenance_hash="a" * 64, k6_provenance_link=True)
    base.update(over)
    return DemonstrationRecord(**base)  # type: ignore[arg-type]


def test_record_roundtrip_and_hash():
    r = _rec()
    assert DemonstrationRecord.from_dict(r.to_dict()) == r                   # exact JSON round-trip
    assert len(r.content_hash()) == 64
    assert r.content_hash() == _rec().content_hash()                        # deterministic
    assert _rec(seed=1).content_hash() != r.content_hash()                  # sensitive to recorded content
    assert replay_matches(r, _rec())                                        # identical re-run matches
    assert not replay_matches(r, _rec(k6=False, outcome_label="DELIVERY_FAILURE"))


def test_content_hash_ignores_wall_clock_planning_time():
    # planning_time_s is wall-clock: a deterministic re-run must hash identically despite a different timing
    assert _rec(planning_time_s=0.02).content_hash() == _rec(planning_time_s=9.9).content_hash()
    assert replay_matches(_rec(planning_time_s=0.02), _rec(planning_time_s=9.9))


# ── bank: success denominator vs rejection panel ─────────────────────────────────────────────────────────────────────
def test_bank_success_denominator_excludes_rejections(tmp_path: Path):
    bank = DemonstrationBank(tmp_path / "bank.jsonl")
    bank.append(_rec(scenario_id="a", k6=True, outcome_label="SUCCESS"))
    bank.append(_rec(scenario_id="b", k6=False, outcome_label="DELIVERY_FAILURE"))
    bank.append(_rec(scenario_id="inv", split="invalid", admissible=False, rejection_reason="start_in_collision",
                     outcome_label="INVALID_INITIAL_CONDITION"))
    assert len(bank.read()) == 3
    s = bank.summarize()
    assert s.admissible == 2 and s.successes == 1 and s.success_rate == pytest.approx(0.5)   # rejects excluded
    assert s.failures_by_class == {"DELIVERY_FAILURE": 1}
    assert s.rejected == 1 and s.rejection_reasons == {"start_in_collision": 1}


# ── deterministic IDs, curriculum mapping, taxonomy completeness, no teacher-free ────────────────────────────────────
def test_scenario_ids_and_poses_deterministic():
    a, b = build_pilot_scenarios(), build_pilot_scenarios()
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    assert all(np.array_equal(x.coin_xy, y.coin_xy) for x, y in zip(a, b))    # deterministic geometry, no RNG in IDs


def test_curriculum_stage_mapping():
    from hymeko_rl.coin_delivery.demo_bank.scenario import curriculum_stage
    assert curriculum_stage("canonical") == "C0"
    assert curriculum_stage("translated_together") == "C1" and curriculum_stage("shifted_coin") == "C2"
    assert curriculum_stage("changed_target") == "C3" and curriculum_stage("invalid_initial_condition") == "INVALID"
    # every pilot scenario maps to a known stage
    assert all(curriculum_stage(s.kind) != "C_UNKNOWN" for s in build_pilot_scenarios())


def test_bank_scenarios_64_unique_and_disjoint_splits():
    from hymeko_rl.coin_delivery.demo_bank.scenario import build_bank_scenarios
    bank = build_bank_scenarios()
    assert len(bank) == 64 and len({s.scenario_id for s in bank}) == 64          # 64 unique
    counts = {"C0": 0, "C1": 0, "C2": 0, "C3": 0}
    from hymeko_rl.coin_delivery.demo_bank.scenario import curriculum_stage
    for s in bank:
        counts[curriculum_stage(s.kind)] += 1
    assert counts == {"C0": 4, "C1": 16, "C2": 20, "C3": 24}                      # planned stage sizes
    by = split_ids(bank)
    tr, dv, te = by[ScenarioSplit.TRAIN], by[ScenarioSplit.DEV], by[ScenarioSplit.TEST]
    assert tr.isdisjoint(dv) and tr.isdisjoint(te) and dv.isdisjoint(te)          # disjoint by construction
    assert len(tr) + len(dv) + len(te) == 64 and len(tr) >= 40                    # ~70/15/15


def test_failure_taxonomy_is_fourteen_distinct_classes():
    values = [c.value for c in FailureClass]
    assert len(values) == 14 and len(set(values)) == 14                       # 14 distinct, no generic "FAILED"


def test_no_teacher_free_labels_anywhere():
    from hymeko_rl.coin_delivery.demo_bank import pipeline as P
    assert P.TEACHER_IDENTITY != "TEACHER_FREE" and P._NO_TEACHER != "TEACHER_FREE"
    assert "TEACHER_FREE" not in {c.value for c in FailureClass} and "TEACHER_FREE" != _rec().teacher_identity


# ── slow: physics-backed pipeline ────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rig():
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
    return _rig()


@pytest.mark.slow
def test_invalid_scenario_rejected_before_planning(rig):
    from hymeko_rl.coin_delivery.demo_bank import run_scenario
    from hymeko_rl.coin_delivery.demo_bank.scenario import _c

    sc = CoinTargetScenario("invalid_right_5cm", _c(0.05, 0.0), None, ScenarioSplit.INVALID, "invalid_initial_condition")
    rec = run_scenario(rig, sc)
    assert not rec.admissible and rec.outcome_label == "INVALID_INITIAL_CONDITION"   # rejected, not a success/failure
    assert not rec.reach_found and rec.teacher_seed == -1                            # never reached the teacher
    assert rec.ic_valid                                                              # the home itself is a valid zero home


@pytest.mark.slow
def test_canonical_scenario_end_to_end_and_replay(rig):
    from hymeko_rl.coin_delivery.demo_bank import run_scenario
    from hymeko_rl.coin_delivery.demo_bank.scenario import build_pilot_scenarios

    canon = next(s for s in build_pilot_scenarios() if s.scenario_id == "c0_canonical")
    rec = run_scenario(rig, canon)
    assert rec.ic_valid and rec.admissible                                   # valid exact-zero start
    assert rec.reach_found and rec.handoff_complete                          # deployed reach + complete handoff
    assert len(rec.mode_trace) >= 3 and rec.energy_measurement_complete      # complete trace + measured ledger
    if rec.is_success:
        assert rec.k6 and rec.k6_provenance_link                             # K6 linked to provenance hash
    rec2 = run_scenario(rig, canon)
    assert replay_matches(rec, rec2)                                         # deterministic reproduction
