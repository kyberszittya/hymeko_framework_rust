"""R11.4A target-conditioned delivery+settle teacher tests.

Fast (no rig): the Phase-A coordinate assembly (frozen early push, searched brake/release), the delivery-record round-trip
+ hash + provenance link, and the phase-energy ledger's derived quantities + completeness. Slow (physics): the two
target-conditioning contract gates (relocating the target changes direction_to_zone and the rollout's terminal goal; no
hidden canonical delivery orbit), the non-invasive energy-instrumentation gate (hook vs no-hook bit-exact), and one
settle-failure recovery.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.delivery_teacher import (
    DeliveryResult,
    DeliverySearchSpec,
    build_delivery_record,
)
from hymeko_rl.coin_delivery.delivery_teacher.phase_energy import (
    PhaseEnergyProbe,
    PhaseSnapshot,
    build_ledger,
)
from hymeko_rl.coin_delivery.delivery_teacher.record import DeliveryRecord


# ── Phase-A coordinate: frozen early push, searched brake/release ─────────────────────────────────────────────────────
def test_phase_a_assemble_freezes_push_and_places_searched():
    spec = DeliverySearchSpec()
    assert spec.search_idx == (3, 4, 5) and spec.frozen == {0: 0.12, 1: 0.15, 2: 0.0}
    theta = spec.assemble(np.array([6.0, 20.0, 1.2]))
    assert theta[0] == 0.12 and theta[1] == 0.15 and theta[2] == 0.0       # early push frozen
    assert theta[3] == 6.0 and theta[4] == 20.0 and theta[5] == 1.2        # brake onset / release / brake gain searched


def test_phase_a_assemble_clips_to_bounds():
    spec = DeliverySearchSpec()
    theta = spec.assemble(np.array([100.0, -5.0, 9.0]))                    # out of [lo,hi]
    assert theta[3] == spec.hi[0] and theta[4] == spec.lo[1] and theta[5] == spec.hi[2]


# ── phase-energy ledger derived quantities ───────────────────────────────────────────────────────────────────────────
def _probe_with_snaps() -> PhaseEnergyProbe:
    p = PhaseEnergyProbe(ramp=4, release=10, dt=0.01, w_pos=0.6, w_neg=0.2, peak_coin_ke=0.5, _entered=True,
                         _min_dtz_after_entry=5.0, _max_dtz_after_entry=18.0)
    p._snaps = {
        "after_capture": PhaseSnapshot(0, 0.10, 0.02, 76.0),
        "braking_onset": PhaseSnapshot(4, 0.30, 0.40, 40.0),
        "target_entry": PhaseSnapshot(7, 0.05, 0.30, 14.0),
        "release": PhaseSnapshot(10, 0.02, 0.10, 12.0),
        "settle_start": PhaseSnapshot(11, 0.01, 0.05, 12.0),
        "terminal": PhaseSnapshot(20, 0.0, 0.01, 11.0),
    }
    return p


def test_phase_energy_ledger_derived_and_complete():
    led = build_ledger(_probe_with_snaps())
    assert led.reached_target and led.t_coin_entry == pytest.approx(0.30)
    assert led.dH_capture_to_entry == pytest.approx((0.05 + 0.30) - (0.10 + 0.02))   # E_entry - E_after_capture
    assert led.dH_entry_to_settle == pytest.approx((0.01 + 0.05) - (0.05 + 0.30))
    assert led.target_directed_energy_ratio == pytest.approx(0.30 / 0.6)
    assert led.overshoot_mm == 18.0 and led.dwell_min_dtz_mm == 5.0
    assert led.is_complete()


def test_phase_energy_ledger_handles_no_target_entry():
    p = PhaseEnergyProbe(ramp=4, release=10, dt=0.01, w_pos=0.3, peak_coin_ke=0.2)
    p._snaps = {"after_capture": PhaseSnapshot(0, 0.1, 0.0, 60.0), "terminal": PhaseSnapshot(16, 0.0, 0.0, 55.0)}
    led = build_ledger(p)
    assert not led.reached_target and led.t_coin_entry is None
    assert led.dH_capture_to_entry is None and led.target_directed_energy_ratio is None
    assert led.is_complete()                                              # reached-phase completeness (no NaNs)


# ── delivery record round-trip + provenance ──────────────────────────────────────────────────────────────────────────
def _result(k6: bool = True, seed: int = 0) -> DeliveryResult:
    led = build_ledger(_probe_with_snaps())
    return DeliveryResult(seed=seed, theta=(0.12, 0.15, 0.0, 6.0, 20.0, 1.2), k6=k6, safe=True,
                          min_dtz_mm=9.0 if k6 else 33.0, measurements={"dtz_end": 0.009}, energy=led)


def _rec(k6: bool = True, seed: int = 0) -> DeliveryRecord:
    return build_delivery_record(
        scenario_id="c1_horiz", curriculum_stage="C1", split="train", phase="A_settle", coin_pose=(0.05, 0.14),
        target_pose=(0.0, 0.17), zone_pose=(0.0, 0.17), relative_goal=(-0.05, 0.03), entry_dtz_mm=76.0,
        original_failure_class="settle_k6_failure", baseline_k6=False, baseline_min_dtz_mm=32.8, search_dims=(3, 4, 5),
        result=_result(k6, seed), git_sha="abc123")


def test_delivery_record_roundtrip_hash_and_transition():
    r = _rec(k6=True)
    assert DeliveryRecord.from_dict(r.to_dict()) == r                     # exact JSON round-trip
    assert len(r.content_hash()) == 64 and r.content_hash() == _rec(k6=True).content_hash()
    assert _rec(k6=True, seed=1).content_hash() != r.content_hash()       # sensitive
    assert r.recovered and r.outcome_transition == "settle_k6_failure->SUCCESS" and r.is_success
    assert len(r.provenance_hash) == 64


def test_delivery_record_non_recovery_transition():
    r = _rec(k6=False)
    assert not r.recovered and r.outcome_transition == "settle_k6_failure->settle_k6_failure"
    assert "ENERGY_LEDGER_COMPLETE" in r.energy_verdicts and r.energy_complete


def test_teacher_never_labelled_free():
    assert _rec().teacher_identity == "rollout_primitive_delivery_settle_CEM"
    assert "FREE" not in _rec().teacher_identity.upper().replace("REBOUND", "")


# ── R11.4A0 corrected delivery contract + taxonomy ───────────────────────────────────────────────────────────────────
def test_delivery_ready_grasp_certificate_valid_and_invalid():
    from hymeko_rl.coin_delivery.delivery_teacher.delivery_contract import DeliveryReadyGraspCertificate

    ok = DeliveryReadyGraspCertificate(bilateral_contacts=True, bilateral_dwell=4, relative_slip=0.01,
                                       coin_speed=0.05, safe=True, continuous_episode=True)
    assert ok.valid and ok.violations() == ()
    bad = DeliveryReadyGraspCertificate(bilateral_contacts=False, bilateral_dwell=1, relative_slip=0.5,
                                        coin_speed=0.9, safe=True, continuous_episode=True)
    assert not bad.valid
    assert set(bad.violations()) == {"bilateral_contacts", "bilateral_dwell", "relative_slip", "coin_speed"}


@pytest.mark.parametrize("grasp,kin,k6,dtz,expected", [
    (True, True, True, 8.0, "k6_with_valid_delivery_mode"),
    (False, False, True, 19.0, "k6_without_delivery_mode_transition"),   # nudge — NOT a delivery success
    (False, False, False, 33.0, "capture_to_delivery_regrasp_failure"),
    (True, True, False, 46.0, "delivery_failure_after_valid_grasp"),
    (True, True, False, 30.0, "settle_failure_after_valid_grasp"),
])
def test_classify_delivery_corrected(grasp, kin, k6, dtz, expected):
    from hymeko_rl.coin_delivery.delivery_teacher.delivery_contract import classify_delivery
    assert classify_delivery(grasp_valid=grasp, reaches_kinetic=kin, k6=k6, min_dtz_mm=dtz).value == expected


def test_reclassify_from_bank_fields_proxy():
    from hymeko_rl.coin_delivery.delivery_teacher.delivery_contract import reclassify_from_bank_fields
    # contacts==2 (grasped) + K6 -> valid delivery mode; contacts==0 + K6 -> nudge (NOT delivery success)
    assert reclassify_from_bank_fields(contacts=2, k6=True, min_dtz_mm=8.0).value == "k6_with_valid_delivery_mode"
    assert reclassify_from_bank_fields(contacts=0, k6=True, min_dtz_mm=19.0).value == "k6_without_delivery_mode_transition"
    assert reclassify_from_bank_fields(contacts=0, k6=False, min_dtz_mm=None).value == "capture_to_delivery_regrasp_failure"
    assert reclassify_from_bank_fields(contacts=2, k6=False, min_dtz_mm=46.0).value == "delivery_failure_after_valid_grasp"


def test_no_nudge_mislabelled_as_delivery_success():
    from hymeko_rl.coin_delivery.delivery_teacher.delivery_contract import DeliveryOutcomeClass, classify_delivery
    nudge = classify_delivery(grasp_valid=False, reaches_kinetic=False, k6=True, min_dtz_mm=15.0)
    assert nudge is DeliveryOutcomeClass.K6_WITHOUT_DELIVERY_MODE_TRANSITION       # a K6 nudge is never K6_WITH_VALID


# ── slow: physics contract gates + mechanism demonstration ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def capture_snapshot():
    """A real post-capture handoff for a settle-failure scenario (reach + capture through the R11.3 pipeline)."""
    import dataclasses

    from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_pilot_scenarios
    from hymeko_rl.coin_delivery.demo_bank import pipeline as P
    from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
    from hymeko_rl.experiments import coin_zero_home_reach as Z
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

    rig = _rig()
    scen = next(s for s in build_pilot_scenarios() if s.scenario_id == "c1_horiz")
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    home, coin = Z._home_with_coin(rig, scen.coin_xy)
    _reason, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg, PipelineConfig(), 0)
    assert rc is not None
    return rc.result.outcome.snapshot, rc.zone


@pytest.mark.slow
def test_energy_instrumentation_noninvasive(capture_snapshot):
    from hymeko_rl.coin_delivery.delivery_teacher.solver import _config
    from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive

    snap, _zone = capture_snapshot
    theta = np.array([0.12, 0.15, 0.0, 6.0, 20.0, 1.2])
    cfg = _config(DeliverySearchSpec())
    m_no = rollout_primitive(snap, theta, cfg)
    probe = PhaseEnergyProbe(ramp=6, release=20, dt=float(snap.stack.control_dt))
    m_hook = rollout_primitive(snap, theta, cfg, frame_hook=probe)
    assert np.array_equal(np.array(m_no["coin_trace"]), np.array(m_hook["coin_trace"]))   # hook is non-behavioural


@pytest.mark.slow
def test_target_relocation_propagates_through_delivery(capture_snapshot):
    snap, zone = capture_snapshot
    rl = snap.branch()
    _u0, dtz0 = rl.inner.direction_to_zone()
    rl.inner._zone_x, rl.inner._zone_y = float(zone[0]) + 0.05, float(zone[1])   # relocate the target
    _u1, dtz1 = rl.inner.direction_to_zone()
    assert abs(dtz1 - dtz0) > 1e-4                                         # relocation changes the measured goal distance


@pytest.mark.slow
def test_grasp_split_mechanism_seed0_vs_seed1():
    """The R11.4A0 mechanism: a GRASPED capture (bank_c0_3 seed 1) enters KINETIC + delivers tight K6; a RELEASED capture
    (seed 0) never enters the delivery mode. This is the known split CONTACT_ACQUIRE_AND_HOLD (Boundary 2) must close."""
    import dataclasses

    from hymeko_rl.coin_delivery.delivery_teacher.regrasp_characterize import characterize_delivery
    from hymeko_rl.coin_delivery.demo_bank import PipelineConfig, build_bank_scenarios
    from hymeko_rl.coin_delivery.demo_bank import pipeline as P
    from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
    from hymeko_rl.experiments import coin_zero_home_reach as Z
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig

    rig = _rig()
    scen = next(s for s in build_bank_scenarios() if s.scenario_id == "bank_c0_3")
    cfg = dataclasses.replace(pga.TransitConfig(), substeps=6, hold_steps=160)
    home, coin = Z._home_with_coin(rig, scen.coin_xy)
    out = {}
    for seed in (0, 1):
        _r, rc = P._do_reach_and_capture(rig, scen, coin, home, cfg, PipelineConfig(teacher_budget=1), seed)
        out[seed] = (rc.result.outcome.contacts, characterize_delivery(rc.result.outcome.snapshot, rig["down"]))
    (c0, m0), (c1, m1) = out[0], out[1]
    assert c1 == 2 and m1.reaches_kinetic and m1.k6                        # grasped seed -> KINETIC -> R2 -> K6
    assert c0 < 2 and not m0.reaches_kinetic                              # released seed never enters the delivery mode
