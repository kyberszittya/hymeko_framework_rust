"""R11.2 gate tests — the HyMeKo hybrid-delivery IR + coin adapter.

Fast (pure IR, numpy only): the ten R11.2 gates over synthetic states — the initial-condition certificate (pass + one
failure per clause), the certificate-filtered distribution + rejection accounting, mode-trace validation + the M0
invariant, transition guards + handoff completeness, the two-level (measured, not conserved) energy contract, and the
provenance hash + success-certificate link. Slow (physics): the same certificates against a real fresh zero home and an
instrumented RRT reach, proving the K6 success certificate is bound to the rollout's provenance hash.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hymeko_rl.ir import (
    AdmissibilityResult,
    EnergyTransitionCertificate,
    HandoffDescriptor,
    HybridMode,
    HybridTransition,
    InitialCondition,
    InitialConditionViolation,
    InitialDistribution,
    MeasuredEnergyLedger,
    ModeTrace,
    RejectionLedger,
    RolloutProvenance,
    RolloutState,
    SuccessCertificate,
    TransitionGuard,
    TransitionGuardError,
    build_mode_trace,
    zero_home_invariant,
)

ZERO_HOME = InitialCondition(name="EXACT_ZERO_HOME_V1", q_expected=np.zeros(4))


def _fresh_state(**over: object) -> RolloutState:
    base = dict(q=np.zeros(4), qdot=np.zeros(4), prev_tau=np.zeros(4), object_vel=np.zeros(2),
                n_task_contacts=0, controller_memory_empty=True, step=0, has_snapshot_parent=False,
                has_teacher_state=False)
    base.update(over)
    return RolloutState(**base)  # type: ignore[arg-type]


# ── Gate 1: EXACT_ZERO_INITIAL_CONDITION_CERTIFICATE_PASS ─────────────────────────────────────────────────────────────
def test_exact_zero_initial_condition_certificate_pass():
    cert = ZERO_HOME.certify(RolloutState.at_rest(np.zeros(4)))
    assert cert.valid and cert.violations == () and all(cert.checks.values())
    assert cert.raise_if_invalid() is cert                                   # valid -> returns self, no raise


def test_rollout_state_post_init_rejects_ragged_arrays():
    with pytest.raises(AssertionError):
        RolloutState(q=np.zeros(4), qdot=np.zeros(3), prev_tau=np.zeros(4), object_vel=np.zeros(2),
                     n_task_contacts=0, controller_memory_empty=True, step=0, has_snapshot_parent=False,
                     has_teacher_state=False)


# ── Gate 3: ROLLOUT_STATE_CONTINUITY_PASS (nonzero rate / advanced step / moving object are rejected) ─────────────────
@pytest.mark.parametrize("field,val,clause", [
    ("qdot", np.array([0.0, 0.0, 1e-3, 0.0]), "qdot_zero"),
    ("object_vel", np.array([1e-2, 0.0]), "object_at_rest"),
    ("prev_tau", np.array([0.0, 0.5, 0.0, 0.0]), "prev_tau_zero"),
    ("step", 1, "step_zero"),
    ("n_task_contacts", 2, "empty_contacts"),
    ("controller_memory_empty", False, "empty_memory"),
])
def test_rollout_state_continuity_each_clause_fails(field, val, clause):
    cert = ZERO_HOME.certify(_fresh_state(**{field: val}))
    assert not cert.valid and clause in cert.violations and not cert.checks[clause]
    with pytest.raises(InitialConditionViolation):
        cert.raise_if_invalid()


# ── Gate 4: NO_SNAPSHOT_INJECTION_PASS ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("field,clause", [("has_snapshot_parent", "no_snapshot"), ("has_teacher_state", "no_teacher")])
def test_no_snapshot_or_teacher_injection(field, clause):
    cert = ZERO_HOME.certify(_fresh_state(**{field: True}))
    assert not cert.valid and clause in cert.violations


# ── Gate 2: INITIAL_DISTRIBUTION_ADMISSIBILITY_PASS ──────────────────────────────────────────────────────────────────
def test_initial_distribution_admissibility_and_rejection_accounting():
    canon = np.array([0.076, 0.143])

    def predicate(pose: np.ndarray) -> AdmissibilityResult:
        return (AdmissibilityResult(True, "ADMISSIBLE") if pose[0] <= 0.10
                else AdmissibilityResult(False, "start_in_collision"))

    dist = InitialDistribution(name="D_coin", lo=canon + [-0.05, -0.04], hi=canon + [0.03, 0.03], predicate=predicate)
    assert dist.admits(canon).admissible                                     # canonical admissible
    assert dist.admits(canon + [0.10, 0.0]).reason == "out_of_bounds"        # outside the box, predicate not consulted
    right5 = np.array([0.105, 0.143])                                        # in-box (hi_x=0.106) but predicate rejects
    assert not dist.admits(right5).admissible and dist.admits(right5).reason == "start_in_collision"

    ledger = RejectionLedger()
    for pose in (canon, canon, right5):
        ledger.record(dist.admits(pose))
    assert ledger.admitted == 2 and ledger.rejected == {"start_in_collision": 1}
    assert ledger.total == 3 and ledger.rejection_rate == pytest.approx(1 / 3)   # rejects out of the denominator
    assert RejectionLedger().rejection_rate == 0.0                            # empty ledger


# ── Gate 5: HYBRID_MODE_TRACE_VALID_PASS ─────────────────────────────────────────────────────────────────────────────
def test_hybrid_mode_trace_valid_and_invalid():
    assert ModeTrace.canonical().is_valid() and ModeTrace.canonical().first_invalid_step() is None
    assert build_mode_trace([HybridMode.ZERO_HOME, HybridMode.FREE_REACH, HybridMode.FREE_REACH]).is_valid()  # repeat ok
    skip = build_mode_trace([HybridMode.ZERO_HOME, HybridMode.CAPTURE])                   # M0 -> M3 skip
    assert not skip.is_valid() and skip.first_invalid_step() == 1
    regress = build_mode_trace([HybridMode.ZERO_HOME, HybridMode.FREE_REACH, HybridMode.ZERO_HOME])
    assert not regress.is_valid()
    assert not build_mode_trace([HybridMode.FREE_REACH]).is_valid()                        # must start at M0


def test_zero_home_invariant_holds_and_fails():
    inv = zero_home_invariant()
    assert inv.mode == HybridMode.ZERO_HOME
    assert inv.holds(RolloutState.at_rest(np.zeros(4)))
    assert not inv.holds(_fresh_state(qdot=np.array([0.0, 0.1, 0.0, 0.0])))
    assert not inv.holds(_fresh_state(q=np.array([0.0, 0.0, 0.0, 0.2])))


# ── Gate 6: HYBRID_TRANSITION_GUARDS_PASS ────────────────────────────────────────────────────────────────────────────
def test_hybrid_transition_guard_fires_and_blocks():
    guard = TransitionGuard("reach_clear", HybridMode.FREE_REACH, HybridMode.PRECONTACT_ALIGNMENT,
                            "no task contact", lambda s: s.n_task_contacts == 0)
    tr = HybridTransition(HybridMode.FREE_REACH, HybridMode.PRECONTACT_ALIGNMENT, guard,
                          reset_map=lambda s: RolloutState.at_rest(s.q))
    ok = _fresh_state(qdot=np.ones(4))
    assert guard.allows(ok)
    assert np.array_equal(tr.fire(ok).qdot, np.zeros(4))                      # reset map applied on fire
    with pytest.raises(TransitionGuardError):
        tr.fire(_fresh_state(n_task_contacts=1))                             # guard blocks -> raises


def test_transition_guard_requires_consecutive_modes():
    with pytest.raises(AssertionError):
        TransitionGuard("bad", HybridMode.ZERO_HOME, HybridMode.CAPTURE, "skip", lambda s: True)


# ── Gate 7: HANDOFF_DESCRIPTOR_COMPLETE_PASS ─────────────────────────────────────────────────────────────────────────
def _handoff(**over: object) -> HandoffDescriptor:
    base = dict(q=np.zeros(4), qdot=np.zeros(4), prev_tau=np.zeros(4), x_coin=np.array([0.076, 0.143]),
                xdot_coin=np.zeros(2), x_goal=np.array([0.2, 0.2]), n_contacts=2, e_phase=0.5,
                mode_from=HybridMode.CAPTURE, mode_to=HybridMode.CONTROLLED_DELIVERY)
    base.update(over)
    return HandoffDescriptor(**base)  # type: ignore[arg-type]


def test_handoff_descriptor_complete_and_incomplete():
    assert _handoff().is_complete()
    assert not _handoff(x_goal=np.array([np.nan, 0.2])).is_complete()        # non-finite field
    assert not _handoff(q=np.array([])).is_complete()                        # empty field
    assert not _handoff(mode_to=HybridMode.TARGET_ENTRY).is_complete()       # non-consecutive modes


# ── Gates 8 & 9: MEASURED_ENERGY_LEDGER_COMPLETE_PASS / ENERGY_BALANCE_RESIDUAL_RECORDED_PASS ────────────────────────
def _ledger(**over: object) -> MeasuredEnergyLedger:
    base = dict(robot_ke=0.4, object_ke=0.01, potential_energy=0.0, w_actuator_pos=0.6, w_actuator_neg=0.2,
                contact_impulse=0.0, dissipation_proxy=0.2, energy_pre=0.0, energy_post=0.41, numerical_residual=0.0)
    base.update(over)
    return MeasuredEnergyLedger(**base)  # type: ignore[arg-type]


def test_measured_energy_ledger_completeness_and_residual():
    led = _ledger()
    cert = EnergyTransitionCertificate(led)
    assert led.is_complete() and cert.is_measurement_complete()
    assert "ENERGY_LEDGER_COMPLETE" in cert.verdicts and "ENERGY_BALANCE_RESIDUAL_RECORDED" in cert.verdicts
    assert math.isfinite(led.balance_residual())
    assert led.balance_residual() == pytest.approx(0.41 - (0.0 + 0.6 - 0.2 - 0.2 - 0.0))
    assert not EnergyTransitionCertificate(_ledger(robot_ke=float("nan"))).is_measurement_complete()  # NaN -> incomplete


def test_energy_certificate_refuses_conservation_verdict():
    with pytest.raises(NotImplementedError):
        EnergyTransitionCertificate(_ledger()).conservation_verdict()        # two-level contract: R11.8, not R11.2


# ── Gate 10: K6_CERTIFICATE_PROVENANCE_LINK_PASS ─────────────────────────────────────────────────────────────────────
def _prov(seed: int = 0, coin=(0.076, 0.143)) -> RolloutProvenance:
    return RolloutProvenance(git_sha="abc123", seed=seed, ic_certificate=ZERO_HOME.certify(RolloutState.at_rest(np.zeros(4))),
                             coin_pose=np.asarray(coin, float), target_pose=None, mode_trace=ModeTrace.canonical(),
                             n_transitions=7, energy_ledger_complete=True)


def test_provenance_hash_deterministic_and_sensitive():
    assert _prov().content_hash() == _prov().content_hash()                  # deterministic
    assert len(_prov().content_hash()) == 64
    assert _prov(seed=1).content_hash() != _prov(seed=0).content_hash()      # sensitive to a recorded field
    assert _prov(coin=(0.10, 0.143)).content_hash() != _prov().content_hash()


def test_success_certificate_links_to_provenance_hash():
    h = _prov().content_hash()
    sc = SuccessCertificate("STRICT_K6", True, 11.54, True, h)
    assert sc.provenance_hash == h and sc.success and sc.metric_mm == pytest.approx(11.54)
    with pytest.raises(AssertionError):
        SuccessCertificate("STRICT_K6", True, 11.54, True, "tooshort")       # must be a 64-char digest


# ── adapter helpers exercisable without the rig ──────────────────────────────────────────────────────────────────────
def test_adapter_pure_helpers():
    from hymeko_rl.coin_delivery import ir_adapter as A

    assert np.array_equal(A.EXACT_ZERO_HOME_V1.q_expected, np.zeros(4))
    assert A.zero_home_reach_trace(captured=False, k6=False).modes == (
        HybridMode.ZERO_HOME, HybridMode.FREE_REACH, HybridMode.PRECONTACT_ALIGNMENT)
    assert A.zero_home_reach_trace(captured=True, k6=True).is_valid()
    assert A.zero_home_reach_trace(captured=True, k6=True).modes[-1] == HybridMode.K6_SUCCESS
    sc = A.k6_success_certificate(type("O", (), {"k6": True, "min_dtz_mm": 9.0, "safe": True})(), "f" * 64)
    assert sc.success and sc.provenance_hash == "f" * 64
    assert isinstance(A._git_sha(), str) and len(A._git_sha()) > 0

    class _D:
        def __init__(self) -> None:
            self.ctrl = np.array([1.0, 0.0, 0.0, 0.0])
            self.qvel = np.array([2.0, 0.0, 0.0, 0.0])
            self.time = 0.0

    class _Rl:
        def __init__(self, d: object) -> None:
            self.inner = type("I", (), {"data": d})()

    d = _D()
    rl = _Rl(d)
    probe = A.EnergyProbe()
    d.time = 0.1
    probe(rl, 0)                                                              # power=+2, dt=0.1 -> +0.2 positive work
    d.ctrl = np.array([-1.0, 0.0, 0.0, 0.0])
    d.time = 0.2
    probe(rl, 1)                                                              # power=-2, dt=0.1 -> +0.2 negative work
    assert probe.w_pos == pytest.approx(0.2) and probe.w_neg == pytest.approx(0.2)


# ── slow: physics-backed certificates against a real rig ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rig():
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
    return _rig()


@pytest.mark.slow
def test_certify_zero_home_and_admissibility_against_rig(rig):
    from hymeko_rl.coin_delivery import ir_adapter as A

    canon = np.array([0.07578387, 0.14279214])
    assert A.certify_zero_home(rig).valid                                     # a real fresh zero home certifies
    assert A.coin_admissibility(rig, canon).admissible                       # canonical coin admissible
    right5 = A.coin_admissibility(rig, canon + [0.05, 0.0])                   # zero home collides with the coin
    assert not right5.admissible and right5.reason == "start_in_collision"
    dist = A.make_coin_distribution(rig, canon + [-0.05, -0.04], canon + [0.03, 0.03])
    assert dist.admits(canon).admissible


@pytest.mark.slow
def test_instrument_reach_rrt_links_k6_certificate_to_provenance(rig):
    from hymeko_rl.coin_delivery import ir_adapter as A

    out = A.instrument_reach_rrt(rig, coin_xy=None, seed=0, git_sha="testsha")
    assert out is not None
    assert out["ic_certificate"].valid                                       # zero-home IC certificate holds
    assert out["energy_certificate"].is_measurement_complete()               # measured ledger complete + residual
    assert out["mode_trace"].is_valid()
    assert out["k6_certificate"].provenance_hash == out["provenance"].content_hash()   # THE link
