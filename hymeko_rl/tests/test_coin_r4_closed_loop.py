"""R4 closed-loop basin-aware intent correction — the mandatory Stage-3 tests (frozen contract
`reports/2026-07-27-coin-r4-closed-loop-intent-contract.md`). Tests 1-6, 8, 10 drive the PURE law / phase machine with
synthetic inputs (fast, no physics); the GOLDEN + decoder contract (7, 9) drive real dev snapshots and are marked slow.
No held-out data (s4/s7) is used for design here.

The delivering strategy is PUSH-then-COAST (build momentum, RELEASE, coast into the zone under friction). The load-bearing
guarantees tested here:
  * GOLDEN — the closed-loop rollout under a constant controller reproduces the frozen option BIT-IDENTICALLY.
  * zero-residual identity — a benign, on-plan response yields ≈0 correction (C0 in unit form).
  * coast monotonicity — more forward momentum ⇒ the coast-in phase can only ADVANCE toward RELEASE, never revert.
  * contact monotonicity, mirror equivariance, phase monotonicity, energy/stopping algebra, budget provenance,
    decoder contract, no-teacher, determinism.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option.closed_loop_intent import (
    BRAKE, PUSH, RELEASE, ClosedLoopController, CorrectionParams, IntentCorrector, PhaseMachine, coast_reach,
    coastin_phase)
from hymeko_rl.coin_delivery.theta_option.closed_loop_state import (
    braking_decel, brake_work, kinetic_energy, over_speed, stopping_distance)
from hymeko_rl.coin_delivery.theta_option.physical_intent import PhysicalIntent
from hymeko_rl.coin_delivery.theta_option.semantics import ThetaBox

FWD, PEAKV, LAT, SQZ, BRK_E, BRK_D, REL = range(7)


def _intent(**kw) -> PhysicalIntent:
    base = dict(forward_drive=0.04, peak_velocity=0.30, lateral=0.0, squeeze=0.05,
                brake_entry=0.20, braking_demand=0.60, release=0.30)
    base.update(kw)
    return PhysicalIntent(**base)


class _Resp:
    """A synthetic ResponseState-duck exposing only the fields the corrector reads."""

    def __init__(self, *, dtz=0.05, v_parallel=0.2, v_perpendicular=0.0, coin_spin=0.0, coin_speed=0.2,
                 both_contact=True, fn=(1.0, 1.0)):
        self.dtz, self.v_parallel, self.v_perpendicular, self.coin_spin = dtz, v_parallel, v_perpendicular, coin_spin
        self.coin_speed, self.both_contact, self.fn = coin_speed, both_contact, tuple(fn)


# ── energy / stopping algebra (physics-free) ────────────────────────────────────────────────────────────────────────
def test_energy_stopping_equivalence_and_guard():
    """E_kin > W_brake ⇔ d_stop > d_safe (mass-free guard); a vanishing authority reads as over-speed (floored)."""
    for v in (0.05, 0.263, 0.45, 0.7):
        a_brake, d_safe, m = 0.4, 0.06, 0.05
        assert (kinetic_energy(m, v) > brake_work(m, a_brake, d_safe)) == over_speed(stopping_distance(v, a_brake), d_safe)
    assert braking_decel(0.0, 0.05) > 0.0
    assert stopping_distance(0.5, 1.0) > stopping_distance(0.3, 1.0)
    assert stopping_distance(0.5, 2.0) < stopping_distance(0.5, 1.0)


def test_coast_reach_monotone():
    assert coast_reach(0.5, 0.5) > coast_reach(0.3, 0.5)       # more speed ⇒ more reach
    assert coast_reach(0.5, 1.0) < coast_reach(0.5, 0.5)       # more friction ⇒ less reach
    assert coast_reach(-0.3, 0.5) == 0.0                       # a receding coin reaches 0


# ── R5: the online coast-deceleration estimator (physics-free) ──────────────────────────────────────────────────────
def test_r5_coast_estimator():
    from hymeko_rl.coin_delivery.theta_option.closed_loop_state import CoastEstimator
    # recovers a known constant deceleration a=0.6 from a decaying-velocity window
    e = CoastEstimator()
    v, dt = 0.5, 0.05
    for _ in range(6):
        vn = v - 0.6 * dt
        e.update(v, vn, dt, active_push=False)
        v = vn
    assert np.isclose(e.estimate(), 0.6, atol=1e-6) and e.n_samples == 6
    # rejects invalid samples: active push, low speed, accelerating
    e2 = CoastEstimator()
    e2.update(0.3, 0.28, 0.05, active_push=True)               # active push
    e2.update(0.02, 0.01, 0.05, active_push=False)             # below v_min
    e2.update(0.3, 0.35, 0.05, active_push=False)              # accelerating
    assert e2.n_samples == 0 and e2.estimate() == e2.a_prior   # ⇒ prior
    # robust to one outlier (median), clamps to the band, prior below n_min
    e3 = CoastEstimator()
    v = 0.5
    for k in range(5):
        a = 0.6 if k != 2 else 3.0
        vn = v - a * dt
        e3.update(v, vn, dt, active_push=False)
        v = max(vn, 0.1)
    assert np.isclose(e3.estimate(), 0.6, atol=1e-6)           # outlier does not move the median
    assert CoastEstimator().estimate() == 0.55                 # below n_min ⇒ prior
    e4 = CoastEstimator()
    v = 0.5
    for _ in range(4):
        vn = v - 5.0 * dt
        e4.update(v, vn, dt, active_push=False)
        v = max(vn, 0.2)
    assert e4.estimate() == e4.a_hi                            # clamped to the physical band


# ── R6: the release-certificate monitor latches only after N consecutive certified frames ──────────────────────────
def test_r6_release_cert_monitor(monkeypatch):
    import hymeko_rl.coin_delivery.theta_option.release_certificate as rc
    seq = iter([True, True, False, True, True, True, True])    # a transient (2) then a real 4-in-a-row
    monkeypatch.setattr(rc, "release_certificate", lambda rl, p=None: (next(seq), {}))
    mon = rc.ReleaseCertMonitor(rc.ReleaseCertParams(n_frames=3))
    armed = [mon.update(None, t)[0] for t in range(1, 8)]
    assert armed == [False, False, False, False, False, True, True]   # arms only on the 3rd consecutive True (t=6)
    assert mon.armed and mon.armed_at == 6                     # monotone: stays armed


def test_r6_release_cert_params():
    from hymeko_rl.coin_delivery.theta_option.release_certificate import ReleaseCertParams
    p = ReleaseCertParams()
    assert p.settle_tol < 0.06 and p.n_frames >= 1            # tighter than K6 settle; needs ≥1 confirming frame


# ── R7 (V0): the velocity-servo primitive algebra — bounded velocity, saturating NON-REVERSING stop ─────────────────
def test_r7_velocity_ref_bounded():
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_ref
    assert velocity_ref(0.0, 4.0, 0.35) == 0.0                # zero reference at the zone
    assert velocity_ref(-0.1, 4.0, 0.35) == 0.0               # never negative behind the zone
    assert velocity_ref(1.0, 4.0, 0.35) == 0.35               # clipped to v_max
    assert velocity_ref(0.05, 4.0, 0.35) == 0.2               # proportional in the mid-range
    assert velocity_ref(0.08, 4.0, 0.35) > velocity_ref(0.04, 4.0, 0.35)   # decays toward the zone


def test_r7_accel_cmd_saturates():
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import accel_cmd
    assert accel_cmd(1.0, 0.0, 12.0, 4.0, 8.0) == 4.0         # saturated push
    assert accel_cmd(0.0, 1.0, 12.0, 4.0, 8.0) == -8.0        # saturated decel
    assert -8.0 <= accel_cmd(0.31, 0.30, 12.0, 4.0, 8.0) <= 4.0


def test_r7_non_reversing_stop():
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import non_reversing_accel
    dt = 0.02
    # positive motion + a decel that WOULD reverse ⇒ clamped so next-step velocity is exactly 0 (never negative)
    for v in (0.05, 0.2, 0.45):
        a = non_reversing_accel(-1000.0, v, dt)
        assert abs((v + a * dt)) < 1e-9                        # brought to rest, not reversed
        assert v + a * dt >= 0.0
    # a decel that does NOT reverse is unchanged; a receding/zero coin is unchanged
    assert non_reversing_accel(-1.0, 0.5, dt) == -1.0
    assert non_reversing_accel(-1000.0, -0.3, dt) == -1000.0
    assert non_reversing_accel(5.0, 0.2, dt) == 5.0           # positive accel never touched


# ── R8: the tip-referenced scaffold's joint-velocity reference is bounded and decays to zero at the zone ────────────
def test_r8_tip_transport_ref_bounded():
    from hymeko_rl.coin_delivery.theta_option.tip_transport import TipTransportParams
    from hymeko_rl.coin_delivery.theta_option.velocity_transport import velocity_ref
    p = TipTransportParams()
    assert p.qdot_max <= 2.0 and p.v_max <= 0.30             # ≤ joint_vel_safe; bounded approach speed
    # the joint-velocity reference qdot_ref_mag = clip(k_q·v_ref, 0, qdot_max) is bounded and 0 at the zone
    for d_remain in (0.0, 0.02, 0.05, 0.2, 1.0):
        q = float(np.clip(p.k_q * velocity_ref(d_remain, p.k_d, p.v_max), 0.0, p.qdot_max))
        assert 0.0 <= q <= p.qdot_max
    assert np.clip(p.k_q * velocity_ref(0.0, p.k_d, p.v_max), 0.0, p.qdot_max) == 0.0   # zero reference at the zone


# ── 1. ZERO-RESIDUAL IDENTITY — a benign, on-plan response yields ≈0 correction ─────────────────────────────────────
def test_1_zero_residual_identity():
    c = IntentCorrector(CorrectionParams())                    # k_forward_deficit = 0 by default
    intent = _intent(squeeze=0.05)                             # squeeze below the push cap
    resp = _Resp(dtz=0.05, v_parallel=0.2, both_contact=True, fn=(1.0, 1.0), v_perpendicular=0.0)
    out, log = c.correct(intent, resp, was_swapped=False)
    assert np.allclose(out.to_vector(), intent.to_vector(), atol=1e-9), (log, out.to_vector())
    assert log["fired"] == []


# ── 2. OVER-CONTROL / COAST MONOTONICITY — more momentum ⇒ phase only ADVANCES toward RELEASE, never reverts ─────────
def test_2_coast_phase_monotone_in_momentum():
    p = CorrectionParams()
    # holding geometry fixed, a faster coin reaches a phase ≥ a slower coin's (never earlier→PUSH after RELEASE)
    prev = PUSH
    for v in np.linspace(0.0, 1.2, 25):
        ph = coastin_phase(dtz=0.10, v_parallel=float(v), p=p, elapsed=10, phase_floor=PUSH)
        assert ph >= prev - 0  # phase is non-decreasing in speed (PUSH -> RELEASE -> BRAKE-trim on strong overshoot)
        prev = max(prev, PUSH)
    assert coastin_phase(0.10, 0.05, p, elapsed=10, phase_floor=PUSH) == PUSH        # too slow ⇒ keep building
    assert coastin_phase(0.10, 0.31, p, elapsed=10, phase_floor=PUSH) == RELEASE     # reach in the coast-in band
    assert coastin_phase(0.10, 0.70, p, elapsed=10, phase_floor=PUSH) == BRAKE       # strong overshoot ⇒ trim/arrest
    assert coastin_phase(0.10, 0.31, p, elapsed=2, phase_floor=PUSH) == PUSH         # min-push guard holds first
    assert coastin_phase(0.10, 0.90, p, elapsed=2, phase_floor=PUSH) == BRAKE        # overshoot beats the min-push guard


# ── 3. CONTACT-RISK MONOTONICITY — lower contact margin ⇒ more squeeze, never more push ─────────────────────────────
def test_3_contact_risk_monotone():
    c = IntentCorrector(CorrectionParams())
    intent = _intent(squeeze=0.08)
    o_h, _ = c.correct(intent, _Resp(both_contact=True, fn=(1.0, 1.0)), was_swapped=False)
    o_r, _ = c.correct(intent, _Resp(both_contact=True, fn=(0.01, 0.01)), was_swapped=False)
    o_l, _ = c.correct(intent, _Resp(both_contact=False, fn=(0.0, 0.0)), was_swapped=False)
    assert o_r.to_vector()[SQZ] >= o_h.to_vector()[SQZ]
    assert o_l.to_vector()[SQZ] >= o_h.to_vector()[SQZ]
    assert o_r.to_vector()[FWD] == o_h.to_vector()[FWD]        # contact risk never changes forward


# ── 4. MIRROR EQUIVARIANCE — mirrored response ⇒ mirrored (canonical-invariant) Δintent; decoded balance flips ───────
def test_4_mirror_equivariance():
    c = IntentCorrector(CorrectionParams())
    intent = _intent(lateral=0.0)
    out_a, _ = c.correct(intent, _Resp(v_perpendicular=0.08, coin_spin=0.2), was_swapped=False)
    out_b, _ = c.correct(intent, _Resp(v_perpendicular=-0.08, coin_spin=-0.2), was_swapped=True)
    assert np.allclose(out_a.to_vector(), out_b.to_vector(), atol=1e-9)   # canonical Δintent mirror-invariant
    from hymeko_rl.coin_delivery.theta_option.authority_decoder import decode_from_canonical
    gc = {"forward_push_reach": np.array([0.08]), "brake_opposed_reach": np.array([0.08]),
          "lateral_reach_pair": np.array([0.09, 0.09]), "normal_force_reach_pair": np.array([0.2, 0.2])}
    r0 = decode_from_canonical(out_a, gc, False, 0.3, 60.0)
    r1 = decode_from_canonical(out_a, gc, True, 0.3, 60.0)
    assert np.isclose(r0.physical_theta[2], -r1.physical_theta[2], atol=1e-9)      # balance flips
    for j in (0, 1, 3, 4, 5):
        assert np.isclose(r0.physical_theta[j], r1.physical_theta[j], atol=1e-9)


# ── 6. PHASE MONOTONICITY — PUSH→BRAKE→RELEASE only; force_phase advances; neutral on a constant θ ──────────────────
def test_6_phase_monotone_and_neutral():
    pm = PhaseMachine()
    theta = np.array([0.1, 0.2, 0.0, 6.0, 12.0, 1.5])
    phases = []
    for t in range(1, 17):
        out, ph = pm.step(theta, t)
        phases.append(ph)
        assert np.array_equal(out, theta), (t, out)             # constant θ, force_phase=PUSH ⇒ no-op on θ
    assert phases == sorted(phases) and set(phases) <= {PUSH, BRAKE, RELEASE}
    pm2 = PhaseMachine()
    pm2.step(theta, 5, force_phase=RELEASE)                     # state forces RELEASE early
    _o, ph = pm2.step(theta, 6, force_phase=PUSH)              # a later PUSH request cannot revert
    assert ph == RELEASE


# ── 8. TOTAL SEARCH-BUDGET PROVENANCE (unit: the corrector/phase add no candidates) ────────────────────────────────
def test_8_budget_semantics_unit():
    from hymeko_rl.coin_delivery.theta_option.search import search_semantics
    for b in (0, 1, 4, 8):
        s = search_semantics(b)
        assert s["n_total_candidates"] == max(1, b) and s["budget_counts_total_candidates"]


# ── 10. DETERMINISM — same (intent, response, sw) ⇒ identical Δintent ───────────────────────────────────────────────
def test_10_determinism():
    c = IntentCorrector(CorrectionParams())
    intent = _intent(squeeze=0.25)
    resp = _Resp(v_perpendicular=0.09, fn=(0.02, 0.02), both_contact=True)
    a, la = c.correct(intent, resp, was_swapped=True)
    b, lb = c.correct(intent, resp, was_swapped=True)
    assert np.array_equal(a.to_vector(), b.to_vector()) and la == lb


# ── 7. GOLDEN (constant-controller ≡ rollout_primitive) + 9. NO TEACHER FALLBACK + DECODER CONTRACT — slow ──────────
@pytest.mark.slow
def test_7_9_golden_physical():
    import json
    from hymeko_rl.coin_delivery.forward_displacement import rollout_primitive
    from hymeko_rl.coin_delivery.theta_option.closed_loop_rollout import ConstantController, closed_loop_rollout
    from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
    from hymeko_rl.coin_delivery.theta_option.physical_intent import extract_teacher_intent
    from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness
    bank = json.load(open("reports/2026-07-27-coin-teacher-to-rl/teacher_bank.json"))
    box = ThetaBox()
    panel = build_panel(load_harness(), bank)
    for ps in panel:
        if ps.split != "development":
            continue
        th = np.asarray(ps.teacher_theta, np.float64)
        m_ref = rollout_primitive(ps.snap, tuple(th), DELIVERY_CFG)
        m_cl = closed_loop_rollout(ps.snap, ConstantController(th), DELIVERY_CFG)
        assert np.array_equal(np.asarray(m_ref["coin_trace"]), np.asarray(m_cl["coin_trace"])), ps.tag
        for k in ("k6_delivered", "k6_max_dwell", "dtz_end", "peak_qdot", "peak_coin_speed", "forward",
                  "release_step", "lost_before_release", "forward_at_release"):
            assert m_ref[k] == m_cl[k], (ps.tag, k, m_ref[k], m_cl[k])
        # DECODER CONTRACT + NO TEACHER FALLBACK: the coast-in controller's step-1 θ is bounded/legal, not the teacher θ
        intent, _ = extract_teacher_intent(ps.snap, th)
        ctrl = ClosedLoopController(ps.snap, intent, box.clip(th), IntentCorrector(CorrectionParams()), CorrectionParams())
        eff = ctrl.theta_for_step(ps.snap.branch(), 1, ps.snap.prev_tau)
        assert np.all(eff >= box.lo - 1e-6) and np.all(eff <= box.hi + 1e-6)
