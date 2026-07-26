"""R3 physical-intent + deterministic authority-aware decoder — the mandatory gates BEFORE D0 (frozen contract
`reports/2026-07-27-coin-r3-physical-intent-decoder-contract.md`). Tests 1-5 drive the PURE `decode_from_canonical` core
with synthetic canonical authority (fast, no physics); the teacher round-trip (test 6) drives real dev snapshots. No
held-out data (s4/s7) is used for design here."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option.authority_decoder import decode_from_canonical, decode_intent
from hymeko_rl.coin_delivery.theta_option.physical_intent import (
    INTENT_HI, INTENT_LO, INTENT_ROLES, PhysicalIntent, extract_teacher_intent)
from hymeko_rl.coin_delivery.theta_option.semantics import ThetaBox

SLEW, H = 0.3, 60.0
BAL_IDX = 2


def _gc(fr=0.08, br=0.08, lr=0.09, nf=0.20):
    """A synthetic CANONICAL grouped-feature dict carrying only the authority quantities the decoder reads."""
    return {"forward_push_reach": np.array([fr]), "brake_opposed_reach": np.array([br]),
            "lateral_reach_pair": np.array([lr, lr]), "normal_force_reach_pair": np.array([nf, nf])}


def _intent(**kw):
    base = dict(forward_drive=0.05, peak_velocity=0.40, lateral=0.02, squeeze=0.10,
               brake_entry=0.20, braking_demand=0.60, release=0.15)
    base.update(kw)
    return PhysicalIntent(**base)


# ── 1. MIRROR EQUIVARIANCE: same canonical intent, mirrored state -> mirrored physical theta (balance sign reversal) ──
def test_1_mirror_equivariance():
    intent, gc = _intent(), _gc()
    r0 = decode_from_canonical(intent, gc, sw=False, slew=SLEW, horizon=H)
    r1 = decode_from_canonical(intent, gc, sw=True, slew=SLEW, horizon=H)
    assert np.allclose(r0.canonical_theta, r1.canonical_theta), "canonical theta is frame-independent"
    for j in range(6):
        if j == BAL_IDX:
            assert np.isclose(r0.physical_theta[j], -r1.physical_theta[j], atol=1e-9), "balance flips sign under mirror"
        else:
            assert np.isclose(r0.physical_theta[j], r1.physical_theta[j], atol=1e-9), j


# ── 2. BOUNDS AND SLEW CONTRACT: every decoded theta stays in the frozen box (slew-admissible by the option) ──
def test_2_bounds_and_slew():
    box = ThetaBox()
    rng = np.random.default_rng(0)
    for _ in range(40):
        v = INTENT_LO + rng.random(7) * (INTENT_HI - INTENT_LO) * 1.5      # push past bounds to exercise clipping
        gc = _gc(fr=rng.uniform(0.02, 0.15), br=rng.uniform(0.02, 0.15), lr=rng.uniform(0.03, 0.15), nf=rng.uniform(0.1, 0.3))
        for sw in (False, True):
            th = decode_from_canonical(PhysicalIntent.from_vector(v), gc, sw=sw, slew=SLEW, horizon=H).physical_theta
            assert np.all(th >= box.lo - 1e-6) and np.all(th <= box.hi + 1e-6), "decoded theta is legal / slew-admissible"


# ── 3. OBJECT / INTERNAL AUTHORITY SEPARATION: forward+brake use object authority, squeeze uses internal ──
def test_3_object_internal_separation():
    gc = _gc()
    base = decode_from_canonical(_intent(), gc, sw=False, slew=SLEW, horizon=H).canonical_theta
    # forward_drive uses forward_push_reach (object) -> changing fr moves theta[1], NOT theta[0]/theta[5]
    hi_fr = decode_from_canonical(_intent(), _gc(fr=0.16), sw=False, slew=SLEW, horizon=H).canonical_theta
    assert not np.isclose(hi_fr[1], base[1]) and np.isclose(hi_fr[0], base[0]) and np.isclose(hi_fr[5], base[5])
    # squeeze uses normal_force_reach (internal) -> changing nf moves theta[0], NOT theta[1]
    hi_nf = decode_from_canonical(_intent(), _gc(nf=0.30), sw=False, slew=SLEW, horizon=H).canonical_theta
    assert not np.isclose(hi_nf[0], base[0]) and np.isclose(hi_nf[1], base[1])
    # brake uses brake_opposed_reach (object) -> changing br moves theta[5], NOT theta[0]
    hi_br = decode_from_canonical(_intent(), _gc(br=0.16), sw=False, slew=SLEW, horizon=H).canonical_theta
    assert not np.isclose(hi_br[5], base[5]) and np.isclose(hi_br[0], base[0])


# ── 4. MONOTONICITY: more forward intent -> not less forward drive; more braking demand -> not less brake gain ──
def test_4_monotonicity():
    gc = _gc()
    prev_fwd, prev_brk = -1.0, -1.0
    for x in np.linspace(0.0, 0.15, 8):                                    # kept below saturation
        th = decode_from_canonical(_intent(forward_drive=x, braking_demand=x * 10), gc, sw=False, slew=SLEW, horizon=H).canonical_theta
        assert th[1] >= prev_fwd - 1e-9 and th[5] >= prev_brk - 1e-9
        prev_fwd, prev_brk = th[1], th[5]


# ── 5. DETERMINISM AND PROVENANCE: same (intent, authority, sw) -> identical theta + identical diagnostics ──
def test_5_determinism():
    intent, gc = _intent(), _gc()
    a = decode_from_canonical(intent, gc, sw=True, slew=SLEW, horizon=H)
    b = decode_from_canonical(intent, gc, sw=True, slew=SLEW, horizon=H)
    assert np.array_equal(a.physical_theta, b.physical_theta) and a.as_dict() == b.as_dict()
    assert a.weak_link in ("push", "squeeze", "brake", "release")


# ── intent vocabulary contract: bounded, correct length, clip idempotent ──
def test_intent_bounded_and_typed():
    assert len(INTENT_ROLES) == 7 and INTENT_LO.shape == (7,) and INTENT_HI.shape == (7,)
    over = PhysicalIntent.from_vector(INTENT_HI * 3.0).clip().to_vector()
    assert np.all(over <= INTENT_HI + 1e-9) and np.all(over >= INTENT_LO - 1e-9)
    assert np.allclose(PhysicalIntent.from_vector(over).clip().to_vector(), over)   # clip idempotent


# ── 6. TEACHER ROUND-TRIP SEMANTICS: extract-then-decode reconstructs a K6-capable theta in the teacher basin (dev only) ──
@pytest.mark.slow
def test_6_teacher_roundtrip_dev():
    import json
    from hymeko_rl.coin_delivery.forward_displacement import delivery_success, rollout_primitive
    from hymeko_rl.coin_delivery.theta_option.deploy import build_panel
    from hymeko_rl.coin_delivery.theta_option.semantics import DELIVERY_CFG
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import load_harness
    bank = json.load(open("reports/2026-07-27-coin-teacher-to-rl/teacher_bank.json"))
    box = ThetaBox()
    panel = build_panel(load_harness(), bank)
    for ps in panel:
        if ps.split != "development":                                      # dev only in the unit test
            continue
        th = np.asarray(ps.teacher_theta, np.float64)
        i0, _ = extract_teacher_intent(ps.snap, th)
        i1, _ = extract_teacher_intent(ps.snap, th)
        assert np.allclose(i0.to_vector(), i1.to_vector()), "extraction is deterministic"
        assert np.all(i0.to_vector() >= INTENT_LO - 1e-9) and np.all(i0.to_vector() <= INTENT_HI + 1e-9), "intent bounded"
        dec = decode_intent(i0, ps.snap).physical_theta
        assert float(np.linalg.norm(box.norm(dec) - box.norm(th))) < 0.05, "reconstructs the teacher basin (invertible)"
        assert delivery_success(rollout_primitive(ps.snap, tuple(dec), DELIVERY_CFG), DELIVERY_CFG), "decoded theta delivers K6"
