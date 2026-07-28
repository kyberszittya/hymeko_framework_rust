"""Tests for the phase-shaped moving-precursor dynamic capture (Stage 1B).

Fast analytic tests cover the quintic BVP, the backward-tangent precursor, the reference/param dataclasses, the
theta->params map, and the cost. Physics tests (module-scoped fixtures) cover the capture roll (determinism +
continues-from-READY), the frozen-downstream adapter, one reduced-budget end-to-end strict-K6 solve, and the boundary
(degenerate zero-shaping capture does not deliver).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
from hymeko_rl.coin_delivery.theta_option import kinetic_contract as kc
from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
from hymeko_rl.coin_delivery.theta_option.home_states import HOME_STATE_V1_GENERIC, build_home_snapshot
from hymeko_rl.coin_delivery.theta_option.kinetic_residual2 import deterministic_residual
from hymeko_rl.experiments.coin_kinetic_ablation import _rebuild

CKPT = Path("reports/2026-07-28-coin-r9-r2-h1-multiseed/seed_01/checkpoint.json")


# ── fast analytic (no physics) ───────────────────────────────────────────────────────────────────────────────────────
def test_quintic_boundary_conditions():
    q0, v0 = np.array([0.1, -0.2, 0.0, 1.0]), np.array([0.0, 0.1, -0.2, 0.0])
    qf, vf = np.array([0.3, -0.1, 0.2, 1.1]), np.array([-0.4, -0.3, 0.0, 0.2])
    T = 0.2
    c = mp.quintic_coeffs(q0, v0, qf, vf, T)
    p0, u0 = mp.quintic_eval(c, 0.0)
    pT, uT = mp.quintic_eval(c, T)
    assert np.allclose(p0, q0) and np.allclose(u0, v0)
    assert np.allclose(pT, qf, atol=1e-9) and np.allclose(uT, vf, atol=1e-9)


def test_precursor_is_backward_tangent():
    ref = mp.HandoffReference(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 0.0, -1.0, 0.5]),
                              np.zeros(4), 0.01)
    assert np.allclose(ref.precursor(0.0), ref.q_star)
    d3 = ref.q_star - ref.precursor(3.0)
    assert np.allclose(d3, 3.0 * 0.01 * ref.qvel_star)             # recedes exactly along the tangent
    assert abs(ref.speed - np.linalg.norm(ref.qvel_star)) < 1e-12


def test_theta_to_params_bounds_and_cost():
    spec = mp.CaptureSearchSpec(s_max=0.6, knots=2)
    p = mp._theta_to_params(np.array([2.0, -0.5, 1.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), spec)
    assert p.s == 0.6 and p.preload_start == 0.0 and p.bmax == 1.0     # clamped
    assert p.residual.shape == (2, 4)
    assert mp._cost(mp.CaptureOutcome(None, 0.9, 1.0, 0.1, 2, k6=True, min_dtz_mm=5.0, safe=True)) == 0.0
    assert mp._cost(mp.CaptureOutcome(None, 0.9, 1.0, 0.1, 2, k6=False, min_dtz_mm=5.0, safe=True)) == 5.0
    assert mp._cost(mp.CaptureOutcome(None, 0.9, 1.0, 0.1, 2, k6=True, min_dtz_mm=5.0, safe=False)) == 1e3


# ── physics fixtures ─────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rig():
    from hymeko_rl.coin_delivery.theta_option.teacher_bank import acquire_snapshot, load_harness
    from hymeko_rl.experiments.coin_kinetic_r2_rl import _load_clone
    model, norm = _load_clone()
    cradle, _ = acquire_snapshot(load_harness(), kc.S1_SEED)
    coin = _coin_xy(cradle.branch())
    home = build_home_snapshot(cradle, HOME_STATE_V1_GENERIC)
    ready = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, pga.CoinStraddleTargets(coin=coin),
                                pga.TransitConfig()).ready_snapshot
    ref = mp.HandoffReference.from_cradle(cradle)
    r2_fn = deterministic_residual(_rebuild(json.load(open(CKPT))["r2_actor_state"]))
    down = mp.FrozenDownstream(model, norm, r2_fn, cradle.stack)
    return {"cradle": cradle, "ready": ready, "ref": ref, "down": down, "stack": cradle.stack}


def test_reference_from_cradle_deterministic(rig):
    a = mp.HandoffReference.from_cradle(rig["cradle"])
    b = mp.HandoffReference.from_cradle(rig["cradle"])
    assert np.array_equal(a.q_star, b.q_star) and np.array_equal(a.qvel_star, b.qvel_star)


def test_capture_continues_from_ready_and_is_deterministic(rig):
    cap = mp.PhaseShapeCapture(rig["ready"], rig["ref"], rig["stack"])
    rd = rig["ready"].branch().inner.data
    assert np.allclose(cap.q0, rd.qpos[:4]) and np.allclose(cap.v0, rd.qvel[:4])   # continues from READY's exact state
    assert np.allclose(cap.prev0, np.asarray(rig["ready"].prev_tau))
    params = mp.CaptureParams(n=3.0, s=0.3, preload_start=0.5, bmax=0.6, residual=np.zeros((2, 4)))
    a = cap.roll(params)
    b = cap.roll(params)
    da, db = a.snapshot.branch().inner.data, b.snapshot.branch().inner.data
    assert np.array_equal(da.qpos, db.qpos) and np.array_equal(da.qvel, db.qvel)
    assert a.cos_dir == b.cos_dir and a.dtau == b.dtau


def test_frozen_downstream_delivers_from_cradle(rig):
    ref = rig["ref"]
    snap = kc.TransportSnapshot.from_live(rig["cradle"].branch(), rig["stack"], ref.tau_star.copy())
    k6, md, safe, kinds = rig["down"].deliver_with_trace(snap)
    assert k6 and safe and md < 5.0 and kinds.count("HANDOFF_RESET") == 1


def test_degenerate_zero_shaping_capture_does_not_deliver(rig):
    """Boundary regression: a zero-shaping capture (no moving segment) must NOT deliver — the phase-shaping is load-bearing."""
    cap = mp.PhaseShapeCapture(rig["ready"], rig["ref"], rig["stack"])
    out = cap.roll(mp.CaptureParams(n=0.0, s=0.0, preload_start=1.0, bmax=0.0, residual=np.zeros((2, 4))))
    assert not rig["down"].deliver(out.snapshot)[0]


def test_plan_capture_reaches_strict_k6(rig):
    """End-to-end (reduced budget): the structured CEM finds a strict-K6 capture, safe, deterministic per seed."""
    spec = mp.CaptureSearchSpec(population=36, iters=10)
    t0 = time.perf_counter()
    res = mp.plan_capture(rig["ready"], rig["ref"], rig["stack"], rig["down"], seed=11, spec=spec)
    assert res.outcome.k6 and res.outcome.safe
    replay = mp.PhaseShapeCapture(rig["ready"], rig["ref"], rig["stack"]).roll(res.params)
    assert rig["down"].deliver(replay.snapshot)[1] == res.outcome.min_dtz_mm   # deterministic replay
    assert time.perf_counter() - t0 < 120.0
