"""Tests for the R10.2 structured-option torque-path action coordinate (pre-training identity machinery).

Fast: theta decoder (zero -> exact-zero offsets, scale bounds, length precondition), the phase bases (transient pinned to
0 at both ends and hitting its knots, terminal ramp 0->1), the offset composition, and the saturation-mask inference.
Physics: the load-bearing identity — the zero-theta roll reproduces the medoid scaffold BIT-EXACT on the *same* code path
(q, qvel, prev_tau, physical action trace, executable torque path, structured params, contacts), a real 15-D zero-init
actor outputs exactly zero, the zero-theta option delivers strict K6, the phase tube matches the zero-theta trace, and the
theta=0 terminal-offset error is exactly zero. A non-zero theta is shown to actually move the terminal preload (the
coordinate is live) while staying in-bounds. No training loop / reward / exploration is tested here (later boundaries).
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option import moving_precapture as mp
from hymeko_rl.coin_delivery.theta_option import torque_path_option as tpo

SLEW = 0.1     # representative slew for the unit tests (physics tests use the real stack slew)


# ── fast: decoder ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_decode_theta_zero_is_exact_zero():
    opt = tpo.decode_theta(np.zeros(tpo.THETA_DIM), SLEW)
    assert opt.ds == 0.0 and opt.dpreload_start == 0.0 and opt.dbmax == 0.0
    assert np.array_equal(opt.k1, np.zeros(4)) and np.array_equal(opt.k2, np.zeros(4))
    assert np.array_equal(opt.dtau_T, np.zeros(4)) and opt.is_zero


def test_decode_theta_scale_bounds_at_saturation():
    sc = tpo.ThetaScales()
    opt = tpo.decode_theta(np.ones(tpo.THETA_DIM), SLEW, sc)
    assert opt.ds == pytest.approx(sc.s) and opt.dpreload_start == pytest.approx(sc.preload_start)
    assert opt.dbmax == pytest.approx(sc.bmax)
    assert np.allclose(opt.k1, sc.knot_frac * SLEW) and np.allclose(opt.dtau_T, sc.terminal_frac * SLEW)
    # out-of-range input is clipped to the band, not extrapolated
    hi = tpo.decode_theta(np.full(tpo.THETA_DIM, 5.0), SLEW, sc)
    assert hi.ds == pytest.approx(sc.s) and np.allclose(hi.dtau_T, sc.terminal_frac * SLEW)


def test_decode_theta_length_precondition():
    with pytest.raises(AssertionError):
        tpo.decode_theta(np.zeros(4), SLEW)


# ── fast: phase bases ────────────────────────────────────────────────────────────────────────────────────────────────
def test_transient_basis_pinned_to_zero_at_endpoints():
    k1, k2 = np.array([1.0, -2.0, 0.5, 3.0]), np.array([-1.0, 2.0, -0.5, 1.0])
    assert np.array_equal(tpo.transient_basis(0.0, k1, k2), np.zeros(4))
    assert np.array_equal(tpo.transient_basis(1.0, k1, k2), np.zeros(4))
    assert np.array_equal(tpo.transient_basis(0.5, np.zeros(4), np.zeros(4)), np.zeros(4))


def test_transient_basis_hits_its_knots():
    k1, k2 = np.array([1.0, -2.0, 0.5, 3.0]), np.array([-1.0, 2.0, -0.5, 1.0])
    assert np.array_equal(tpo.transient_basis(tpo.TRANSIENT_KNOT_PHASES[0], k1, k2), k1)
    assert np.array_equal(tpo.transient_basis(tpo.TRANSIENT_KNOT_PHASES[1], k1, k2), k2)


def test_transient_basis_is_c1_at_knots_and_boundaries():
    """The upgraded (Catmull-Rom) transient basis is C1: continuous first derivative at the interior knots and zero slope
    at both endpoints (a linear-interp basis would fail the knot-continuity assertion)."""
    k1, k2 = np.array([1.0, -2.0, 0.5, 3.0]), np.array([-1.0, 2.0, -0.5, 1.0])
    h = 1e-6
    for knot in tpo.TRANSIENT_KNOT_PHASES:
        left = (tpo.transient_basis(knot, k1, k2) - tpo.transient_basis(knot - h, k1, k2)) / h
        right = (tpo.transient_basis(knot + h, k1, k2) - tpo.transient_basis(knot, k1, k2)) / h
        assert np.allclose(left, right, atol=1e-3)                                  # continuous first derivative (C1)
    assert np.allclose((tpo.transient_basis(h, k1, k2) - tpo.transient_basis(0.0, k1, k2)) / h, 0.0, atol=1e-3)
    assert np.allclose((tpo.transient_basis(1.0, k1, k2) - tpo.transient_basis(1.0 - h, k1, k2)) / h, 0.0, atol=1e-3)


def test_terminal_basis_is_a_zero_to_one_ramp():
    assert tpo.terminal_basis(0.0) == 0.0 and tpo.terminal_basis(1.0) == 1.0
    assert tpo.terminal_basis(0.5) == pytest.approx(0.5)
    assert tpo.terminal_basis(-1.0) == 0.0 and tpo.terminal_basis(2.0) == 1.0     # clamped domain
    assert tpo.terminal_basis(0.25) < tpo.terminal_basis(0.75)                    # monotone


def test_torque_path_offset_vanishes_at_phase0_and_for_zero_option():
    opt = tpo.decode_theta(np.ones(tpo.THETA_DIM), SLEW)
    assert np.array_equal(tpo.torque_path_offset(0.0, opt), np.zeros(4))          # both bases vanish at phase 0
    zero = tpo.decode_theta(np.zeros(tpo.THETA_DIM), SLEW)
    for phase in (0.0, 0.3, 0.5, 0.7, 1.0):
        assert np.array_equal(tpo.torque_path_offset(phase, zero), np.zeros(4))   # zero option -> zero offset everywhere


def test_step_masks_zero_policy_never_action_clips():
    a_pi0 = np.array([1.0, -1.0, 0.3, -0.9])            # two joints at the slew bound
    m = tpo._step_masks(a_pi0, np.zeros(4), np.zeros(4), np.zeros(4), SLEW, np.full(4, -10.0), np.full(4, 10.0))
    assert np.array_equal(m.slew_limited, np.array([True, True, False, False]))
    assert not np.any(m.action_clipped)                 # offset 0 -> composed == a_pi0 in [-1,1]
    # a composed correction beyond +/-1 does register as action-clipped
    m2 = tpo._step_masks(np.array([0.9, 0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0, 0.0]),
                         np.zeros(4), np.zeros(4), SLEW, np.full(4, -10.0), np.full(4, 10.0))
    assert bool(m2.action_clipped[0]) and not np.any(m2.action_clipped[1:])


# ── physics: the identity (module-scoped rig reused from the audit — no third rig copy) ───────────────────────────────
@pytest.fixture(scope="module")
def rig():
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
    return _rig()


@pytest.fixture(scope="module")
def roller(rig):
    return tpo.TorquePathCaptureRoll(rig["ready"], rig["ref"], rig["stack"], rig["pi0"], rig["coin"])


def test_structural_params_zero_is_pi0_bit_exact(roller, rig):
    p = roller.structural_params(tpo.decode_theta(np.zeros(tpo.THETA_DIM), roller.slew))
    pi0 = rig["pi0"]
    assert p.s == pi0.s and p.preload_start == pi0.preload_start and p.bmax == pi0.bmax and p.n == pi0.n


def test_structural_params_nonzero_shifts_within_bounds(roller):
    opt = tpo.decode_theta(np.ones(tpo.THETA_DIM), roller.slew)
    p = roller.structural_params(opt)
    assert p.s != roller.pi0.s and 0.05 <= p.s <= 0.6 and 0.0 <= p.preload_start <= 1.0 and 0.0 <= p.bmax <= 1.0


def test_zero_theta_is_scaffold_bit_exact(roller, rig):
    """Cornerstone: zero-theta reproduces PhaseShapeCapture.roll(pi0) bit-exact (same code path)."""
    scaffold = mp.PhaseShapeCapture(rig["ready"], rig["ref"], rig["stack"]).roll(rig["pi0"])
    res = roller.rollout(np.zeros(tpo.THETA_DIM, dtype=np.float32))
    sd, rd = scaffold.snapshot.branch().inner.data, res["snapshot"].branch().inner.data
    assert np.array_equal(sd.qpos, rd.qpos) and np.array_equal(sd.qvel, rd.qvel)
    assert np.array_equal(np.asarray(scaffold.snapshot.prev_tau), np.asarray(res["prev"]))
    assert all(np.all(np.abs(a) <= 1.0) for a in res["acts"])


def test_real_zero_theta_actor_outputs_exactly_zero(rig, roller):
    from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
    res = roller.rollout(np.zeros(tpo.THETA_DIM, dtype=np.float32))
    fn = crl.policy_residual(crl.make_zero_actor(1, act_dim=tpo.THETA_DIM))
    out = np.array([fn(o) for o in res["obs"]])
    assert out.shape[1] == tpo.THETA_DIM and np.all(out == 0.0)


def test_zero_theta_delivers_strict_k6(roller, rig):
    res = roller.rollout(np.zeros(tpo.THETA_DIM, dtype=np.float32))
    k6, md, safe = rig["down"].deliver(res["snapshot"])
    assert k6 and safe and md < 10.0


def test_record_phase_tube_shapes_and_self_consistent(roller, rig):
    tube = tpo.record_phase_tube(roller)
    steps = rig["pi0"].steps
    assert tube["q0"].shape == (steps, 4) and tube["qvel0"].shape == (steps, 4) and tube["tau0"].shape == (steps, 4)
    assert tube["tau0_terminal"].shape == (4,) and not np.isnan(tube["q0"]).any()
    res = roller.rollout(np.zeros(tpo.THETA_DIM, dtype=np.float32))
    assert np.array_equal(tube["q0"], np.asarray(res["obs"])[:, 0:4])       # tube IS the zero-theta trajectory


def test_terminal_offset_zero_theta_is_exactly_zero(roller):
    tube = tpo.record_phase_tube(roller)
    rep = tpo.terminal_offset_report(roller, np.zeros(tpo.THETA_DIM, dtype=np.float32), tube)
    assert rep["err_norm"] == 0.0 and np.array_equal(rep["requested"], np.zeros(4))
    assert np.array_equal(rep["executed"], np.zeros(4))


def test_nonzero_theta_moves_terminal_preload_but_stays_in_bounds(roller):
    """The coordinate is live: a non-zero terminal-offset theta shifts the executed terminal preload off the tube, while
    every physical action stays in [-1,1] (governed stack inherited)."""
    tube = tpo.record_phase_tube(roller)
    z = np.zeros(tpo.THETA_DIM, dtype=np.float32)
    z[11:15] = 1.0                                                          # request a full-scale terminal offset
    res = roller.rollout(z)
    executed = np.asarray(res["prev"]) - tube["tau0_terminal"]
    assert np.linalg.norm(executed) > 1e-6                                  # actually moved
    assert all(np.all(np.abs(a) <= 1.0) for a in res["acts"])              # still bounded
