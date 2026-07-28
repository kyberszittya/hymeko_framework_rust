"""Tests for the R10.2 Boundary-3 conditioning: the exploration transform, the frozen per-dimension normalization, and
the local ``+/- eps`` sensitivity audit.

Fast (no physics): ``sample_theta`` (sigma=0 -> exact zero option; per-dimension ``D`` scaling; clipped to [-1,1]) and
``freeze_normalization`` (bounded, inverse to the physical effect). Physics: ``axis_sensitivity`` on the frozen scaffold
roller — every option dimension moves the task (no dead dim), the terminal-offset dims move the terminal torque, and the
standardised Jacobian SVD has a sane rank. No training / reward / actor update is exercised.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option import torque_path_conditioning as cond
from hymeko_rl.coin_delivery.theta_option import torque_path_option as tpo


def _resp(dim: int, effect: float) -> cond.AxisResponse:
    d = np.zeros(12)
    d[0] = effect                                              # ||direction|| == |effect|
    return cond.AxisResponse(dim=dim, d_action_rms=0.0, d_tau_term=0.0, d_q_term=0.0, d_qvel_term=0.0,
                             d_min_dtz=0.0, direction=d)


# ── fast ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_sample_theta_zero_sigma_is_exact_zero():
    d = np.full(tpo.THETA_DIM, 1.5)
    z = np.linspace(-2, 2, tpo.THETA_DIM)
    assert np.array_equal(cond.sample_theta(0.0, d, z), np.zeros(tpo.THETA_DIM, dtype=np.float32))


def test_sample_theta_applies_D_per_dim_and_clips():
    d = np.array([1.0, 2.0] + [0.5] * (tpo.THETA_DIM - 2))
    out = cond.sample_theta(0.1, d, np.ones(tpo.THETA_DIM))
    assert out[0] == pytest.approx(0.1) and out[1] == pytest.approx(0.2) and out[2] == pytest.approx(0.05)
    big = cond.sample_theta(1.0, np.full(tpo.THETA_DIM, 5.0), np.full(tpo.THETA_DIM, 3.0))
    assert np.all(big <= 1.0) and np.all(big >= -1.0)          # clipped to the actor's tanh range


def test_freeze_normalization_bounded_and_inverse_to_effect():
    responses = [_resp(i, e) for i, e in enumerate(np.linspace(0.2, 3.0, tpo.THETA_DIM))]
    d = cond.freeze_normalization(responses)
    assert d.shape == (tpo.THETA_DIM,) and np.all(d >= cond._D_CLAMP[0]) and np.all(d <= cond._D_CLAMP[1])
    assert d[0] >= d[-1]                                       # the weakest-effect dim gets the larger normalisation


def test_freeze_normalization_does_not_amplify_dead_dim_beyond_clamp():
    responses = [_resp(i, 1.0) for i in range(tpo.THETA_DIM)]
    responses[3] = _resp(3, 0.0)                              # a dead dim
    d = cond.freeze_normalization(responses)
    assert d[3] == cond._D_CLAMP[1]                           # clamped, not infinite


# ── physics (rig reused from the audit — no third rig copy) ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def nominal_roller():
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
    rig = _rig()
    roller = tpo.TorquePathCaptureRoll(rig["ready"], rig["ref"], rig["stack"], rig["pi0"], rig["coin"])
    return roller, rig["down"]


def test_axis_sensitivity_every_dim_moves_the_task(nominal_roller):
    roller, down = nominal_roller
    sens = cond.axis_sensitivity(roller, down)
    assert len(sens["responses"]) == tpo.THETA_DIM
    assert 1 <= sens["effective_rank"] <= 12 and len(sens["singular_values"]) == 12
    assert all(abs(r.d_min_dtz) > 1e-6 for r in sens["responses"])          # no dim is exactly task-dead
    for r in sens["responses"][11:15]:                                       # the terminal-offset dims move the preload
        assert r.d_tau_term > 0.0


def test_freeze_normalization_from_real_sensitivity_is_bounded(nominal_roller):
    roller, down = nominal_roller
    d = cond.freeze_normalization(cond.axis_sensitivity(roller, down)["responses"])
    assert d.shape == (tpo.THETA_DIM,) and np.all(d >= cond._D_CLAMP[0]) and np.all(d <= cond._D_CLAMP[1])
