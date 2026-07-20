"""Tests for COIN-GRASP-CERT-1 (contact-mode classifier + clamp-establishment + active certification).

Covers: the contact-mode state machine (N/L/R/B_t/B_p/S/J classification), dwell accumulation/reset, slip/jam
classification, the grasp-establishment actors + params, the certification probes, and the establish/certify/
micro-transport rollout structure.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from hymeko_rl.experiments.coin_delivery_acquisition1 import _states
from hymeko_rl.train.coin_delivery_acquisition import AcqParams, ApproachMode
from hymeko_rl.train.coin_grasp_cert import (
    ContactModeClassifier,
    GraspFamily,
    GraspParams,
    Mode,
    Probe,
    certify_grasp,
    establish_grasp,
    grasp_action,
    micro_transport,
)
from hymeko_rl.train.coin_delivery_acquisition import make_acq_env
from hymeko_rl.train.coin_transport import extract_handoffs


class _M:
    def __init__(self, left, right, speed=0.0, arm=False):
        self.left_contact, self.right_contact, self.disk_speed = left, right, speed
        self.arm_self_contact, self.fingers_self_contact = arm, False


class _Inner:
    def __init__(self, m):
        self._planar_metrics = m


def _acq() -> AcqParams:
    d = json.loads(open("experiments/2026_07_20_coin_delivery_acquisition/manifests/coin_delivery_acquisition.json").read())["best_params"]
    d = {k: (ApproachMode(v) if k == "approach_mode" else v) for k, v in d.items()}
    return replace(AcqParams(**d), regrasp=False)


# ── contact-mode classifier ──────────────────────────────────────────────────────────────────────────────────────────
def test_mode_none() -> None:
    assert ContactModeClassifier().classify(_Inner(_M(False, False))) == Mode.N


def test_mode_left_right() -> None:
    assert ContactModeClassifier().classify(_Inner(_M(True, False))) == Mode.L
    assert ContactModeClassifier().classify(_Inner(_M(False, True))) == Mode.R


def test_mode_bilateral_transient_then_preloaded() -> None:
    clf = ContactModeClassifier(preload_dwell=3)
    inner = _Inner(_M(True, True, speed=0.0))
    modes = [clf.classify(inner) for _ in range(5)]
    assert modes[0] == Mode.B_t and modes[1] == Mode.B_t     # transient until dwell reached
    assert modes[-1] == Mode.B_p                             # preloaded after dwell with low slip


def test_mode_bilateral_high_slip_not_preloaded() -> None:
    clf = ContactModeClassifier(preload_dwell=1)
    m = clf.classify(_Inner(_M(True, True, speed=0.5)))       # both but fast slip → transient, not preloaded
    assert m == Mode.B_t


def test_mode_slipping_one_sided() -> None:
    assert ContactModeClassifier().classify(_Inner(_M(True, False, speed=0.5))) == Mode.S


def test_mode_jam() -> None:
    assert ContactModeClassifier().classify(_Inner(_M(True, True, arm=True))) == Mode.J


def test_dwell_resets_on_contact_loss() -> None:
    clf = ContactModeClassifier(preload_dwell=3)
    for _ in range(4):
        clf.classify(_Inner(_M(True, True)))                 # build dwell
    clf.classify(_Inner(_M(False, False)))                   # lose contact
    assert clf.both_dwell == 0


# ── grasp actors + params ────────────────────────────────────────────────────────────────────────────────────────────
def test_grasp_params_from_unit() -> None:
    lo = GraspParams.from_unit(np.zeros(4))
    assert lo.close_aperture == -0.2 and lo.squeeze == 0.4


def test_grasp_action_shape_and_no_midpoint_translation() -> None:
    a = grasp_action(_Inner(_M(True, True)), np.zeros(41, np.float32), Mode.B_t, GraspFamily.G1_SYMMETRIC, GraspParams())
    assert a.shape == (6,)
    assert a[0] == np.float32(0) and a[1] == np.float32(0)   # G1 establishes the grasp IN PLACE (no midpoint move)


def test_grasp_action_asymmetric_seeks_missing_side() -> None:
    p = GraspParams(differential=0.5)
    aL = grasp_action(_Inner(_M(True, False)), np.zeros(41, np.float32), Mode.L, GraspFamily.G2_ASYMMETRIC, p)
    aR = grasp_action(_Inner(_M(False, True)), np.zeros(41, np.float32), Mode.R, GraspFamily.G2_ASYMMETRIC, p)
    assert np.sign(aL[4]) != np.sign(aR[4])                  # seeks opposite sides for L vs R


# ── establish / certify / micro-transport rollouts ───────────────────────────────────────────────────────────────────
def test_establish_grasp_keys() -> None:
    env = make_acq_env()
    H = extract_handoffs(env, _states()["acquisition_wall"][:6], _acq())
    r = establish_grasp(env, H[0], GraspFamily.G1_SYMMETRIC, GraspParams(), steps=15)
    for k in ("reached_B_p", "max_bp_dwell", "bilateral_frac", "final_mode", "modes"):
        assert k in r


def test_fingertip_shape_variant() -> None:
    from hymeko_rl.env.planar_grasp_env import make_planar_arms_mjcf, with_fingertip_shape, with_fingertip_sites
    mjcf = with_fingertip_sites(make_planar_arms_mjcf())
    assert with_fingertip_shape(mjcf, "sphere", "0.014") == mjcf          # sphere is a no-op (canonical unchanged)
    box = with_fingertip_shape(mjcf, "box", "0.006 0.016 0.02")
    assert box.count('type="box"') == 2                                   # both fingertips retyped (mirror symmetry)
    assert 'name="fingertip_left"' in box and 'name="fingertip_right"' in box   # frames preserved


def test_certify_and_micro_transport_keys() -> None:
    env = make_acq_env()
    H = extract_handoffs(env, _states()["acquisition_wall"][:6], _acq())
    c = certify_grasp(env, H[0], GraspFamily.G1_SYMMETRIC, GraspParams(), Probe.P4_COMBINED, establish=8, probe_steps=8)
    mt = micro_transport(env, H[0], GraspFamily.G1_SYMMETRIC, GraspParams(), establish=8, epsilons=(0.1, 0.3), move_steps=6)
    assert "certified" in c and "retained" in c
    assert "largest_retaining_eps" in mt and "micro_transport" in mt
