"""Tests for the R10.2 Boundary-4 HOME-start env + K6-dominant reward + the frozen exploration coordinate.

Fast: the frozen ``D``/``sigma``/decision record (honest — strict gate did NOT pass), the outcome classifier, and the
K6-dominant reward ordering (k6 >> safe_negative > boundary_route_variation, safety dominates; tube is a mild capped
regulariser). Physics: the env's zero-theta option is a strict-K6 success through the whole chain, the explicit
HOME->READY->option->downstream->K6 end-to-end delivery, and a drift guard that the frozen ``D`` still equals the
recomputed sensitivity normalization. No training / actor update is exercised here.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.theta_option import torque_path_env as tpe
from hymeko_rl.coin_delivery.theta_option import torque_path_frozen as frz
from hymeko_rl.coin_delivery.theta_option import torque_path_option as tpo


# ── fast ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_frozen_coordinate_shape_and_helper():
    assert len(frz.FROZEN_D) == tpo.THETA_DIM and frz.SIGMA == 0.05
    assert np.array_equal(frz.frozen_normalization(), np.array(frz.FROZEN_D))


def test_review_decision_is_honest_not_a_strict_pass():
    d = frz.REVIEW_DECISION
    assert d["verdict"] == "STRUCTURED_THETA_EXPLORATION_REVIEW_ACCEPTED"
    assert d["strict_preregistered_admissibility"] is False and d["reviewed_training_admissibility"] is True
    assert d["physical_safety_violations"] == "0/96"


def test_classify_priority():
    assert tpe.classify(True, False, 1) == "unsafe"           # safety dominates even a "K6"
    assert tpe.classify(True, True, 2) == "boundary_route_variation"
    assert tpe.classify(True, True, 1) == "k6"
    assert tpe.classify(False, True, 1) == "safe_negative"


def test_reward_ordering_and_k6_dominance():
    w = tpe.RewardWeights()
    r_k6 = tpe.option_reward("k6", 3.0, 0.0, w)
    r_neg = tpe.option_reward("safe_negative", 40.0, 0.0, w)
    r_bnd = tpe.option_reward("boundary_route_variation", 40.0, 0.0, w)
    r_uns = tpe.option_reward("unsafe", 40.0, 0.0, w)
    assert r_k6 > r_neg > r_bnd > r_uns
    # a K6 with the WORST possible tube/progress still beats a safe_negative with the best — K6 is dominant
    assert tpe.option_reward("k6", w.miss_cap_mm, w.tube_cap, w) > tpe.option_reward("safe_negative", 0.0, 0.0, w)


def test_tube_penalty_zero_on_tube_and_capped():
    steps = 20
    tube = {"q0": np.zeros((steps, 4)), "qvel0": np.zeros((steps, 4)), "tau0": np.zeros((steps, 4))}
    on_tube = np.zeros((steps, 12 + 19))                      # obs[:, :12] == tube -> zero deviation
    assert tpe._tube_penalty(on_tube, tube, tpe.RewardWeights()) == 0.0
    far = np.zeros((steps, 31))
    far[:, 0:12] = 100.0
    assert tpe._tube_penalty(far, tube, tpe.RewardWeights()) == tpe.RewardWeights().tube_cap   # capped


# ── physics (rig reused from the audit) ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def rig():
    from hymeko_rl.experiments.coin_kinetic_capture_exploration_audit import _rig
    return _rig()


def test_env_zero_theta_is_strict_k6(rig):
    from hymeko_rl.coin_delivery.theta_option import capture_rl as crl
    env = tpe.StructuredOptionCaptureEnv(rig, crl.perturbation_panel(n=3, seed=90210))
    obs = env.reset(0)
    assert obs.shape == (crl.OBS_DIM,)
    s2, r, done, info = env.step(np.zeros(tpo.THETA_DIM, dtype=np.float32))
    assert done and info["class"] == "k6" and info["reset"] == 1 and info["min_dtz"] < 10.0 and r > 9.0


def test_home_to_k6_full_chain_zero_theta(rig):
    """HOME-start requirement: HOME -> frozen analytic HOME->READY -> zero-theta option -> frozen downstream -> strict K6."""
    from hymeko_rl.coin_delivery.forward_displacement import _coin_xy
    from hymeko_rl.coin_delivery.theta_option import planar_geometric_approach as pga
    from hymeko_rl.coin_delivery.theta_option.home_states import HOME_STATE_V1_GENERIC, build_home_snapshot
    home = build_home_snapshot(rig["cradle"], HOME_STATE_V1_GENERIC)
    coin = _coin_xy(rig["cradle"].branch())
    ready = pga.execute_transit(home, HOME_STATE_V1_GENERIC.q, pga.CoinStraddleTargets(coin=coin),
                                pga.TransitConfig()).ready_snapshot
    roller = tpo.TorquePathCaptureRoll(ready, rig["ref"], rig["stack"], rig["pi0"], rig["coin"])
    res = roller.rollout(np.zeros(tpo.THETA_DIM, dtype=np.float32))
    k6, md, safe = rig["down"].deliver(res["snapshot"])
    assert k6 and safe and md < 10.0


def test_frozen_D_matches_recomputed_sensitivity(rig):
    from hymeko_rl.coin_delivery.theta_option import torque_path_conditioning as cond
    roller = tpo.TorquePathCaptureRoll(rig["ready"], rig["ref"], rig["stack"], rig["pi0"], rig["coin"])
    d = cond.freeze_normalization(cond.axis_sensitivity(roller, rig["down"])["responses"])
    assert np.allclose(d, np.array(frz.FROZEN_D), atol=1e-6)   # drift guard: frozen D still equals the audit's D
