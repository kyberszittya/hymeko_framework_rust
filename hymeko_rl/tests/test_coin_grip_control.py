"""Tests for COIN-GRIP-CONTROL-1 (active clamp-load regulation + friction audit).

Covers: the actuator-capability facts (position servos, observable contact force), coordinate decoupling (§4: midpoint
c ⊥ clamp d in the factored action), the observed normal contact forces, the four position-based controllers, and the
STRICT grasp-integrity predicate that separates a genuine bilateral grasp-transport from a single-finger bulldoze —
the guard that falsified the campaign's clamp-regulation hypothesis.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from hymeko_rl.env.contact_formation_env import ContactFormationEnv
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.experiments.coin_delivery_acquisition1 import _states
from hymeko_rl.experiments.exp_v3_handoff_gate import _DIFF, _MAX_STEPS
from hymeko_rl.experiments.pedc_selection import _C1_HORIZON, _ctx, _load_pkl_bank, c1_config
from hymeko_rl.train.coin_delivery_acquisition import AcqParams, ApproachMode
from hymeko_rl.train.coin_grip_control import (
    GripController,
    GripParams,
    controlled_micro_transport,
    grip_action,
    normal_contact_forces,
)
from hymeko_rl.train.coin_transport import extract_handoffs, obj_to_env


def _acq() -> AcqParams:
    d = json.loads(open("experiments/2026_07_20_coin_delivery_acquisition/manifests/coin_delivery_acquisition.json").read())["best_params"]
    d = {k: (ApproachMode(v) if k == "approach_mode" else v) for k, v in d.items()}
    return replace(AcqParams(**d), regrasp=False)


def _env(**kw) -> ContactFormationEnv:
    planar = PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=_DIFF, task_graph=True,
                            fingertip_shape="capsule", fingertip_size="0.012 0.03", **kw)
    return ContactFormationEnv(planar, _load_pkl_bank("c1_heldseed_bank.pkl", holdout=False), _ctx()["contract"],
                               horizon=_C1_HORIZON, cfg=c1_config(), seed=0)


def _handoffs(env):
    return extract_handoffs(env, _states()["acquisition_wall"], _acq())


# --- §4 coordinate decoupling: the factored action keeps midpoint c ⊥ clamp d ------------------------------------
def test_midpoint_action_is_decoupled_from_clamp_dims() -> None:
    a_mid = obj_to_env(np.array([0.5, -0.3]), 0.0, 0.0, 0.0)          # midpoint move only
    assert a_mid[2] == 0.0 and a_mid[3] == 0.0 and a_mid[4] == 0.0     # aperture/squeeze/differential untouched
    a_clamp = obj_to_env(np.array([0.0, 0.0]), 0.7, -0.2, 0.6)         # clamp-only command
    assert a_clamp[0] == 0.0 and a_clamp[1] == 0.0                     # midpoint untouched → orthogonal axes


def test_grip_action_returns_six_dof_for_every_controller() -> None:
    env = _env()
    H = _handoffs(env)
    assert H, "need at least one certified handoff"
    from hymeko_rl.train.coin_transport import restore_planar
    inner = env._env
    restore_planar(inner, H[0].snap)
    obs = H[0].obs.copy()
    d = np.array([1.0, 0.0])
    for c in GripController:
        a = grip_action(inner, obs, c, GripParams(), d)
        assert a.shape == (6,) and np.all(np.abs(a) <= 1.0 + 1e-6)


# --- observed normal contact forces (a real force, mj_contactForce) ---------------------------------------------
def test_normal_contact_forces_nonnegative_pair() -> None:
    env = _env()
    H = _handoffs(env)
    from hymeko_rl.train.coin_transport import restore_planar
    restore_planar(env._env, H[0].snap)
    fl, fr = normal_contact_forces(env._env)
    assert fl >= 0.0 and fr >= 0.0                                    # magnitudes, never negative


# --- controllers + strict grasp-integrity predicate -------------------------------------------------------------
def test_controlled_micro_transport_reports_strict_keys() -> None:
    env = _env()
    H = _handoffs(env)
    r = controlled_micro_transport(env, H[0], GripController.C1_APERTURE_FB, GripParams())
    for k in ("rho", "retained", "both_frac", "align", "final_bilateral", "micro_transport", "micro_strict"):
        assert k in r
    assert isinstance(r["micro_strict"], bool) and isinstance(r["micro_transport"], bool)


def test_strict_predicate_requires_final_bilateral() -> None:
    # the strict grasp-integrity guard cannot pass without ending bilateral (rules out a single-finger bulldoze)
    env = _env()
    H = _handoffs(env)
    rows = [controlled_micro_transport(env, h, GripController.C1_APERTURE_FB, GripParams()) for h in H]
    for r in rows:
        if r["micro_strict"]:
            assert r["final_bilateral"] and r["both_frac"] >= 0.75 and r["align"] >= 0.7


def test_position_clamp_does_not_strictly_micro_transport() -> None:
    # the campaign's decisive measurement: NO position-based clamp controller strictly micro-transports (Case E_falsified)
    env = _env()
    H = _handoffs(env)
    for c in GripController:
        rows = [controlled_micro_transport(env, h, c, GripParams()) for h in H]
        assert sum(r["micro_strict"] for r in rows) == 0            # strict grasp-integrity: 0 at canonical friction
