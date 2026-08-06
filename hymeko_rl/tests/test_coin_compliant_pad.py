"""Tests for COIN-COMPLIANT-PAD-1 (contact-patch & tangential-coupling audit).

Covers: compliance-parameter serialization (rigid vs compliant matching), the contact-model inventory, the contact-
wrench decomposition (F_N/F_T/η) and support-resistance calculation, per-side compliance symmetry, and — critically for
metric integrity — the strict micro-transport predicate's CALIBRATION: it must PASS a synthetic firm-grip drag and
REJECT bulldoze / slip / soft-grip / support-only motion. If the predicate could not pass an idealized transport, it
would be pathologically strict and every downstream verdict would be a phantom.
"""
from __future__ import annotations

import json
from dataclasses import replace

import mujoco

from hymeko_rl.env.contact_formation_env import ContactFormationEnv
from hymeko_rl.env.planar_grasp_env import PlanarGraspEnv
from hymeko_rl.experiments.coin_delivery_acquisition1 import _states
from hymeko_rl.experiments.exp_v3_handoff_gate import _DIFF, _MAX_STEPS
from hymeko_rl.experiments.pedc_selection import _C1_HORIZON, _ctx, _load_pkl_bank, c1_config
from hymeko_rl.train.coin_compliant_pad import (
    Compliance,
    compliance_params,
    contact_model_inventory,
    contact_wrench_budget,
    set_runtime_compliance,
    strict_micro_transport_pass,
)
from hymeko_rl.train.coin_delivery_acquisition import AcqParams, ApproachMode
from hymeko_rl.train.coin_transport import extract_handoffs, restore_planar

_FIRM = dict(coin_disp=0.05, align=0.99, rho=0.95, both_frac=1.0, final_bilateral=True,
             max_eta=0.6, min_FN=2.0, penetration_max=0.003, solver_ok=True)


def _acq() -> AcqParams:
    d = json.loads(open("experiments/2026_07_20_coin_delivery_acquisition/manifests/coin_delivery_acquisition.json").read())["best_params"]
    d = {k: (ApproachMode(v) if k == "approach_mode" else v) for k, v in d.items()}
    return replace(AcqParams(**d), regrasp=False)


def _env(compliance=Compliance.P0_RIGID) -> ContactFormationEnv:
    planar = PlanarGraspEnv(robot=None, max_steps=_MAX_STEPS, difficulty=_DIFF, task_graph=True,
                            fingertip_shape="capsule", fingertip_size="0.012 0.03",
                            fingertip_compliance=compliance_params(compliance))
    return ContactFormationEnv(planar, _load_pkl_bank("c1_heldseed_bank.pkl", holdout=False), _ctx()["contract"],
                               horizon=_C1_HORIZON, cfg=c1_config(), seed=0)


# --- compliance serialization: rigid = canonical, compliant = softer solref + priority ---------------------------
def test_compliance_params_rigid_is_none_compliant_is_softer() -> None:
    assert compliance_params(Compliance.P0_RIGID) is None
    (solref, _solimp) = compliance_params(Compliance.P2_MODERATE)
    assert solref[0] > 0.02                                       # softer than the canonical 0.02 timeconst


def test_rigid_env_matches_canonical_solref() -> None:
    m = _env(Compliance.P0_RIGID)._env.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    assert list(m.geom_solref[gid]) == [0.02, 1.0] and int(m.geom_priority[gid]) == 0


def test_compliant_env_applies_softer_contact_and_priority() -> None:
    m = _env(Compliance.P2_MODERATE)._env.model
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    assert m.geom_solref[gid][0] > 0.02 and int(m.geom_priority[gid]) == 1   # fingertip governs the fingertip↔coin contact


# --- inventory + wrench decomposition ---------------------------------------------------------------------------
def test_contact_model_inventory_reports_solver_and_support() -> None:
    inv = contact_model_inventory(_env()._env)
    for k in ("timestep", "integrator", "cone", "fingertip_solref", "coin_support_dofs"):
        assert k in inv
    assert inv["coin_support_dofs"]["disk_x"]["damping"] > 0.0     # the coin slide-joint damping IS the table resistance


def _handoffs(env):
    return extract_handoffs(env, _states()["acquisition_wall"], _acq())


def test_wrench_budget_decomposes_and_bounds_eta() -> None:
    env = _env()
    H = _handoffs(env)
    restore_planar(env._env, H[0].snap)
    wb = contact_wrench_budget(env._env)
    for k in ("F_N_min", "F_T_total", "eta", "support_resistance", "penetration_max", "finger_exceeds_support"):
        assert k in wb
    assert wb["F_N_min"] >= 0.0 and wb["F_T_total"] >= 0.0 and wb["support_resistance"] >= 0.0
    assert 0.0 <= wb["eta"] <= 1.5                                # per-contact friction utilization, physical range


def test_set_runtime_compliance_is_per_side() -> None:
    env = _env(Compliance.P0_RIGID)
    set_runtime_compliance(env._env, Compliance.P2_MODERATE, Compliance.P0_RIGID)   # left compliant, right rigid
    m = env._env.model
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    gr = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_right")
    assert m.geom_solref[gl][0] > 0.02 and m.geom_solref[gr][0] == 0.02   # asymmetric compliance applied


# --- metric-integrity: the strict predicate must PASS a firm drag and REJECT the artifacts ----------------------
def test_strict_predicate_passes_firm_grip_drag() -> None:
    # calibration: an idealized firm-grip transport (high ρ, low η, bilateral, bounded) MUST pass — else the predicate
    # is pathologically strict and every downstream verdict is a phantom.
    assert strict_micro_transport_pass(**_FIRM) is True


def test_strict_predicate_rejects_bulldoze_and_slip_and_soft() -> None:
    assert strict_micro_transport_pass(**{**_FIRM, "final_bilateral": False}) is False   # single-finger bulldoze
    assert strict_micro_transport_pass(**{**_FIRM, "both_frac": 0.4}) is False            # lost clamp mid-motion
    assert strict_micro_transport_pass(**{**_FIRM, "max_eta": 1.0}) is False              # friction-cone saturation (slip)
    assert strict_micro_transport_pass(**{**_FIRM, "min_FN": 0.0}) is False               # soft/slack grip (support-only)
    assert strict_micro_transport_pass(**{**_FIRM, "align": 0.2}) is False                # perpendicular squirt
    assert strict_micro_transport_pass(**{**_FIRM, "penetration_max": 0.05}) is False     # runaway penetration
    assert strict_micro_transport_pass(**{**_FIRM, "rho": 0.1}) is False                  # coin lags the midpoint
