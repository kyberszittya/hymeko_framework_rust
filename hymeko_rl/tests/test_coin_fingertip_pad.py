"""Tests for COIN-FINGERTIP-PAD-2 (fingertip-pad geometry-validity gate).

Covers: the live clamp-geometry measurement (clamp axis + inward alignment) and the flat-pad geometry-validity gate
(the local-inward angular spread that determines whether a fixed inward-facing flat pad is constructible on the
wrist-less arm) — the reusable G-FIX gate that keeps invalid (world-aligned / misoriented) pads out of the result table.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from hymeko_rl.experiments.coin_delivery_acquisition1 import _states
from hymeko_rl.train.coin_delivery_acquisition import AcqParams, ApproachMode, make_acq_env
from hymeko_rl.train.coin_fingertip_pad import clamp_geometry, flat_pad_validity
from hymeko_rl.train.coin_transport import extract_handoffs


def _acq() -> AcqParams:
    d = json.loads(open("experiments/2026_07_20_coin_delivery_acquisition/manifests/coin_delivery_acquisition.json").read())["best_params"]
    d = {k: (ApproachMode(v) if k == "approach_mode" else v) for k, v in d.items()}
    return replace(AcqParams(**d), regrasp=False)


def test_clamp_geometry_keys_and_unit_axis() -> None:
    env = make_acq_env()
    env.reset(seed=64_010)
    g = clamp_geometry(env._env)
    for k in ("clamp_axis", "inward_L", "inward_R", "dotL", "dotR"):
        assert k in g
    assert abs(np.linalg.norm(g["clamp_axis"]) - 1.0) < 1e-6      # unit clamp axis


def test_flat_pad_validity_reports_local_inward_spread() -> None:
    env = make_acq_env()
    H = extract_handoffs(env, _states()["acquisition_wall"][:6], _acq())
    fv = flat_pad_validity(env, H)
    for k in ("local_inward_angle_std_deg", "mean_inward_dot_clamp_axis", "fixed_flat_pad_geometry_valid"):
        assert k in fv
    assert fv["local_inward_angle_std_deg"] >= 0.0


def test_flat_pad_invalid_when_inward_direction_varies() -> None:
    # the full 12-handoff cohort has a large local-inward spread (no wrist DOF) → a fixed flat pad is geometry-INVALID
    env = make_acq_env()
    H = extract_handoffs(env, _states()["acquisition_wall"], _acq())
    fv = flat_pad_validity(env, H)
    assert fv["local_inward_angle_std_deg"] > 15.0                # the arm has no wrist DOF → wide orientation spread
    assert fv["fixed_flat_pad_geometry_valid"] is False          # so a fixed inward-facing flat pad cannot be built
