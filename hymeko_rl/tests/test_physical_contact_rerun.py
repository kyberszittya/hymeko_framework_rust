"""Unit coverage for the physical-contact SAC/TD3 rerun driver + evaluator (2026-07-22).

Covers the pure/decidable units: the BC zero-residual init (both SAC ``mu`` and TD3 ``head`` architectures reduce to
zero residual → behaviour-equivalent init) and the §11 verdict rule (decided on NATIVE VAL, strict is context only).
The training orchestration (``run_one``) is exercised end-to-end by the wiring smokes.
"""
from __future__ import annotations

import numpy as np
import torch

from hymeko_rl.experiments.coin_physical_contact_eval import _verdict
from hymeko_rl.experiments.coin_physical_contact_rerun import bc_init_zero_residual


def test_bc_init_zeroes_both_architectures_to_identical_residual():
    from hymeko_rl.train.ddpg import build_offpolicy
    from hymeko_rl.train.sac import build_sac
    sac, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    td3, _ = build_offpolicy("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0, n_critics=2)
    assert bc_init_zero_residual(sac) == "mu"
    assert bc_init_zero_residual(td3) == "head"
    probe = torch.as_tensor(np.random.default_rng(0).standard_normal((64, 41)).astype(np.float32))
    with torch.no_grad():
        ys, yt = sac.action_mean(probe), td3.action_mean(probe)
    assert float(ys.abs().max()) == 0.0 and float(yt.abs().max()) == 0.0      # both output zero residual
    assert float((ys - yt).abs().max()) == 0.0                                # identical init (max Δ = 0)


def test_bc_init_raises_on_headless_actor():
    class _NoHead:
        pass
    try:
        bc_init_zero_residual(_NoHead())
    except AttributeError:
        return
    raise AssertionError("bc_init_zero_residual must raise on an actor with no known head (never a silent no-op)")


def _dists(val_native, val_strict=0):
    return {"VAL": {"native_zone": val_native, "strict_count": val_strict},
            "panel": {"native_zone": 1.0, "strict_count": 6},
            "heldout": {"native_zone": 0.7, "strict_count": 0}}


def test_verdict_positive_noeffect_regression_invalid():
    scripted = _dists(0.30)
    assert _verdict(scripted, _dists(0.40)) == "PHYSICS_FIXED_POSITIVE"   # +0.10 > eps
    assert _verdict(scripted, _dists(0.31)) == "NO_EFFECT"                # within eps
    assert _verdict(scripted, _dists(0.20)) == "REGRESSION"              # -0.10 < -eps
    assert _verdict(scripted, _dists(float("nan"))) == "RUN_INVALID"     # NaN guard
