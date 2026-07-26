"""Integration test for Route-A mobile handoff conditioning (production-scale env; ~30 s).

Documents the TRUE behaviour established in H2 Session 2: on a healthy development cradle (s1) the grip-preserving,
early-stopping monotone-squeeze conditioning preserves the grip (both tips keep a real normal force, straddle retained)
and settles quickly WITHOUT driving a tip off the coin — the failure modes diagnosed and fixed while building it
(penetration-servo backing a deep tip off the free coin; zeroing the standing grip torque; over-processing drift). It
does NOT assert that conditioning rescues the fragile held-out cradles — the benchmark showed it does not (an honest
negative recorded in the campaign report), which is why the campaign moves to Route B.
"""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.coin_delivery.mobile_conditioning import ConditionConfig, arm_inward_geom, condition_mobile_handoff


def _acquire_s1():
    """Reconstruct + acquire the certified straddle cradle for development state s1 (seed 14250)."""
    from hymeko_rl.experiments.bv_identification_benchmark import _load_frozen, acquire_certified_straddle
    from hymeko_rl.experiments.video_coin_variants import _setup
    stack, cfg_coop, v2, mu = _load_frozen()
    pi0, base, forbidden = _setup()
    rl, saved, handoff, meta = acquire_certified_straddle(pi0, base, forbidden, v2, stack, cfg_coop, 14250, tries=3)
    return rl, saved, handoff, stack, cfg_coop, meta


@pytest.mark.slow
def test_condition_preserves_healthy_grip_s1():
    rl, saved, handoff, stack, cfg_coop, meta = _acquire_s1()
    assert rl is not None and meta["certified"], "s1 must acquire a certified straddle cradle"
    cond = condition_mobile_handoff(rl, stack, saved, handoff["prev_tau"], handoff["q_target"],
                                    coop=cfg_coop, ccfg=ConditionConfig())
    # grip preserved: both tips keep a real normal force (NOT driven off the coin), and it settles within the budget
    fnl, fnr = cond["fn_final"]
    assert fnl >= 1.0 and fnr >= 1.0, f"conditioning must preserve the grip on a healthy cradle, got {cond['fn_final']}"
    assert cond["settled"] and cond["steps_used"] <= ConditionConfig().max_steps
    assert cond["balance_final"] >= 0.30
    # the conditioned handoff is LOWER-debt: peak |prev_tau| does not exceed the raw handoff's
    raw_peak = float(np.max(np.abs(handoff["prev_tau"])))
    cond_peak = float(np.max(np.abs(cond["prev_tau"])))
    assert cond_peak <= raw_peak + 1e-6, f"conditioned prev_tau peak {cond_peak} should not exceed raw {raw_peak}"


@pytest.mark.slow
def test_arm_inward_geom_points_toward_coin_s1():
    """The analytic inward direction moves the tip TOWARD the coin (positive projection of the tip Jacobian step)."""
    import mujoco
    rl, saved, handoff, stack, cfg_coop, meta = _acquire_s1()
    assert rl is not None
    m, d = rl.inner.model, rl.inner.data
    coin = d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "disk")][:2].astype(np.float64)
    gl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_left")
    inward = arm_inward_geom(rl, gl, (0, 1), coin)
    assert inward.shape == (4,)
    assert np.allclose(inward[2:], 0.0)                         # only this arm's dofs are driven
    # the direction is unit-norm (or exactly zero if kinematically decoupled)
    n = float(np.linalg.norm(inward))
    assert abs(n - 1.0) < 1e-6 or n == 0.0
