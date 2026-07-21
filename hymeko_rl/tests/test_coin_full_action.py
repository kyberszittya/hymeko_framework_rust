"""Standalone full-action env contract (2026-07-22): u_exec = clip(policy(obs)), no scripted base, no prefix."""
from __future__ import annotations

import numpy as np

from hymeko_rl.train.coin_delivery_rl import p_grasp_carry
from hymeko_rl.train.coin_full_action import eval_full_action, make_full_action_env, rollout_expert


def test_zero_action_does_not_deliver_base_is_disabled():
    """A null policy must NOT deliver — proving no scripted base contributes an online command (unlike the residual
    env, where zero residual = the scripted grasp_carry base delivers)."""
    env = make_full_action_env(fingertip_geometry="POINT", horizon=160)
    r = eval_full_action(lambda _o: np.zeros(6, np.float32), (1011, 1045, 1164), env)
    assert r["strict_count"] == 0, "zero-action policy strictly delivered — a scripted base is leaking in"


def test_scripted_expert_as_full_action_delivers():
    """The scripted grasp_carry expert, applied as the FULL action from the transport-prepared start, delivers to
    center on most panel seeds (the competence BC is trained to clone)."""
    env = make_full_action_env(fingertip_geometry="POINT", horizon=160)
    res = [rollout_expert(env, s, record=False) for s in (1011, 1045, 1174, 1278, 1447)]
    assert sum(r["center"] for r in res) >= 4, "scripted full-action expert failed to deliver on the panel"


def test_step_applies_full_action_not_residual():
    """Executing action a from a reset state must move the coin (the action is applied directly, not as a small
    residual around a base)."""
    env = make_full_action_env(fingertip_geometry="POINT", horizon=160)
    env.reset(seed=1011)
    a = np.asarray(p_grasp_carry(env.inner, 0), np.float32)
    dz0 = float(env.inner._planar_metrics.disk_to_zone)
    for _ in range(40):
        env.step(a)
    dz1 = float(env.inner._planar_metrics.disk_to_zone)
    assert dz1 != dz0, "the full action produced no coin motion (step is not applying it)"


def test_eval_reports_temporal_diagnostics():
    env = make_full_action_env(fingertip_geometry="POINT", horizon=160)
    r = eval_full_action(lambda _o: np.asarray(p_grasp_carry(env.inner, env._suffix_t), np.float32),
                         (1011, 1045), env)
    for key in ("center_rate", "strict_count", "success_by_time", "tts_median", "success_curve_auc", "return_median"):
        assert key in r, f"eval_full_action missing temporal/diagnostic key {key}"
    assert set(r["success_by_time"]) >= {30, 60, 90, 120}
