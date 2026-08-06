"""Evaluation-horizon contract (2026-07-22).

The shared evaluator must roll the DECLARED environment horizon (``env.cfg.horizon``), never a hard-coded
truncation. A shorter probe is a time-to-success *diagnostic* only — it must never be the default, must never drive
checkpoint selection or headline reporting, and must be reportable as a DISTINCT view of the task (the 60-vs-120
horizon artifact of 2026-07-22). These tests pin that contract.
"""
from __future__ import annotations

from hymeko_rl.experiments.coin_physical_contact_rerun import bc_init_zero_residual
from hymeko_rl.experiments.coin_two_arm_sac import direct_env, evaluate
from hymeko_rl.train.sac import build_sac

_SEEDS = tuple(range(64100, 64104))


def _scripted_actor():
    """A BC zero-residual actor: its residual is 0 so the env's grasp_carry base drives — i.e. the scripted expert,
    which delivers only near the full 120-step horizon (the slow base that exposed the truncation)."""
    ac, _ = build_sac("mlp", obs_dim=41, flat_dim=41, action_dim=6, action_scale=1.0)
    bc_init_zero_residual(ac)
    ac.eval()
    return ac


def test_default_eval_horizon_equals_declared_env_horizon():
    """The default (max_steps=None) must roll the DECLARED horizon — identical discrete metrics to an explicit
    env.cfg.horizon call (auxiliary floats may differ by BLAS non-determinism; the task-defining counts must not)."""
    env = direct_env()
    ac = _scripted_actor()
    assert env.cfg.horizon != 60, "declared horizon must not silently equal the old truncation constant"
    default = evaluate(env, ac, _SEEDS)
    matched = evaluate(env, ac, _SEEDS, max_steps=env.cfg.horizon)
    assert default["strict_count"] == matched["strict_count"], "default horizon must equal the declared env horizon"
    assert default["zone_rate"] == matched["zone_rate"], "default horizon must equal the declared env horizon"


def test_selection_horizon_equals_reporting_horizon():
    """Selection and final reporting both go through evaluate()'s default → the same declared horizon by construction;
    two fresh envs declare the identical horizon (deterministic contract), and it is NOT the old 60."""
    a, b = direct_env().cfg.horizon, direct_env().cfg.horizon
    assert a == b and a != 60


def test_short_probe_never_reports_more_success_than_the_full_horizon():
    """A shorter probe is a time-to-success diagnostic; by construction it can never report MORE strict success than
    the deployment horizon (no probe silently inflates / redefines the task upward)."""
    env = direct_env()
    ac = _scripted_actor()
    full = evaluate(env, ac, _SEEDS)                      # deployment horizon (declared)
    for k in (30, 60, 90):
        probe = evaluate(env, ac, _SEEDS, max_steps=k)   # diagnostic only
        assert probe["strict_count"] <= full["strict_count"], \
            f"{k}-step probe reported MORE strict success ({probe['strict_count']}) than the full horizon"


def test_success_by_time_is_monotone_nondecreasing_in_horizon():
    """Success-by-time (the legitimate temporal diagnostic) can only grow with more steps — a policy that has
    succeeded by step k has succeeded by step k' > k."""
    env = direct_env()
    ac = _scripted_actor()
    counts = [evaluate(env, ac, _SEEDS, max_steps=k)["strict_count"] for k in (30, 60, 90, 120)]
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)), \
        f"strict success-by-time must be non-decreasing in the horizon, got {counts}"
