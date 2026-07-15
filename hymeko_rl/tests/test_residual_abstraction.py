"""Regression guard for the residual-RL abstraction (F-PP-009, 2026-07-15 SAC-residual abstraction audit).

Two HARD invariants the audit established. If either breaks, a residual-RL collapse is an INTERFACE bug (halt +
audit the abstraction), NOT an RL-difficulty story — this is the gate that keeps the next agent from mis-attributing
a 0.458 collapse to "RL fails" / the scenario, the exact failure the audit corrected (F-PP-008 -> F-PP-009):

  1. zero-residual reproduces the frozen base  — the residual chain is clip(base + delta*r, lo, hi); r=0 -> base.
  2. the reactive teacher target is NOT weaker than the base — clip(base + delta*clip((expert-base)/delta))
     executed closed-loop must be >= base, else anchoring to it would drag the base DOWN.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BASE = Path("experiments/hybrid_dagger_gif/policies/hybrid_dagger_hsikan_s0_best.pt")
_N = 6                                  # small but horizon-full (max_steps unchanged) — deterministic, non-flaky


def _cfg_base():
    from hymeko_rl.experiments.pick_place_residual_rl import ResidualCfg, _load_base
    return ResidualCfg(delta=0.25, reward_mode="settle"), _load_base("hsikan")


@pytest.mark.skipif(not _BASE.exists(), reason="deployed base checkpoint not present")
def test_zero_residual_reproduces_base():
    """HARD GATE: a zero residual runs the frozen base. If this drops, base+residual plumbing is broken."""
    from hymeko_rl.experiments.gripper_pick_bc import eval_success
    from hymeko_rl.experiments.pick_place_residual_rl import _renv, _ZeroResidual
    cfg, base = _cfg_base()
    score = eval_success(_renv(base, cfg), _ZeroResidual(), _N, seed=20_000)[1]
    assert score > 0.5, f"zero-residual score {score:.3f} — residual chain no longer reproduces the base (INTERFACE bug)"


@pytest.mark.skipif(not _BASE.exists(), reason="deployed base checkpoint not present")
def test_reactive_teacher_target_not_weaker_than_base():
    """The anchor TARGET, executed reactively (recomputed each step), must be >= base. Measured 1.000 vs base 0.875;
    if it drops below base the anchor would drag the base down and no anchor strength could help (F-PP-009)."""
    from hymeko_rl.experiments.gripper_pick_bc import eval_success
    from hymeko_rl.experiments.pick_place_residual_rl import PickResidualExpertTeacher, _renv, _ZeroResidual
    cfg, base = _cfg_base()
    base_score = eval_success(_renv(base, cfg), _ZeroResidual(), _N, seed=20_000)[1]
    teacher = PickResidualExpertTeacher()
    wins = 0
    for s in range(_N):
        renv = _renv(base, cfg)
        renv.reset(seed=20_000 + s)
        info: dict = {}
        for _ in range(renv.max_steps):
            _o, _rw, term, trunc, info = renv.step(teacher.action(renv))
            if term or trunc:
                break
        wins += int(bool(info.get("reached")))
    reactive = wins / _N
    assert reactive >= base_score, (
        f"reactive-teacher target {reactive:.3f} < base {base_score:.3f}: the anchor target is weaker than the base "
        "— anchoring would drag it down (abstraction/target regression, F-PP-009)")
