"""Tests for the multi-seed off-policy architecture-eval driver.

Cheap layers (unit + the default-path regression) run on the 2-vertex cart-pole; one integration cell
exercises the real-topology Galambos path at a toy budget. Seeds are fixed; ``pytest -p no:randomly``.
"""
from __future__ import annotations

import math

import pytest

from hymeko_rl.ddpg import OffPolicyConfig, build_offpolicy, train_offpolicy
from hymeko_rl.env.inverted_pendulum_env import InvertedPendulumEnv, emit_cartpole_mjcf
from hymeko_rl.offpolicy_eval import (
    Budget,
    aggregate_records,
    compare_offpolicy,
    greedy_return_eval,
)
from hymeko_rl.offpolicy_tables import to_latex, to_markdown
from hymeko_rl.sac import SACConfig, build_sac, train_sac


def _rec(task: str, algo: str, backbone: str, seed: int, curve_max: float, params: int) -> dict:
    return dict(key=f"{task}/{algo}/{backbone}/{seed}", task=task, algo=algo, backbone=backbone,
                seed=seed, curve=[curve_max], curve_max=curve_max, final=curve_max, n_params=params)


def _cartpole_sac() -> tuple[InvertedPendulumEnv, object, list[object]]:
    mj = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=mj)
    space = env.observation_space.shape
    assert space is not None
    nv, feat = int(space[0]), int(space[1])
    actor, critics = build_sac("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                               action_scale=env.force_mag, n_critics=2, hidden=16)
    return env, actor, critics


# ---- unit -------------------------------------------------------------------------------------------
def test_greedy_return_eval_is_finite() -> None:
    env, actor, _ = _cartpole_sac()
    val = greedy_return_eval(n_eval=2)(env, actor)
    assert isinstance(val, float)
    assert math.isfinite(val)


def test_compare_offpolicy_structure_and_params() -> None:
    budget = Budget(total_steps=300, eval_every=150, n_eval=2, hidden={"mlp": 16, "hsikan": 16})
    rep = compare_offpolicy("cartpole", algos=["sac"], backbones=["mlp"], seeds=[0], budget=budget)
    assert set(rep) == {"sac/mlp"}
    cell = rep["sac/mlp"]
    assert cell["n_params"] > 0
    assert len(cell["per_seed_max"]) == 1
    assert "curve_max_median" in cell and math.isfinite(cell["curve_max_median"])


def test_unknown_task_and_backbone_raise() -> None:
    budget = Budget(total_steps=10, eval_every=10, n_eval=1, hidden={"mlp": 8})
    with pytest.raises(ValueError, match="unknown task"):
        compare_offpolicy("nope", algos=["sac"], backbones=["mlp"], seeds=[0], budget=budget)
    with pytest.raises(ValueError, match="unknown backbone"):
        compare_offpolicy("cartpole", algos=["sac"], backbones=["bogus"], seeds=[0], budget=budget)


# ---- regression: the default (eval_fn=None) path is byte-for-byte the cart-pole upright-steps curve --
def test_train_sac_default_eval_is_upright_steps() -> None:
    env, actor, critics = _cartpole_sac()
    cfg = SACConfig(total_steps=300, start_steps=50, batch_size=32, eval_every=150, n_eval=2, seed=0)
    curve = train_sac(actor, critics, env, cfg)   # eval_fn=None -> eval_balance
    assert len(curve) == 2
    assert all(0.0 <= v <= float(env.max_steps) for v in curve)


def test_train_offpolicy_default_eval_is_upright_steps() -> None:
    mj = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=mj)
    space = env.observation_space.shape
    assert space is not None
    nv, feat = int(space[0]), int(space[1])
    actor, critics = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                     action_scale=env.force_mag, n_critics=1, hidden=16)
    cfg = OffPolicyConfig(total_steps=300, start_steps=50, batch_size=32, eval_every=150, n_eval=2, seed=0)
    curve = train_offpolicy(actor, critics, env, cfg)
    assert len(curve) == 2
    assert all(0.0 <= v <= float(env.max_steps) for v in curve)


# ---- regression: the injected eval_fn replaces the curve metric --------------------------------------
def test_injected_eval_fn_is_used() -> None:
    env, actor, critics = _cartpole_sac()
    cfg = SACConfig(total_steps=300, start_steps=50, batch_size=32, eval_every=150, n_eval=2, seed=0)
    sentinel = -123.5

    def const_eval(_env: object, _actor: object) -> float:
        return sentinel

    curve = train_sac(actor, critics, env, cfg, eval_fn=const_eval)
    assert curve == [sentinel, sentinel]


# ---- regression: the off-policy MLP backbone now honours `hidden` (params-matchable) ----------------
def test_mlp_backbone_width_is_honoured() -> None:
    mj = emit_cartpole_mjcf()
    env = InvertedPendulumEnv(mjcf=mj)
    space = env.observation_space.shape
    assert space is not None
    nv, feat = int(space[0]), int(space[1])

    def n_params(hidden: int) -> int:
        actor, _ = build_offpolicy("mlp", obs_dim=feat, flat_dim=nv * feat, action_dim=1,
                                   action_scale=env.force_mag, n_critics=1, hidden=hidden)
        return int(sum(p.numel() for p in actor.parameters()))

    assert n_params(128) > n_params(32)   # would have been equal before the _backbone fix


# ---- aggregation + tables (synthetic records, no training) ------------------------------------------
def test_aggregate_records_groups_and_stats() -> None:
    recs = [_rec("galambos", "sac", "hsikan", s, m, 14728)
            for s, m in zip(range(3), (10.0, 12.0, 14.0))]
    agg = aggregate_records(recs)
    g = agg[("galambos", "sac", "hsikan")]
    assert g["n_seeds"] == 3 and g["n_params"] == 14728
    assert g["curve_max_median"] == 12.0 and g["curve_max_worst"] == 10.0 and g["curve_max_best"] == 14.0


def test_tables_verdict_win_and_tie() -> None:
    # HSiKAN clearly above MLP with zero spread -> a win; equal medians -> a tie.
    win = ([_rec("quadruped", "sac", "hsikan", s, 50.0, 14096) for s in range(3)]
           + [_rec("quadruped", "sac", "mlp", s, 10.0, 14092) for s in range(3)])
    md = to_markdown(win)
    assert "HSiKAN +40.00" in md
    assert r"\begin{tabular}" in to_latex(win)
    tie = ([_rec("cartpole", "sac", "hsikan", s, 5.0 + s, 13186) for s in range(3)]
           + [_rec("cartpole", "sac", "mlp", s, 5.0 + s, 12982) for s in range(3)])
    assert "tie (" in to_markdown(tie)


# ---- resume: a journalled cell is skipped on rerun --------------------------------------------------
def test_compare_offpolicy_resumes_from_journal(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json as _json
    journal = tmp_path / "cells.jsonl"
    journal.write_text(_json.dumps(_rec("cartpole", "sac", "mlp", 0, 99.0, 12982)) + "\n", encoding="utf-8")
    budget = Budget(total_steps=10, eval_every=10, n_eval=1, hidden={"mlp": 16, "hsikan": 16})
    rep = compare_offpolicy("cartpole", algos=["sac"], backbones=["mlp"], seeds=[0],
                            budget=budget, journal=journal)
    # the pre-seeded cell is reused verbatim (curve_max 99.0), NOT retrained.
    assert rep["sac/mlp"]["curve_max_median"] == 99.0


# ---- integration: the real-topology Galambos cell runs end to end at a toy budget --------------------
def test_galambos_cell_smoke() -> None:
    budget = Budget(total_steps=400, eval_every=200, n_eval=2, hidden={"hsikan": 16, "mlp": 16})
    rep = compare_offpolicy("galambos", algos=["sac"], backbones=["hsikan"], seeds=[0], budget=budget)
    cell = rep["sac/hsikan"]
    assert cell["n_params"] > 0
    assert math.isfinite(cell["curve_max_median"])
    assert len(cell["per_seed_final"]) == 1
