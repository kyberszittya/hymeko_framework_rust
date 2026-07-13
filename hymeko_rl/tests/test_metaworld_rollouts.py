"""Tests for spec_bench MetaWorld rollout IO + run_bench (roll_coffee_push needs metaworld → exercised on kato15)."""
from __future__ import annotations

from hymeko_rl.eval.spec_bench.metaworld_rollouts import load_rollouts, run_bench, save_rollouts
from hymeko_rl.eval.spec_bench.spec_bench import Rollout, ScriptedModel, synth_rollouts


def test_save_load_roundtrip(tmp_path) -> None:
    rolls = [Rollout(trace=[{"in_place": 0.9, "obj_to_target": 0.05}], success=True),
             Rollout(trace=[{"in_place": 0.2, "obj_to_target": 0.4}], success=False)]
    p = save_rollouts(rolls, tmp_path / "r.json")
    back = load_rollouts(p)
    assert len(back) == 2
    assert back[0].trace == rolls[0].trace and back[0].success is True and back[1].success is False


def test_run_bench_shape_and_formal_ceiling() -> None:
    verif, test = synth_rollouts(40, seed=100), synth_rollouts(60, seed=200)
    # a scripted model that emits the faithful structure (raw), so the bench has a real row.
    models = {"scripted": ScriptedModel(replies=["F(in_place >= 0.9)"] * 8)}
    rep = run_bench(models, verif, test, prompt="p", system="s", formal="F(in_place >= 0.9)", k=2)
    assert rep["formal_f1"] > 0.85
    assert rep["n_test"] == 60 and rep["test_balance"] == 30
    row = rep["models"][0]
    assert row["model"] == "scripted" and row["gate_f1"] > 0.85
