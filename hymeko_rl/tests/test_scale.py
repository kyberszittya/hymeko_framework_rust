"""Tests for scaled temporal-form refinement via a coverage P-graph (our hymeko_pgraph)."""
from __future__ import annotations

from hymeko_rl.eval.spec_bench.scale import (
    coverage_pgraph_hymeko,
    refine_scaled,
    synth_conj_temporal,
    synth_single_settle,
    temporal_variants,
)
from hymeko_rl.eval.spec_bench.spec_bench import formula_f1


def test_single_settle_makes_temporal_form_decisive() -> None:
    test = synth_single_settle(120, seed=200)
    assert sum(r.success for r in test) == 60                       # balanced
    # F accepts the touch-then-drift negative (false positives); the late-window G rejects it.
    assert formula_f1("F(obj_to_target <= 0.1)", test) < 0.8
    assert formula_f1("G[0,4](obj_to_target <= 0.1)", test) > 0.95


def test_temporal_variants_pool() -> None:
    v = temporal_variants("obj_to_target")
    assert len(v) == 6                                              # {>=,<=} × {F, G, G[0,4]}
    assert any("G[0,4]" in f for f in v) and any(f.startswith("F(") for f in v)


def test_coverage_pgraph_format() -> None:
    src, umap = coverage_pgraph_hymeko(["in_place", "obj_to_target"])
    assert "success <material, product>;" in src
    assert "in_place_ok <material>;" in src and "obj_to_target_ok <material>;" in src
    assert "@SUCCESS <unit>" in src and "-in_place_ok, -obj_to_target_ok, +success" in src
    assert len(umap) == 12                                          # 2 aspects × 6 variants


def test_scaled_search_selects_late_window_G() -> None:
    # the decisive axis-1 result: given the temporal-variant pool, the search picks G[0,4] and reaches ~ceiling.
    verif, test = synth_single_settle(80, seed=100), synth_single_settle(120, seed=200)
    refined = refine_scaled(["obj_to_target"], verif)
    assert refined is not None and "G[0,4]" in refined
    assert formula_f1(refined, test) > 0.95


def test_conj_temporal_ground_truth_faithful() -> None:
    test = synth_conj_temporal(120, seed=200)
    assert sum(r.success for r in test) == 60
    gt = "(F(in_place >= 0.9) AND G[0,4](obj_to_target <= 0.1))"
    assert formula_f1(gt, test) > 0.9                              # the intended conjunction is faithful here


def test_refine_scaled_no_aspects_returns_none() -> None:
    assert refine_scaled([], synth_single_settle(20, seed=1)) is None
