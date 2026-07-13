"""Tests for reducing spec conjunct-pruning to a hymeko_pgraph SSG solution."""
from __future__ import annotations

from hymeko_rl.eval.spec_bench.pgraph_refine import (
    _PGRAPH_BIN,
    _predicates,
    _subsets_fallback,
    predicates_to_pgraph_hymeko,
    refine_via_pgraph,
    solve_ssg,
)
from hymeko_rl.eval.spec_bench.spec_bench import formula_f1, synth_rollouts

_BAD = "F(in_place >= 0.9 AND obj_to_target <= 0.01 AND near_object >= 0.7 AND grasp_success == 1)"


def test_pgraph_hymeko_reduction_format() -> None:
    src = predicates_to_pgraph_hymeko(_predicates(_BAD))
    assert "success <material, product>;" in src
    assert "in_place <material, raw>;" in src and "grasp_success <material, raw>;" in src
    assert src.count("<unit>") == 4                      # one operating unit per candidate predicate
    assert "(-in_place, +success)" in src


def test_subsets_fallback_count() -> None:
    assert len(_subsets_fallback(4)) == 15               # 2^4 - 1 non-empty subsets
    assert len(_subsets_fallback(1)) == 1


def test_ssg_solve_returns_structures_when_binary_present() -> None:
    structs = solve_ssg(predicates_to_pgraph_hymeko(_predicates(_BAD)))
    if _PGRAPH_BIN.exists():
        assert structs is not None and len(structs) >= 1  # our crate enumerated feasible structures
    else:
        assert structs is None                            # graceful when unbuilt (refine falls back)


def test_refine_prunes_noise_conjunct_and_lifts_f1() -> None:
    verif, test = synth_rollouts(40, seed=100), synth_rollouts(80, seed=200)
    refined = refine_via_pgraph(_BAD, verif)
    assert formula_f1(_BAD, test) < 0.1                  # the over-constrained original is dead
    assert formula_f1(refined, test) > 0.85              # pruning + calibration lifts it to ~ceiling
    assert "grasp_success" not in refined                # the noise conjunct is dropped


def test_refine_single_predicate_calibrates_only() -> None:
    verif = synth_rollouts(30, seed=1)
    refined = refine_via_pgraph("F(in_place >= 0.3)", verif)
    assert "in_place" in refined and refined.startswith("F(")   # structure kept, threshold calibrated


def test_refine_never_lowers_below_calibrated_original() -> None:
    # pruning is monotone: it appends the calibrated original as a candidate, so it can only help or tie.
    verif, test = synth_rollouts(40, seed=5), synth_rollouts(80, seed=6)
    good = "F(in_place >= 0.5 AND obj_to_target <= 0.5)"
    refined = refine_via_pgraph(good, verif)
    assert formula_f1(refined, test) >= formula_f1(good, test) - 1e-9
