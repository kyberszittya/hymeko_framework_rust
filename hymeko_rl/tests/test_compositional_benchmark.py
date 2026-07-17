"""Tests for the compositional-spec benchmark: the non-trivial K-conjunction + D-distractor generator, the greedy
conjunct-selection baseline, and the sweep runner. The point the benchmark makes — genuine K-compositionality
(no single signal, and no (K-1)-subset, separates success) — is asserted directly, since it is exactly what
MetaWorld's single-proxy interface cannot provide."""
from __future__ import annotations

import numpy as np
import pytest

from hymeko_rl.eval.spec_bench.pgraph_refine import greedy_conjunct_select, refine_via_pgraph
from hymeko_rl.eval.spec_bench.scale import (
    compositional_ground_truth,
    compositional_raw_spec,
    synth_compositional,
    synth_compositional_lure,
)
from hymeko_rl.eval.spec_bench.spec_bench import formula_f1
from hymeko_rl.eval.spec_bench.spec_reward import spec_reward_separation


def test_synth_compositional_validates() -> None:
    with pytest.raises(ValueError):
        synth_compositional(0, 3, 40, seed=0)              # k_true < 1
    with pytest.raises(ValueError):
        synth_compositional(2, 3, 2, seed=0)               # n < 4


def test_synth_compositional_balanced_and_deterministic() -> None:
    a = synth_compositional(3, 4, 60, seed=1)
    b = synth_compositional(3, 4, 60, seed=1)
    assert len(a) == 60 and 20 <= sum(r.success for r in a) <= 40      # roughly balanced
    assert [r.success for r in a] == [r.success for r in b]            # deterministic in seed
    assert a[0].trace[0].keys() == b[0].trace[0].keys()


def test_non_trivial_no_single_signal_separates() -> None:
    """The regime MetaWorld cannot provide: for K>=2 no single-signal spec separates success (AUC << 1)."""
    test = synth_compositional(3, 5, 120, seed=2)
    signals = [f"true_{i}" for i in range(3)] + [f"dist_{j}" for j in range(5)]
    best_single = max(spec_reward_separation(f"F({s} >= 0.9)", test).auc for s in signals)
    assert best_single < 0.8                                          # non-trivial by construction


def test_compositional_necessity_all_k_conjuncts_required() -> None:
    """A (K-1)-subset spec must under-perform the full K-conjunct ceiling (each true conjunct is necessary)."""
    verif = synth_compositional(3, 2, 160, seed=3)
    full = formula_f1(compositional_ground_truth(3), verif)
    km1 = formula_f1("F(true_0 >= 0.9 AND true_1 >= 0.9)", verif)     # drops true_2
    assert full > km1 + 0.05 and full >= 0.9


def test_arbiter_recovers_true_spec_from_distractor_pool() -> None:
    """The arbiter (pgraph SSG conjunct-pruning) lifts the raw over-constrained spec back to ~ceiling."""
    verif = synth_compositional(3, 4, 140, seed=4)
    test = synth_compositional(3, 4, 140, seed=5)
    raw = compositional_raw_spec(3, 4)
    assert formula_f1(raw, test) < 0.6                               # over-constrained (distractors reject positives)
    assert formula_f1(refine_via_pgraph(raw, verif), test) >= 0.85   # pruned back toward ceiling


def test_greedy_baseline_recovers_and_ties_or_loses_to_pgraph() -> None:
    """Greedy recovers the conjunction too (honest: on pure conjunctions the pgraph SSG does not beat it on F1)."""
    verif = synth_compositional(3, 3, 140, seed=6)
    test = synth_compositional(3, 3, 140, seed=7)
    raw = compositional_raw_spec(3, 3)
    greedy_f1 = formula_f1(greedy_conjunct_select(raw, verif), test)
    pgraph_f1 = formula_f1(refine_via_pgraph(raw, verif), test)
    assert greedy_f1 >= 0.85                                          # greedy recovers
    assert pgraph_f1 >= greedy_f1 - 0.02                              # pgraph ties or beats (never much worse)


def test_calibration_escape_greedy_matches_ssg_on_lure() -> None:
    """The unification's honest core: a luring distractor is the best single predictor (so greedy picks it first),
    yet greedy ties the SSG — because threshold *calibration* neutralises the lure conjunct (drives it vacuous).
    So the SSG has NO accuracy win over greedy in the spec-conjunction language; its accuracy advantage is reserved
    for non-neutralisable interaction (causal joint mechanisms — SignedHyperLiNGAM)."""
    verif = synth_compositional_lure(2, 140, seed=0)
    test = synth_compositional_lure(2, 140, seed=50)
    signals = ["true_0", "true_1", "lure"]
    best_single = max(signals, key=lambda s: spec_reward_separation(f"F({s} >= 0.9)", test).auc)
    assert best_single == "lure"                                     # the lure DOES lure greedy (best single)
    raw = "F(true_0 >= 0.9 AND true_1 >= 0.9 AND lure >= 0.9)"
    greedy_f1 = formula_f1(greedy_conjunct_select(raw, verif), test)
    pgraph_f1 = formula_f1(refine_via_pgraph(raw, verif), test)
    assert abs(greedy_f1 - pgraph_f1) < 0.06                         # calibration escape: greedy ties the SSG
    assert greedy_f1 >= 0.88                                         # greedy still reaches ~ceiling despite the lure


def test_greedy_single_predicate_just_calibrates() -> None:
    verif = synth_compositional(2, 0, 60, seed=8)
    out = greedy_conjunct_select("F(true_0 >= 0.5)", verif)
    assert "true_0" in out and out.startswith("F(")                  # single pred → calibrated, structure intact


def test_ground_truth_and_raw_spec_formats() -> None:
    assert compositional_ground_truth(1) == "F(true_0 >= 0.9)"
    assert compositional_ground_truth(3).count("AND") == 2
    raw = compositional_raw_spec(2, 3)
    assert raw.count("true_") == 2 and raw.count("dist_") == 3        # all K+D candidates


def test_run_benchmark_smoke(tmp_path: object) -> None:
    from pathlib import Path

    from hymeko_rl.experiments.exp_compositional_spec_benchmark import run_benchmark
    s = run_benchmark(k_true=3, d_values=(2,), n=80, out_dir=Path(str(tmp_path)))
    cell = s["cells"][0]
    assert {"best_single_auc", "raw_f1", "greedy_f1", "pgraph_f1", "ceiling_f1", "abb_explored"} <= set(cell)
    assert s["non_trivial_all_cells"] and s["arbiter_recovers_ceiling"]
    assert (Path(str(tmp_path)) / "compositional_spec_benchmark.json").exists()
    assert np.isfinite(cell["best_single_auc"])
