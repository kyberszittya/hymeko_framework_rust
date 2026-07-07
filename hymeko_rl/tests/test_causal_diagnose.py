"""Frame + prioritizer + orchestrator tests: the CIP diagnostic layer end-to-end (contract guards + perf).

Correctness of the LiNGAM measure itself is covered by ``test_causal_lingam.py``; here we test aggregation,
the continuous-only contract (a categorical routed into the linear model must raise), disagreement ranking,
stratification, the assembled :class:`DiagnosisReport`, and a wall-time performance budget.
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import pytest

from hymeko_rl.eval.causal import (
    CausalDiagnosis,
    CipPrioritizer,
    DirectLiNGAM,
    RolloutFrame,
)
from hymeko_rl.eval.causal.frame import VarKind


def _make_verdicts(n, seed, disagreement=True):
    """n verdict-mappings with varied continuous CIP variables (+ optional reward/monitor disagreement)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        prog = float(rng.uniform(0.0, 1.0))
        dis = float(rng.uniform(0.0, 1.0)) if disagreement else 0.0
        out.append({
            "monitor_pass": bool(prog > 0.5), "monitor_score": prog,
            "approach_score": 0.1, "contact_score": 0.1, "progress_score": prog,
            "delivery_score": 0.5, "anti_exploit_score": 0.1, "violation_reason": "none",
            "sub_verdicts": {"stagnation": {
                "name": "stagnation", "passed": True, "score": 0.1, "violations": [],
                "time_indices": [], "slices": [], "stagnation_duration": int(rng.integers(0, 20)),
                "stagnated": bool(rng.integers(0, 2))}},
            "forbidden_contact_count": int(rng.integers(0, 5)),
            "clearance_min": float(rng.uniform(0.0, 0.2)),
            "reward_progress_disagreement": dis,
        })
    return out


def _frame(n=120, seed=0, disagreement=True):
    verdicts = _make_verdicts(n, seed, disagreement)
    rng = np.random.default_rng(seed + 1)
    cats = {"method": [["bc", "dagger", "td3_bc"][i % 3] for i in range(n)],
            "architecture": [["mlp", "hsikan"][i % 2] for i in range(n)]}
    extra = {"final_distance": [float(rng.uniform(0, 1)) for _ in range(n)]}
    return RolloutFrame.from_verdicts(verdicts, extra_continuous=extra, categoricals=cats)


# -- frame ----------------------------------------------------------------------------------------------------
def test_frame_splits_continuous_and_categorical():
    f = _frame(n=60)
    assert f.n == 60
    # continuous CIP vars present; binary CIP vars are categorical (not in the linear-model columns)
    for name in ("progress_score", "stagnation_duration", "forbidden_contact_count",
                 "clearance_min", "reward_progress_disagreement", "final_distance"):
        assert name in f.continuous
    for name in ("success_monitor_pass", "stagnated", "phase_transition_failure"):
        assert name in f.categorical
    assert "method" in f.categorical and "architecture" in f.categorical


def test_missing_variable_flagged_and_dropped_from_matrix():
    # A fully-absent continuous var (never sourced) must be flagged missing and dropped from the model matrix.
    verdicts = [{"monitor_pass": True, "monitor_score": 0.5, "approach_score": 0.1, "contact_score": 0.1,
                 "progress_score": float(p), "delivery_score": 0.5, "anti_exploit_score": 0.1,
                 "violation_reason": "none", "sub_verdicts": {}} for p in np.linspace(0, 1, 30)]
    f = RolloutFrame.from_verdicts(verdicts)
    # clearance_min was never sourced -> all-missing -> dropped
    assert f.missing["clearance_min"].all()
    matrix, kept, dropped = f.continuous_matrix()
    assert "clearance_min" in dropped and dropped["clearance_min"] == "all_missing"
    assert "progress_score" in kept
    assert matrix.shape[0] == 30


def test_continuous_matrix_raises_on_categorical_name():
    f = _frame(n=40)
    with pytest.raises(ValueError, match="categorical"):
        f.continuous_matrix(["method"])          # a categorical must never reach the linear model


def test_var_kinds_enum_used_not_strings():
    from hymeko_rl.eval.causal.frame import CIP_VAR_KINDS
    assert CIP_VAR_KINDS["progress_score"] is VarKind.CONTINUOUS
    assert CIP_VAR_KINDS["success_monitor_pass"] is VarKind.CATEGORICAL


def test_group_by_and_subset_stratify():
    f = _frame(n=60)
    groups = f.group_by(["architecture"])
    assert set(k[0] for k in groups) == {"mlp", "hsikan"}
    sub = f.subset(groups[("mlp",)])
    assert sub.n == len(groups[("mlp",)])
    assert all(a == "mlp" for a in sub.categorical["architecture"])


# -- prioritizer ----------------------------------------------------------------------------------------------
def test_prioritizer_ranks_disagreement():
    f = _frame(n=100, disagreement=True)
    findings = CipPrioritizer().rank(f)
    assert findings and findings[0].variable == "reward_progress_disagreement"
    assert findings[0].candidate_cause == "reward_farming_candidate"
    assert findings[0].mean_disagreement > 0.0
    assert findings[0].top_rollouts                       # non-empty: highest-disagreement rollouts flagged


def test_prioritizer_rank_rollouts_descending():
    f = _frame(n=50)
    ranked = CipPrioritizer().rank_rollouts(f)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(ranked) == 50


# -- orchestrator ---------------------------------------------------------------------------------------------
def test_diagnosis_report_wellformed():
    f = _frame(n=120, seed=2, disagreement=True)
    report = CausalDiagnosis().run(f)
    kept = report.provenance["continuous_used"]
    assert len(kept) >= 2
    assert sorted(report.causal_order) == sorted(kept)     # order is a permutation of the used columns
    assert "Reward rises without monitor progress" in report.next_intervention
    assert report.ablation_plan and report.ablation_plan[0].startswith("Decider:")
    assert report.provenance["lingam"].startswith("fit:")
    d = report.as_dict()
    assert "PROPOSES structure" in d["_disclaimer"]


def test_diagnosis_selects_none_when_reward_agrees():
    f = _frame(n=80, seed=5, disagreement=False)          # reward_progress_disagreement all zero -> agree
    report = CausalDiagnosis().run(f)
    assert "no dominant disagreement" in report.next_intervention or "agree" in report.next_intervention


def test_run_raises_on_categorical_routed_to_model():
    f = _frame(n=40)
    with pytest.raises(ValueError, match="categorical"):
        CausalDiagnosis().run(f, continuous_names=["method"])


def test_run_stratified_per_architecture():
    f = _frame(n=120, seed=3)
    reports = CausalDiagnosis().run_stratified(f, stratify_by=["architecture"])
    assert set(k[0] for k in reports) == {"mlp", "hsikan"}
    for rep in reports.values():
        assert rep.provenance["n_rollouts"] < 120         # each stratum is a subset


def test_lingam_skipped_gracefully_when_too_few_columns():
    # only one usable continuous column -> LiNGAM cannot run, prioritizer still does; no fabrication
    verdicts = [{"monitor_pass": True, "monitor_score": 0.5, "approach_score": 0.1, "contact_score": 0.1,
                 "progress_score": float(p), "delivery_score": 0.5, "anti_exploit_score": 0.1,
                 "violation_reason": "none", "sub_verdicts": {}} for p in np.linspace(0, 1, 20)]
    f = RolloutFrame.from_verdicts(verdicts)
    report = CausalDiagnosis().run(f, continuous_names=["progress_score"])
    assert report.causal_order == []
    assert "skipped" in report.provenance["lingam"]


# -- performance ----------------------------------------------------------------------------------------------
def test_directlingam_perf_budget():
    rng = np.random.default_rng(0)
    x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(200, 8))
    model = DirectLiNGAM()
    model.fit(x)                                           # warm-up
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.fit(x)
        times.append(time.perf_counter() - t0)
    median = statistics.median(times)
    assert median < 0.15, f"DirectLiNGAM N=200,d=8 median {median*1e3:.1f} ms exceeds 150 ms budget"
