"""Unit tests for the CIP-export bridge (monitor verdict → named CIP scalar variables).

Covers both input shapes (a live ``TaskVerdict`` and its ``as_dict()`` mapping), the missing-vs-default policy
(no fabricated measurements), the reward/monitor disagreement path via an existing ``RewardConsistencyMonitor``
output, determinism of defaults, and read-only handling of the input verdict.
"""
from __future__ import annotations

import copy

import pytest

from hymeko_rl.eval.task_monitor import (
    CIP_DEFAULTS,
    CIP_VARIABLES,
    RewardConsistencyMonitor,
    StagnationVerdict,
    TaskVerdict,
    export_cip_variables,
)


def verdict_dict(**overrides):
    """A TaskVerdict.as_dict()-shaped mapping; overrides inject optional keys (stagnation, reward_consistency, …)."""
    d = {
        "monitor_pass": True,
        "monitor_score": 0.5,
        "approach_score": 0.1,
        "contact_score": 0.1,
        "progress_score": 0.42,
        "delivery_score": 0.9,
        "anti_exploit_score": 0.1,
        "violation_reason": "none",
        "sub_verdicts": {},
    }
    d.update(overrides)
    return d


def _stagnation_subverdict(duration=7, stagnated=True):
    return {"name": "stagnation", "passed": not stagnated, "score": -0.4 if stagnated else 0.4,
            "violations": ["no_progress_window (...)"] if stagnated else [], "time_indices": [], "slices": [],
            "stagnation_duration": duration, "stagnated": stagnated}


def test_none_input_raises():
    with pytest.raises(TypeError):
        export_cip_variables(None)


def test_export_from_plain_dict():
    exp = export_cip_variables(verdict_dict(progress_score=0.42))
    assert set(exp.variables) == set(CIP_VARIABLES)          # every CIP variable is always present
    assert exp.variables["success_monitor_pass"] is True
    assert exp.variables["progress_score"] == 0.42
    assert "monitor_pass" in exp.source_keys and "progress_score" in exp.source_keys


def test_export_from_task_verdict_object():
    tv = TaskVerdict(
        monitor_pass=True, monitor_score=0.5, approach_score=0.1, contact_score=0.1,
        progress_score=0.42, delivery_score=0.9, anti_exploit_score=0.1, violation_reason="none",
        sub_verdicts={"stagnation": StagnationVerdict("stagnation", True, 0.3, [], [], [],
                                                      stagnation_duration=5, stagnated=False)})
    exp = export_cip_variables(tv)   # object path: getattr on TaskVerdict + on the StagnationVerdict object
    assert exp.variables["success_monitor_pass"] is True
    assert exp.variables["progress_score"] == 0.42
    assert exp.variables["stagnation_duration"] == 5 and exp.variables["stagnated"] is False


def test_stagnation_exported_when_present():
    exp = export_cip_variables(verdict_dict(sub_verdicts={"stagnation": _stagnation_subverdict(7, True)}))
    assert exp.variables["stagnation_duration"] == 7 and exp.variables["stagnated"] is True
    assert "stagnation_duration" not in exp.missing and "stagnated" not in exp.missing


def test_missing_variables_reported_not_fabricated():
    exp = export_cip_variables(verdict_dict())
    for name in ("stagnation_duration", "stagnated", "forbidden_contact_count",
                 "clearance_min", "phase_transition_failure", "reward_progress_disagreement"):
        assert name in exp.missing                    # reported as missing …
        assert exp.variables[name] == CIP_DEFAULTS[name]   # … and set to the stable default, not invented
    assert "success_monitor_pass" not in exp.missing and "progress_score" not in exp.missing


def test_reward_disagreement_from_precomputed_scalar():
    exp = export_cip_variables(verdict_dict(reward_progress_disagreement=0.5))
    assert exp.variables["reward_progress_disagreement"] == 0.5
    assert "reward_progress_disagreement" not in exp.missing
    assert "reward_progress_disagreement" in exp.source_keys


def test_reward_disagreement_from_reward_consistency_monitor():
    # inverted pair: reward prefers 'a', monitor prefers 'b' → RewardConsistencyMonitor flags disagreement
    rc = RewardConsistencyMonitor().check_reward_alignment(
        [{"policy": "a", "total_reward": -100, "monitor_score": 0.1},
         {"policy": "b", "total_reward": -200, "monitor_score": 0.5}])
    assert not rc.passed
    exp = export_cip_variables(verdict_dict(reward_consistency=rc.as_dict()))
    assert exp.variables["reward_progress_disagreement"] > 0.0   # 1 - concordance
    assert "reward_progress_disagreement" not in exp.missing
    assert "reward_consistency.score" in exp.source_keys
    # aligned case → zero disagreement, still sourced (not missing)
    ok = RewardConsistencyMonitor().check_reward_alignment(
        [{"policy": "a", "total_reward": -100, "monitor_score": 0.5},
         {"policy": "b", "total_reward": -200, "monitor_score": 0.1}])
    exp_ok = export_cip_variables(verdict_dict(reward_consistency=ok.as_dict()))
    assert exp_ok.variables["reward_progress_disagreement"] == 0.0
    assert "reward_progress_disagreement" not in exp_ok.missing


def test_defaults_are_deterministic():
    a = export_cip_variables(verdict_dict())
    b = export_cip_variables(verdict_dict())
    assert a == b   # frozen dataclass equality over variables + missing + source_keys
    assert a.variables["clearance_min"] == 0.0
    assert a.variables["forbidden_contact_count"] == 0
    assert a.variables["stagnated"] is False


def test_no_mutation_of_input():
    d = verdict_dict(sub_verdicts={"stagnation": _stagnation_subverdict(3, True)})
    snapshot = copy.deepcopy(d)
    exp = export_cip_variables(d)
    assert d == snapshot            # input verdict is untouched
    assert exp.variables is not d   # returned variables is a fresh object
