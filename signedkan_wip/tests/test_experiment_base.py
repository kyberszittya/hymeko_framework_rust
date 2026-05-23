"""Tests for the ExperimentBase / SimpleExperiment / observer
machinery shipped by Slice H pilot (signedkan_wip reorg, 2026-05-19).

The base lives in
``signedkan_wip/experiments/runs/_experiment_base.py``. These tests
pin its public contract without exercising any concrete experiment
(no GPU, no datasets — fully synthetic).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from signedkan_wip.experiments.runs._experiment_base import (
    CallbackObserver,
    EpochEvent,
    JsonlObserver,
    RunEvent,
    SeedEvent,
    SimpleExperiment,
    StdoutObserver,
)


# ─── SimpleExperiment ──────────────────────────────────────────────


class _Plus1Experiment(SimpleExperiment):
    """Trivial experiment: result is just (seed + offset)."""

    def __init__(self, offset: int) -> None:
        super().__init__(label="plus1")
        self.offset = offset

    def run_seed(self, seed: int, **cfg) -> dict:
        return {
            "seed":       seed,
            "score":      float(seed + self.offset),
            "elapsed_s":  0.1,
        }


def test_simple_experiment_runs_one_seed():
    exp = _Plus1Experiment(offset=10)
    results = exp.run([7])
    assert len(results) == 1
    assert results[0]["seed"] == 7
    assert results[0]["score"] == pytest.approx(17.0)


def test_simple_experiment_runs_five_seeds():
    exp = _Plus1Experiment(offset=0)
    results = exp.run([0, 1, 2, 3, 4])
    assert [r["seed"] for r in results] == [0, 1, 2, 3, 4]
    assert [r["score"] for r in results] == [0, 1, 2, 3, 4]


def test_simple_experiment_run_seed_must_be_overridden():
    exp = SimpleExperiment(label="abstract")
    with pytest.raises(NotImplementedError):
        exp.run([0])


# ─── Observer protocol ─────────────────────────────────────────────


def test_callback_observer_fires_per_seed():
    exp = _Plus1Experiment(offset=0)
    seen: list[int] = []
    obs = CallbackObserver(on_seed_end=lambda ev: seen.append(ev.seed))
    exp.add_observer(obs)
    exp.run([0, 1, 2])
    assert seen == [0, 1, 2]


def test_jsonl_observer_writes_one_line_per_seed(tmp_path):
    exp = _Plus1Experiment(offset=100)
    log = tmp_path / "results.jsonl"
    exp.add_observer(JsonlObserver(str(log), mode="w"))
    exp.run([0, 1])
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["seed"] == 0
    assert parsed[0]["score"] == 100.0
    assert parsed[1]["seed"] == 1
    assert parsed[1]["score"] == 101.0


def test_run_summary_includes_mean_and_std():
    exp = _Plus1Experiment(offset=0)
    summary_holder: dict = {}
    obs = CallbackObserver(
        on_seed_end=lambda ev: None,
    )
    # Use a class-style observer to capture on_run_end too.
    from signedkan_wip.experiments.runs._experiment_base import (
        ExperimentObserver,
    )

    class _Capture(ExperimentObserver):
        def on_run_end(self, ev: RunEvent) -> None:
            summary_holder.update(ev.summary)

    exp.add_observer(_Capture())
    exp.run([0, 1, 2, 3, 4])
    # Mean over [0,1,2,3,4] = 2.0; std = 1.414... (population std).
    assert summary_holder["n_seeds"] == 5
    assert summary_holder["score_mean"] == pytest.approx(2.0)
    assert summary_holder["score_std"] == pytest.approx(1.4142, abs=0.01)


def test_multiple_observers_all_fire():
    exp = _Plus1Experiment(offset=0)
    n_seen_a = 0
    n_seen_b = 0

    def cb_a(ev: SeedEvent) -> None:
        nonlocal n_seen_a
        n_seen_a += 1

    def cb_b(ev: SeedEvent) -> None:
        nonlocal n_seen_b
        n_seen_b += 1

    exp.add_observer(CallbackObserver(on_seed_end=cb_a))
    exp.add_observer(CallbackObserver(on_seed_end=cb_b))
    exp.run([0, 1, 2])
    assert n_seen_a == 3
    assert n_seen_b == 3


def test_run_event_carries_label_and_seeds():
    exp = _Plus1Experiment(offset=0)
    capture: list[RunEvent] = []
    from signedkan_wip.experiments.runs._experiment_base import (
        ExperimentObserver,
    )

    class _Cap(ExperimentObserver):
        def on_run_start(self, ev: RunEvent) -> None:
            capture.append(ev)

    exp.add_observer(_Cap())
    exp.run([7, 8, 9])
    assert capture[0].label == "plus1"
    assert capture[0].seeds == [7, 8, 9]


def test_add_observer_is_chainable():
    exp = _Plus1Experiment(offset=0)
    returned = exp.add_observer(CallbackObserver())
    assert returned is exp


def test_event_dataclasses_are_frozen():
    ev = SeedEvent(seed=42, final_metrics={"x": 1.0})
    with pytest.raises(Exception):
        ev.seed = 99  # type: ignore[misc]
