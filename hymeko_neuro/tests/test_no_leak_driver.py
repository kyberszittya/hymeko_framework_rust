"""Regression tests for the no-leak grid driver's dispatch + resume keying.

Covers the reachability threading added 2026-06-14: the gomb path is unchanged,
baseline cells pass ``--reachability``, and the resume key separates
strict/topo/full rows (so they coexist in one JSONL and legacy rows default to
strict). Pure / fast — no training.

Run: ``pytest -p no:randomly hymeko_neuro/tests/test_no_leak_driver.py``
"""
from __future__ import annotations

from hymeko_neuro.experiments.runs import run_no_leak_benchmark as D


def test_gomb_argv_tail_unchanged() -> None:
    """The structural-prior runner's argv must not gain reachability flags."""
    cell = D.Cell.make("gomb", "bitcoin_alpha", n_epochs=60)
    tail = D.SUBPROC_RUNNERS["gomb"].arg_builder(cell)
    assert tail == ["--d-middle", str(cell.width), "--middle-n-layers", str(cell.depth)]
    assert "--reachability" not in tail


def test_baseline_argv_carries_reachability() -> None:
    for rule in ("strict", "topo", "full"):
        cell = D.Cell(model="sgt", dataset="bitcoin_alpha", n_epochs=80,
                      width=32, depth=0, reachability=rule)
        tail = D.SUBPROC_RUNNERS["sgt"].arg_builder(cell)
        assert tail == ["--model", "sgt", "--reachability", rule]


def test_resume_key_separates_reachability() -> None:
    base = dict(dataset="bitcoin_alpha", model="sgt", seed=0, shuffle=False)
    k_strict = D._result_key({**base, "reachability": "strict"})
    k_topo = D._result_key({**base, "reachability": "topo"})
    assert k_strict != k_topo
    # Legacy row (no reachability field) is treated as strict — no collision with
    # topo, and resumes an old strict-only JSONL cleanly.
    assert D._result_key(base) == k_strict


def test_cell_defaults_to_strict() -> None:
    assert D.Cell.make("sgcn", "bitcoin_alpha", 120).reachability == "strict"
    assert all(c.reachability == "strict" for c in D.BASELINE_CELLS)
