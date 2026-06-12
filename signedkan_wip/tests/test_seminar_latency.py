"""Tests for the seminar ``latency`` demo (Demo 2 / plan-item 4).

No torch / no engine needed — the demo reads the committed
``inference_bench.json`` and renders bars via the pure ``bench_to_png`` helpers.
Run via the venv interpreter::

    PYTHONPATH=. .venv/Scripts/python.exe -m pytest -p no:randomly \
        signedkan_wip/tests/test_seminar_latency.py
"""
from __future__ import annotations

import json

import pytest

from signedkan_wip.demos.seminar.base import DemoRunner
from signedkan_wip.demos.seminar.cli import build_parser
from signedkan_wip.demos.seminar.demos import DEMOS
from signedkan_wip.demos.seminar.demos.latency import (
    DEFAULT_JSON, JOINT_MODEL, LEAN_MODEL, width_ratios,
)

# ─── Unit: width_ratios (pure) ──────────────────────────────────────────


def _row(dataset: str, device: str, model: str, med: float,
         iqr: float = 1.0, worst: float | None = None, params: int = 100):
    return {
        "dataset": dataset, "device": device, "model": model,
        "fwd_med_ms": med, "fwd_iqr_ms": iqr,
        "fwd_worst_ms": med if worst is None else worst, "n_params": params,
    }


def test_width_ratios_computes_joint_over_lean() -> None:
    rows = [
        _row("d1", "cpu", LEAN_MODEL, 100.0, params=15),
        _row("d1", "cpu", JOINT_MODEL, 350.0, params=61),
    ]
    out = width_ratios(rows)
    assert len(out) == 1
    rec = out[0]
    assert rec["ratio"] == 3.5
    assert rec["lean_ms"] == 100.0 and rec["joint_ms"] == 350.0
    assert rec["lean_params"] == 15 and rec["joint_params"] == 61


def test_width_ratios_skips_cell_missing_a_variant() -> None:
    # Only the lean cell present -> no record (never fabricated).
    rows = [_row("d1", "cpu", LEAN_MODEL, 100.0)]
    assert width_ratios(rows) == []


def test_width_ratios_skips_nonpositive_lean() -> None:
    rows = [
        _row("d1", "cpu", LEAN_MODEL, 0.0),
        _row("d1", "cpu", JOINT_MODEL, 350.0),
    ]
    assert width_ratios(rows) == []


# ─── Data invariants on the committed json (CLAUDE.md §3) ───────────────

_HAS_JSON = DEFAULT_JSON.is_file()


@pytest.mark.skipif(not _HAS_JSON, reason="needs inference_bench.json")
def test_committed_json_has_stable_benchmark_fields() -> None:
    rows = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    hsikan = [r for r in rows if r["model"] in (LEAN_MODEL, JOINT_MODEL)]
    assert hsikan, "no HSiKAN cells in the committed bench"
    for r in hsikan:
        assert r["fwd_iqr_ms"] >= 0.0, r
        assert r["fwd_worst_ms"] >= r["fwd_med_ms"], r


# ─── Integration via DemoRunner ─────────────────────────────────────────


def test_latency_is_registered() -> None:
    assert "latency" in DEMOS and DEMOS["latency"].name == "latency"


def _run(tmp_path, extra: list[str]):
    args = build_parser().parse_args(["latency", "--device", "cpu", *extra])
    return DemoRunner(DEMOS["latency"], out_root=tmp_path).run(args)


@pytest.mark.skipif(not _HAS_JSON, reason="needs inference_bench.json")
def test_latency_reports_ratios_and_honest_note(tmp_path) -> None:
    result = _run(tmp_path, ["--no-figures"])
    # At least the CPU within-family ratio is surfaced and positive.
    ratio_keys = [k for k in result.metrics if k.endswith("joint/lean")]
    assert ratio_keys, result.metrics
    for k in ratio_keys:
        assert float(str(result.metrics[k]).rstrip("x")) > 0.0
        assert result.provenance.get(k, "").endswith("inference_bench.json")
    # The "not faster than SGCN" honesty line is present.
    assert any("not 'faster than SGCN'" in n for n in result.notes)


@pytest.mark.skipif(not _HAS_JSON, reason="needs inference_bench.json")
def test_latency_renders_bars(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    result = _run(tmp_path, [])
    assert result.artifacts, "no PNG written"
    for p in result.artifacts:
        assert p.is_file() and p.suffix == ".png"


@pytest.mark.skipif(not _HAS_JSON, reason="needs inference_bench.json")
def test_latency_no_figures_writes_nothing(tmp_path) -> None:
    result = _run(tmp_path, ["--no-figures"])
    assert result.artifacts == []
