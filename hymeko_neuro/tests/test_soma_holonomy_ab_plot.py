"""Tests for the Gömb-Soma holonomy A/B plot shaper + renderer."""
from __future__ import annotations

from pathlib import Path

from hymeko_neuro.experiments.runs.soma_holonomy_ab_plot import (
    ARM_ORDER,
    render,
    summarize,
)

_RECORDS = [
    {"model": "gomb_soma_holonomy", "test_acc": 0.60, "n_params": 1226},
    {"model": "linear", "test_acc": 0.90, "n_params": 7850},
    {"model": "gomb_soma", "test_acc": 0.50, "n_params": 2010},
    {"model": "gomb_soma", "test_acc": 0.54, "n_params": 2010},
    {"model": "gomb_soma_holonomy", "test_acc": 0.64, "n_params": 1226},
]


def test_summarize_orders_arms_and_computes_mean_pstd() -> None:
    bars = summarize(_RECORDS)
    # Present arms appear in canonical ARM_ORDER order, regardless of record order.
    present = {r["model"] for r in _RECORDS}
    assert [b["model"] for b in bars] == [m for m in ARM_ORDER if m in present]
    soma = next(b for b in bars if b["model"] == "gomb_soma")
    assert soma["n_seeds"] == 2
    assert abs(soma["mean"] - 0.52) < 1e-9
    assert abs(soma["pstd"] - 0.02) < 1e-9  # pstdev of {0.50, 0.54}


def test_summarize_single_seed_has_zero_pstd() -> None:
    bars = summarize([{"model": "linear", "test_acc": 0.9, "n_params": 7850}])
    assert len(bars) == 1
    assert bars[0]["pstd"] == 0.0
    assert bars[0]["n_seeds"] == 1


def test_summarize_appends_unknown_arm_without_inventing() -> None:
    bars = summarize(_RECORDS + [{"model": "mystery", "test_acc": 0.1, "n_params": 5}])
    models = [b["model"] for b in bars]
    assert models[-1] == "mystery"          # appended, not reordered into ARM_ORDER
    present = {r["model"] for r in _RECORDS}
    assert set(models) == present | {"mystery"}


def test_render_writes_png(tmp_path: Path) -> None:
    out = render(summarize(_RECORDS), tmp_path / "ab.png")
    assert out.exists() and out.stat().st_size > 0
