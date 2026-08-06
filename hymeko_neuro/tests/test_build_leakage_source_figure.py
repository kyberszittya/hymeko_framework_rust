"""Tests for the leakage-by-source figure builder (paperkit).

Layers (CLAUDE.md §3): unit (Arm stats, finite filtering, dedup last-wins,
lattice ordering), integration (both builders emit pdf+png+eps to a tmp dir on
synthetic data; partial-data tolerance). All deterministic, no network, no GPU.

Run: ``pytest -p no:randomly hymeko_neuro/tests/test_build_leakage_source_figure.py``
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hymeko_neuro.paperkit import build_leakage_source_figure as mod
from hymeko_neuro.paperkit.build_leakage_source_figure import (
    Arm,
    build_lattice_figure,
    build_source_contrast_figure,
    load_baselines,
    load_lattice,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


# --- Arm (unit) -------------------------------------------------------------

def test_arm_mean_std_count() -> None:
    a = Arm.from_values([0.4, 0.6])
    assert a is not None
    assert a.mean == pytest.approx(0.5)
    assert a.std == pytest.approx(0.1)
    assert a.n == 2


def test_arm_drops_nan_and_none() -> None:
    a = Arm.from_values([0.5, float("nan"), None, 0.7])  # type: ignore[list-item]
    assert a is not None and a.n == 2
    assert a.mean == pytest.approx(0.6)


def test_arm_all_nonfinite_returns_none() -> None:
    assert Arm.from_values([float("nan"), None]) is None  # type: ignore[list-item]
    assert Arm.from_values([]) is None


# --- loaders (unit) ---------------------------------------------------------

def test_load_lattice_pools_and_orders(tmp_path: Path) -> None:
    rows = []
    for rule, real, shuf in (("strict", 0.5, 0.5), ("topo", 0.8, 0.47), ("full", 0.88, 0.73)):
        for seed in range(3):
            rows.append({"dataset": "d", "rule": rule, "seed": seed, "shuffle": False, "auc": real})
            rows.append({"dataset": "d", "rule": rule, "seed": seed, "shuffle": True, "auc": shuf})
    p = tmp_path / "grid.jsonl"
    _write_jsonl(p, rows)
    lat = load_lattice(p)
    assert set(lat) == {"strict", "topo", "full"}
    assert lat["full"][False].mean == pytest.approx(0.88)
    assert lat["full"][True].mean == pytest.approx(0.73)
    assert lat["topo"][True].mean == pytest.approx(0.47)
    # ordering invariant that the figure relies on
    assert lat["strict"][True].mean <= lat["topo"][True].mean + 0.05
    assert lat["full"][True].mean > lat["topo"][True].mean


def test_load_baselines_dedups_last_wins(tmp_path: Path) -> None:
    # seed 0 appears twice (a re-run); the later value must win, not be averaged in.
    rows = [
        {"model": "sgcn", "dataset": "d", "seed": 0, "shuffle": False, "auc": 0.10},
        {"model": "sgcn", "dataset": "d", "seed": 0, "shuffle": False, "auc": 0.90},
        {"model": "sgcn", "dataset": "d", "seed": 1, "shuffle": False, "auc": 0.90},
        {"model": "sgcn", "dataset": "d", "seed": 0, "shuffle": True, "auc": 0.50},
        {"model": "sgcn", "dataset": "d", "seed": 1, "shuffle": True, "auc": 0.50},
    ]
    p = tmp_path / "base.jsonl"
    _write_jsonl(p, rows)
    b = load_baselines(p)
    # if the 0.10 dup were counted, the real mean would be (0.1+0.9+0.9)/3 = 0.633
    assert b["sgcn"][False].mean == pytest.approx(0.90)
    assert b["sgcn"][False].n == 2  # two distinct seeds, not three rows
    assert b["sgcn"][True].mean == pytest.approx(0.50)


def test_load_baselines_skips_model_missing_an_arm(tmp_path: Path) -> None:
    rows = [
        {"model": "only_real", "dataset": "d", "seed": 0, "shuffle": False, "auc": 0.8},
        {"model": "both", "dataset": "d", "seed": 0, "shuffle": False, "auc": 0.8},
        {"model": "both", "dataset": "d", "seed": 0, "shuffle": True, "auc": 0.5},
    ]
    p = tmp_path / "base.jsonl"
    _write_jsonl(p, rows)
    b = load_baselines(p)
    assert "both" in b
    assert "only_real" not in b  # partial-data tolerance: no shuffle arm -> dropped


# --- builders (integration) -------------------------------------------------

def _emitted(stem: str, fig_dir: Path) -> bool:
    return all((fig_dir / f"{stem}.{ext}").exists() for ext in ("pdf", "png", "eps"))


def test_builders_emit_all_extensions_on_synthetic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    lat = mod._synthetic_lattice()
    bs = mod._synthetic_baselines(0.52)
    bt = mod._synthetic_baselines(0.51)

    n_rules = build_lattice_figure(lat, "lat_test")
    n_models = build_source_contrast_figure(bs, bt, lat, "src_test")

    assert n_rules == 3
    assert n_models == len(mod.BASELINE_ORDER)
    assert _emitted("lat_test", tmp_path)
    assert _emitted("src_test", tmp_path)


def test_lattice_figure_tolerates_partial_rules(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    partial = {
        "topo": {False: Arm(0.8, 0.0, 5), True: Arm(0.47, 0.0, 5)},
        "full": {False: Arm(0.88, 0.0, 5), True: Arm(0.73, 0.0, 5)},
    }
    n = build_lattice_figure(partial, "lat_partial")  # strict absent -> still draws 2
    assert n == 2
    assert _emitted("lat_partial", tmp_path)


def test_self_check_main_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "FIG_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["prog", "--self-check"])
    mod.main()
    assert _emitted("leakage_lattice", tmp_path)
    assert _emitted("leakage_source_contrast", tmp_path)


# --- contract sanity on the real data shape (skipped if data absent) --------

def test_real_data_lattice_shape_if_present() -> None:
    if not mod.GRID.exists():
        pytest.skip("cycle_reachability_grid.jsonl not present")
    lat = load_lattice(mod.GRID)
    assert {"strict", "topo", "full"} <= set(lat)
    # measured finding: leak (shuffle above chance) only at full
    assert lat["strict"][True].mean == pytest.approx(0.5, abs=0.02)
    assert lat["topo"][True].mean < mod.LEAK_THRESHOLD
    assert lat["full"][True].mean > mod.LEAK_THRESHOLD
    assert not math.isnan(lat["full"][False].mean)
