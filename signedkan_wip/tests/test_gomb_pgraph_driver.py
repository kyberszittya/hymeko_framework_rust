"""Gömb + P-graph driver / mapping tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from signedkan_wip.src.gomb_pgraph_mapping import merge_structure_knobs
from signedkan_wip.src.hymeko_driver import parse_training, run_single_gomb


def test_merge_structure_knobs_orders_units():
    m = merge_structure_knobs(["gomb_slow", "gomb_fit"])
    assert m["topk"] == 48
    assert m["cycle_abb_mode"] == "start_local"
    assert m["n_epochs"] == 4


def test_parse_gomb_training_fixture():
    repo = Path(__file__).resolve().parents[2]
    k = parse_training(str(repo / "data" / "hsikan" / "gomb_training.hymeko"))
    assert k["dataset_name"] == "sbm_n200"
    assert k["gomb_topk"] == 24


@pytest.mark.timeout(180)
def test_run_single_gomb_smoke_subprocess():
    repo = Path(__file__).resolve().parents[2]
    k = parse_training(str(repo / "data" / "hsikan" / "gomb_training.hymeko"))
    k["n_epochs"] = 1
    k["gomb_topk"] = 12
    out = run_single_gomb(k, dataset=None, device="cpu")
    assert out.get("backend") == "gomb"
    assert "error" not in out, out
    assert out.get("n_cycles", 0) > 0


@pytest.mark.timeout(120)
def test_hymeko_driver_gomb_sweep_one_cell():
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    cmd = [
        sys.executable,
        "-m",
        "signedkan_wip.src.hymeko_driver",
        "--backend",
        "gomb",
        "--sweep",
        str(repo / "data" / "hsikan" / "sweep_grid_gomb.hymeko"),
        "--dataset",
        "sbm_n200",
        "--device",
        "cpu",
        "--max-runs",
        "1",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, proc.stdout[-2000:]
    row = json.loads(lines[0])
    assert row.get("backend") == "gomb"


@pytest.mark.timeout(180)
def test_run_gomb_msg_sweep_msg_phase_only():
    repo = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            "cargo",
            "build",
            "-q",
            "-p",
            "hymeko_pgraph",
            "--bin",
            "hymeko_pgraph_dump",
        ],
        cwd=str(repo),
        check=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    cmd = [
        sys.executable,
        "-m",
        "signedkan_wip.experiments.runs.run_gomb_msg_sweep",
        "--pgraph",
        str(repo / "data" / "hsikan" / "sweep_msg_gomb.hymeko"),
        "--algorithm",
        "msg",
        "--training",
        str(repo / "data" / "hsikan" / "gomb_training.hymeko"),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert "gomb_fast" in proc.stdout or "msg_units" in proc.stdout
