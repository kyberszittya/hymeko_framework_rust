"""Smoke for ``run_gomb_cycle_abb_compare`` benchmark driver."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.timeout(240)
def test_run_gomb_cycle_abb_compare_two_modes_sbm_cpu():
    """Two paired smokes (none vs start_local) must exit 0 and emit two JSON rows."""
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    cmd = [
        sys.executable,
        "-m",
        "signedkan_wip.src.benchmarks.run_gomb_cycle_abb_compare",
        "--dataset",
        "sbm_n200",
        "--edge-split",
        "80_10_10",
        "--device",
        "cpu",
        "--seed",
        "0",
        "--n-epochs",
        "1",
        "--topk",
        "12",
        "--d-embed",
        "16",
        "--d-outer",
        "8",
        "--M-outer",
        "2",
        "--d-middle",
        "16",
        "--d-core",
        "16",
        "--modes",
        "none",
        "start_local",
        "--timeout-s",
        "180",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env=env,
        timeout=200,
    )
    assert proc.returncode == 0, (proc.stderr or "")[-4000:]
    out = proc.stdout
    assert "| mode |" in out and "n_cycles" in out
    rows: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("| ") and not line.startswith("| ---"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if parts and parts[0] in ("none", "start_local", "global_min"):
                rows.append({"mode": parts[0], "raw": parts})
    assert len(rows) >= 2


def test_parse_last_gomb_json_line_from_compare_module():
    from signedkan_wip.src.benchmarks.run_gomb_cycle_abb_compare import (
        parse_last_gomb_json_line,
    )

    blob = 'noise\n{"dataset": "x", "n_cycles": 42, "wall_s": 1.5}\n'
    row = parse_last_gomb_json_line(blob)
    assert row["dataset"] == "x"
    assert row["n_cycles"] == 42

    with pytest.raises(ValueError):
        parse_last_gomb_json_line("no json here\n")
