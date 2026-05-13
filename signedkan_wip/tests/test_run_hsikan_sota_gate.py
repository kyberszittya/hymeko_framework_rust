"""Fast CLI checks for ``run_hsikan_sota_gate`` (no full training)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_require_cuda_rejects_cpu_device():
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "signedkan_wip.src.benchmarks.run_hsikan_sota_gate",
            "--datasets",
            "bitcoin_alpha",
            "--seeds",
            "0",
            "--device",
            "cpu",
            "--require-cuda",
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr
    assert "require-cuda" in proc.stderr.lower()


def test_gate_help_lists_vram_knobs():
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "signedkan_wip.src.benchmarks.run_hsikan_sota_gate",
            "--help",
        ],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = (proc.stdout or "") + (proc.stderr or "")
    assert "--cycle-batch" in out
    assert "--max-k3" in out
