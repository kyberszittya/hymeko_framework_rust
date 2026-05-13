from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_gomb_jsonl_summarize_table(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    p.write_text(
        json.dumps({"trial": 0, "test_auroc": 0.7}) + "\n"
        + json.dumps(
            {
                "tuner_phase_summary": True,
                "dataset": "bitcoin_alpha",
                "best_score": 0.88,
                "tuner_pick_best_by": "val_auroc",
                "best_test_auroc": 0.85,
                "best_val_auroc": 0.88,
                "best_n_params": 1000,
                "trials": 3,
                "wall_s": 12.5,
            },
        )
        + "\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "signedkan_wip.src.gomb_jsonl_summarize",
            str(p),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    assert proc.returncode == 0, proc.stderr
    assert "bitcoin_alpha" in proc.stdout
    assert "0.88" in proc.stdout
    assert "val_auroc" in proc.stdout
