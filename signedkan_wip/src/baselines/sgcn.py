"""Phase 3 baseline: SGCN (Derr et al. 2018).

The reference implementation lives at
    https://github.com/benedekrozemberczki/SGCN

Phase 3 strategy:
  1. Clone the reference repo to /tmp/sgcn_ref (manual; the sandbox
     blocked auto-clone).
  2. Adapt their data loader to consume our SignedGraph + the same
     edge split.
  3. Run on Bitcoin Alpha + Bitcoin OTC at default hyperparameters.
  4. Verify reproducibility against published numbers
     (AUC ≈ 0.93 on Bitcoin Alpha link sign prediction at the
     standard 80/10/10 split).

This stub provides the data-format adapter; the SGCN model itself is
imported from the cloned reference. If reproduction fails on Phase 3
morning, fall back to comparing our results against the published
table values directly (acceptable per WiPI standards).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..datasets import SignedGraph


def export_for_sgcn(g: SignedGraph, out_path: Path,
                    train_idx: np.ndarray) -> None:
    """SGCN expects a CSV of `source target sign` for the training
    edges only (test/val edges are masked out)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(int(s), int(t), int(sg))
            for (s, t), sg in zip(g.edges[train_idx], g.signs[train_idx])]
    with out_path.open("w") as f:
        f.write("source,target,sign\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")


def sgcn_command_line(graph_csv: Path, output_dir: Path,
                       epochs: int = 100) -> str:
    """Return the SGCN reference command line. The user runs this
    after cloning the reference repo."""
    return (
        f"python3 /tmp/sgcn_ref/src/main.py "
        f"--edge-path {graph_csv} "
        f"--output-path {output_dir} "
        f"--epochs {epochs}"
    )
