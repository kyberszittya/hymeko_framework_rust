"""Phase 3 baseline: SiGAT (Huang et al. 2019).

Same pattern as SGCN: keep the reference impl out-of-tree, write a
small data-format adapter so our SignedGraph + edge splits feed
their model. Reference code: search "SiGAT signed graph attention"
on author repos.

Stub for Phase 3 morning. Acceptable fallback: compare against
published SiGAT numbers if reproduction proves expensive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..datasets import SignedGraph


def export_for_sigat(g: SignedGraph, out_path: Path,
                     train_idx: np.ndarray) -> None:
    """SiGAT expects a similar `source target sign` triples format
    as SGCN. Reuse the same exporter."""
    from .sgcn import export_for_sgcn
    export_for_sgcn(g, out_path, train_idx)
