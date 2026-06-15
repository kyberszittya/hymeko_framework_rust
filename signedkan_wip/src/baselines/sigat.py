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
from .registry import GraphMeta, HParams, SignedLinkBaseline, register


@register("sigat")
class SiGATBaseline(SignedLinkBaseline):
    """SiGAT-style motif-typed multi-head attention (pos/neg buckets)."""

    def build_model(self, meta: GraphMeta, hp: HParams):
        from .sigat_model import SiGATAttn
        return SiGATAttn(n_nodes=meta.n_nodes, hidden_dim=hp.hidden,
                         n_heads=hp.n_heads, n_layers=hp.n_layers)

    def build_context(self, edges, signs, n_nodes, device):
        from .sigat_model import build_neighbour_lists
        # Python neighbour buckets; the attention builds its tensors on h.device.
        return build_neighbour_lists(edges, signs, n_nodes)

    def default_hparams(self) -> HParams:
        return HParams(hidden=32, n_layers=1, n_heads=4)


def export_for_sigat(g: SignedGraph, out_path: Path,
                     train_idx: np.ndarray) -> None:
    """SiGAT expects a similar `source target sign` triples format
    as SGCN. Reuse the same exporter."""
    from .sgcn import export_for_sgcn
    export_for_sgcn(g, out_path, train_idx)
