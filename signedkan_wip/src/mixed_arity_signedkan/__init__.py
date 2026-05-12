"""MixedAritySignedKAN — split out of the 1247-LOC monolith 2026-05-11
per CLAUDE.md §6.5 #4. Sub-modules (all ≤ 300 LOC):

  config.py            MixedAritySignedKANConfig
  scatter.py           scatter softmax helpers (incl. sparse top-K)
  attention.py         _AttentionM_e + _QuaternionAttentionM_e
  utils.py             subsample_tuples / build_edge_to_tuples / …
  encoding_full.py     full-graph encoding (extracted method body)
  encoding_batched.py  cycle-batched encoding (extracted method body)
  model.py             MixedAritySignedKAN class (delegates to encoding_*)

External `from signedkan_wip.src.mixed_arity_signedkan import X` stays flat
via the re-exports below.
"""
from .config import MixedAritySignedKANConfig
from .model import MixedAritySignedKAN
from .attention import _AttentionM_e, _QuaternionAttentionM_e
from .utils import (
    subsample_tuples, build_edge_to_tuples, build_vertex_to_tuples,
)
__all__ = [
    "MixedAritySignedKANConfig",
    "MixedAritySignedKAN",
    "_AttentionM_e",
    "_QuaternionAttentionM_e",
    "subsample_tuples", "build_edge_to_tuples", "build_vertex_to_tuples",
]
