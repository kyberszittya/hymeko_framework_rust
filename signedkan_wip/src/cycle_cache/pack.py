"""Auto-split from cycle_cache.py 2026-05-11 (CLAUDE.md §6.5 #4)."""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any
import numpy as np



def _pack_and_drop(t_list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Serialize a list[SignedNTuple] (or SignedTriad) to numpy arrays,
    releasing each entry as it is copied so peak RSS during the pack is
    bounded by ``len(t_list) + arrays`` instead of ``2 * len(t_list)``.

    On return, ``t_list`` contains ``None`` placeholders — the original
    SignedNTuple objects become eligible for garbage collection. The
    caller should ``del t_list`` to drop the outer list shell as well.

    Output shapes:
        v          : (N, arity)  int32
        sigma      : (N, arity)  int8
        edge_signs : (N, arity)  int8   (zeros if not provided)
    """
    n = len(t_list)
    if n == 0:
        empty_i = np.zeros((0, 0), dtype=np.int32)
        empty_s = np.zeros((0, 0), dtype=np.int8)
        return empty_i, empty_s, empty_s.copy()
    arity = len(t_list[0].v)
    v = np.empty((n, arity), dtype=np.int32)
    sigma = np.empty((n, arity), dtype=np.int8)
    edge_signs = np.zeros((n, arity), dtype=np.int8)
    for i in range(n):
        t = t_list[i]
        v[i] = t.v
        sigma[i] = t.sigma
        es = getattr(t, "edge_signs", None)
        if es is not None and len(es) == arity:
            edge_signs[i] = es
        t_list[i] = None
    return v, sigma, edge_signs


def _unpack_to_ntuples(v: np.ndarray, sigma: np.ndarray,
                        edge_signs: np.ndarray | None):
    """Rebuild list[SignedNTuple] from packed arrays. All five
    SignedNTuple dataclass fields are populated:

    - ``arity`` is derived from ``v.shape[1]``
    - ``edge_signs`` is read from the packed array (or zero-filled if
      missing — backward-compat for cache files written before
      edge_signs was stored)
    - ``balanced`` is derived from ``prod(edge_signs) == +1``; if
      ``edge_signs`` is unavailable, ``balanced`` defaults to False

    Performance: the per-row int casts go through ``ndarray.tolist()``
    (C-level conversion to native Python ints) rather than a
    ``tuple(int(x) for x in row)`` generator — ~10× faster on
    multi-million-cycle Epinions enumerations where the unpack stage
    dominates seed-1..N cache-hit latency.
    """
    from ..core.n_tuples import SignedNTuple
    n = v.shape[0]
    arity = v.shape[1] if v.ndim == 2 else 0
    if n == 0:
        return []
    has_edge_signs = (
        edge_signs is not None
        and edge_signs.ndim == 2
        and edge_signs.shape == v.shape
        and bool(np.any(edge_signs))
    )
    v_lists = v.tolist()
    sigma_lists = sigma.tolist()
    if has_edge_signs:
        es_lists = edge_signs.tolist()
        balanced_arr = (edge_signs.astype(np.int32).prod(axis=1) == 1)
    else:
        es_lists = None
        balanced_arr = None
    out = [None] * n
    for i in range(n):
        if has_edge_signs:
            es = tuple(es_lists[i])
            bal = bool(balanced_arr[i])
        else:
            es = ()
            bal = False
        out[i] = SignedNTuple(
            v=tuple(v_lists[i]),
            sigma=tuple(sigma_lists[i]),
            edge_signs=es,
            balanced=bal,
            arity=arity,
        )
    return out

