"""Joint **c3,c4,w2,w3** tuple pools for HymeKo-Gömb (same recipe as ``joint_ba``).

Builds per-slot ``(cycles, signs)`` tensors on **train edges only**, mirroring
``run_final_cell`` / ``HSIKAN_MIXED_TUPLES=c3,c4,w2,w3`` enumeration semantics
without importing ``run_final_cell`` (avoids heavy training deps in import
graph).

Cycle slots **c3** / **c4** call ``hymeko.enumerate_cycles_rs`` with optional
**ABB** modes (``cycle_abb_mode`` / ``cycle_abb_fullness_gate``) — same knobs as
``run_gomb_smoke --cycle-abb-mode …``.
"""
from __future__ import annotations

import numpy as np

import hymeko

from ..datasets import SignedGraph
from ..core.walks import construct_walks

# Canonical joint-mix slot order (matches committed JSONL ``tuple_labels``).
JOINT_BA_SLOTS: tuple[str, ...] = ("c3", "c4", "w2", "w3")

# Vertices per slot (FIR / HSiKAN / CPML all use ``k`` as tuple width).
SLOT_K: dict[str, int] = {"c3": 3, "c4": 4, "w2": 3, "w3": 4}


def _enumerate_cycles_rs(
    edges: np.ndarray,
    signs: np.ndarray,
    n: int,
    k: int,
    m_per_vertex: int,
    *,
    abb_mode: str = "none",
    abb_fullness_gate: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    eu = np.ascontiguousarray(edges[:, 0], dtype=np.uint32)
    ev = np.ascontiguousarray(edges[:, 1], dtype=np.uint32)
    es = np.ascontiguousarray(signs, dtype=np.int8)
    arr, _ = hymeko.enumerate_cycles_rs(
        eu, ev, es, n, k, m_per_vertex,
        score_kind="fraction_negative",
        pruner_kind="none",
        filter_kind="none",
        filter_min_degree=2,
        abb_mode=abb_mode,
        fullness_gate=float(abb_fullness_gate),
        tiers=[],
        adaptive_c=0.0,
        adaptive_m_min=0,
        adaptive_m_max=0,
    )
    cycles = np.asarray(arr, dtype=np.int64)
    sign_of: dict[tuple[int, int], int] = {}
    for (u, v), s in zip(edges, signs):
        sign_of[(int(u), int(v))] = int(s)
        sign_of[(int(v), int(u))] = int(s)
    cyc_signs = np.zeros_like(cycles, dtype=np.int8)
    for ci, cycle in enumerate(cycles):
        for j in range(k):
            u, v = int(cycle[j]), int(cycle[(j + 1) % k])
            cyc_signs[ci, j] = sign_of.get((u, v), 1)
    return cycles, cyc_signs


def _pack_signed_ntuples(
    tuples: list, k: int, cap: int | None, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``SignedNTuple`` list → ``(M, k)`` int64 vertices + int8 σ."""
    if not tuples:
        return (
            np.zeros((0, k), dtype=np.int64),
            np.zeros((0, k), dtype=np.int8),
        )
    if cap is not None and len(tuples) > cap:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(tuples), size=cap, replace=False)
        tuples = [tuples[int(i)] for i in pick]
    v_mat = np.array([t.v for t in tuples], dtype=np.int64)
    s_mat = np.array([t.sigma for t in tuples], dtype=np.int8)
    return v_mat, s_mat


def build_joint_ba_pools(
    edges: np.ndarray,
    signs: np.ndarray,
    n_nodes: int,
    *,
    topk_c3: int = 64,
    topk_c4: int = 64,
    max_walks_w2: int | None = 50_000,
    max_walks_w3: int | None = 50_000,
    walk_seed: int = 0,
    subsample_walks_seed: int = 0,
    cycle_abb_mode: str = "none",
    cycle_abb_fullness_gate: float = 0.25,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``slot -> (cycles M×k, signs M×k)`` for each joint-ba slot.

    Preconditions:
        ``edges`` / ``signs`` are **train-only** splits (caller responsibility).
        ``n_nodes`` matches graph order used in vertex indices.
    """
    g = SignedGraph(
        edges=np.asarray(edges, dtype=np.int64),
        signs=np.asarray(signs, dtype=np.int8),
        n_nodes=int(n_nodes),
    )
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    c3, s3 = _enumerate_cycles_rs(
        edges, signs, n_nodes, 3, topk_c3,
        abb_mode=cycle_abb_mode,
        abb_fullness_gate=cycle_abb_fullness_gate,
    )
    out["c3"] = (c3, s3)
    c4, s4 = _enumerate_cycles_rs(
        edges, signs, n_nodes, 4, topk_c4,
        abb_mode=cycle_abb_mode,
        abb_fullness_gate=cycle_abb_fullness_gate,
    )
    out["c4"] = (c4, s4)
    w2_list = construct_walks(g, walk_len=2, max_walks=max_walks_w2, seed=walk_seed)
    out["w2"] = _pack_signed_ntuples(w2_list, k=3, cap=max_walks_w2, seed=subsample_walks_seed)
    w3_list = construct_walks(g, walk_len=3, max_walks=max_walks_w3, seed=walk_seed + 1)
    out["w3"] = _pack_signed_ntuples(w3_list, k=4, cap=max_walks_w3, seed=subsample_walks_seed + 1)
    return out


__all__ = [
    "JOINT_BA_SLOTS",
    "SLOT_K",
    "build_joint_ba_pools",
]
