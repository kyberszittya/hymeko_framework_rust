"""Phase-2 mixed-arity HSiKAN — k=3 + k=4 with learnable αₖ.

Default recipe = the leanest config that passed Phase 1
(h=16, G=3, L=2, smooth-only). Per-arity tuple count capped at
``--max_k4`` (default 30k) for memory + runtime.

3 seeds × 2 datasets, plus a k=3-only control at the same recipe
so we can compare "mixed vs k=3-only" cleanly.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .datasets import load, split, deduplicate_pairs
from .hyperedges import construct
from .n_tuples import construct_k, construct_2
from .mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_edge_to_tuples,
                                      build_vertex_to_tuples)
from .signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from .entropy_reg import SplineSmoothRegulariser
from .participation_reg import ParticipationRegulariser, triad_degree


def _build_edge_incidence(edges_array, edge_to_tuples, n_tuples, device,
                            directed: bool = False):
    rows, cols, vals = [], [], []
    for ei, e in enumerate(edges_array):
        u, v = int(e[0]), int(e[1])
        key = (u, v) if directed else (min(u, v), max(u, v))
        ids = edge_to_tuples.get(key, [])
        if not ids:
            continue
        w = 1.0 / float(len(ids))
        for t in ids:
            rows.append(ei); cols.append(int(t)); vals.append(w)
    if not rows:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,)),
            (edges_array.shape[0], n_tuples),
        ).to(device)
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    v = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        idx, v, (edges_array.shape[0], n_tuples),
    ).coalesce()


def _build_edge_incidence_vertex_adj_scipy(edges_array, vertex_to_tuples,
                                              edge_to_self_idx, n_tuples,
                                              device, n_nodes,
                                              exclude_self: bool = True):
    """Vertex-adjacency M_e via scipy sparse mat-mul.

    The line-graph-style incidence is exactly the sparse product
    ``Q @ B`` where:
        Q[q, v] = 1 if v is an endpoint of query edge q   (E_q × n_nodes)
        B[v, t] = 1 if v is in cycle/hyperedge t          (n_nodes × T)

    The output ``(Q @ B)[q, t]`` = # vertex matches between query q
    and cycle t. Binarising and row-normalising gives M_e in one
    sparse mm + a few O(nnz) post-passes.

    scipy.sparse.csr × csr uses tuned C-level SpGEMM; on Bitcoin
    k=4 (17M nnz output) this is 5-20× faster than the per-query
    Python loop and uses far less peak Python memory.
    """
    from scipy.sparse import csr_matrix
    import numpy as _np

    e_q = edges_array.shape[0]
    if e_q == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,)),
            (0, n_tuples),
        ).to(device)

    # Build B: vertex-to-tuple sparse (n_nodes, T).
    b_rows: list[int] = []
    b_cols: list[int] = []
    for v, ts in vertex_to_tuples.items():
        b_rows.extend([v] * len(ts))
        b_cols.extend(ts)
    b_data = _np.ones(len(b_rows), dtype=_np.float32)
    B = csr_matrix(
        (b_data, (_np.asarray(b_rows, dtype=_np.int64),
                   _np.asarray(b_cols, dtype=_np.int64))),
        shape=(n_nodes, n_tuples),
    )

    # Build Q: query-edge-to-vertex sparse (E_q, n_nodes).
    edges_int = edges_array.astype(_np.int64, copy=False)
    q_rows = _np.concatenate([_np.arange(e_q), _np.arange(e_q)])
    q_cols = _np.concatenate([edges_int[:, 0], edges_int[:, 1]])
    q_data = _np.ones(2 * e_q, dtype=_np.float32)
    Q = csr_matrix((q_data, (q_rows, q_cols)),
                    shape=(e_q, n_nodes))

    # M = Q @ B → sparse (E_q, T). Entries are # vertex matches; binarise.
    M = Q @ B
    M.data = _np.ones_like(M.data, dtype=_np.float32)

    # Self-edge exclusion (only fires for k=2, where edge_to_self_idx
    # is non-empty and a query edge can also appear as its own tuple).
    if exclude_self and edge_to_self_idx:
        # Collect (row, col) pairs to zero out, then eliminate_zeros.
        ei_arr = _np.arange(e_q)
        u_arr = edges_int[:, 0]
        v_arr = edges_int[:, 1]
        u_lo = _np.minimum(u_arr, v_arr)
        v_hi = _np.maximum(u_arr, v_arr)
        # Look up each query's self-tuple via the dict (small overhead
        # because k=2 only).
        zero_rows: list[int] = []
        zero_cols: list[int] = []
        for ei in range(e_q):
            self_t = edge_to_self_idx.get(
                (int(u_lo[ei]), int(v_hi[ei]))
            )
            if self_t is not None:
                zero_rows.append(ei); zero_cols.append(self_t)
        if zero_rows:
            mask = csr_matrix(
                (_np.ones(len(zero_rows), dtype=_np.float32),
                 (_np.asarray(zero_rows, dtype=_np.int64),
                  _np.asarray(zero_cols, dtype=_np.int64))),
                shape=(e_q, n_tuples),
            )
            M = (M - mask).maximum(0)
            M.eliminate_zeros()

    # Row-normalise: each query row by its nnz.
    M = M.tocsr()
    nnz_per_row = _np.diff(M.indptr).astype(_np.float32)
    nnz_per_row[nnz_per_row == 0] = 1.0
    inv = 1.0 / nnz_per_row
    # Repeat inv per row so each entry of M.data gets scaled by 1/|row|.
    counts = _np.diff(M.indptr).astype(_np.int64)
    M.data = M.data * _np.repeat(inv, counts)

    # Return as CSR — torch.sparse.mm dispatches to a much faster
    # cuSPARSE CSR kernel for sparse-dense mm. COO mm was 71ms / forward
    # on Slashdot; CSR is 2-5× faster.
    M_csr = M.tocsr()
    crow = torch.from_numpy(M_csr.indptr.astype(_np.int64)).to(device)
    col  = torch.from_numpy(M_csr.indices.astype(_np.int64)).to(device)
    val  = torch.from_numpy(M_csr.data).to(device)
    return torch.sparse_csr_tensor(
        crow, col, val, (e_q, n_tuples),
    )


def _build_edge_incidence_vertex_adj(edges_array, vertex_to_tuples,
                                       edge_to_self_idx, n_tuples, device,
                                       exclude_self: bool = True,
                                       n_nodes: int | None = None):
    """Line-graph-style adjacency for arbitrary k: M_e[e, t] = 1/|N(e)|
    if cycle/hyperedge t shares at least one endpoint with query edge
    e (i.e., one of t's vertices is u or v from e=(u,v)). When
    ``exclude_self=True`` and a tuple corresponds exactly to e (only
    possible at k=2), exclude it to prevent the σ-as-label leak.

    The σ values inside each cycle still depend on the cycle's own
    edges, not the query edge — so this M_e construction *removes the
    σ-as-label leak* for k≥3 as well: query edge's sign never enters
    any cycle's σ pattern.

    Vectorised numpy implementation. Builds vertex_to_tuples as a CSR
    structure (row_ptr + col_idx) once, then for each query uses
    ``np.unique(np.concatenate(...))`` instead of Python set-union.
    Empirically ~10-30× faster than the dict-based loop for ≥10k cycles
    on Bitcoin/Slashdot-scale graphs.

    ``vertex_to_tuples``: dict[vertex_id → list of tuple indices].
    ``edge_to_self_idx``: dict[(u,v) sorted → tuple index of t==e],
                          only meaningful for k=2 hyperedges.
    ``n_nodes``: required for CSR construction when not all vertices
                 appear in vertex_to_tuples.
    """
    if n_nodes is None:
        if vertex_to_tuples:
            n_nodes = max(vertex_to_tuples.keys()) + 1
        else:
            n_nodes = 0

    # Build CSR rep of vertex_to_tuples.
    n_per_vertex = np.zeros(n_nodes, dtype=np.int64)
    for v, ts in vertex_to_tuples.items():
        n_per_vertex[v] = len(ts)
    row_ptr = np.empty(n_nodes + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(n_per_vertex, out=row_ptr[1:])
    total = int(row_ptr[-1])
    col_idx = np.empty(total, dtype=np.int64)
    for v, ts in vertex_to_tuples.items():
        s = int(row_ptr[v])
        col_idx[s : s + len(ts)] = ts

    rows_chunks: list[np.ndarray] = []
    cols_chunks: list[np.ndarray] = []
    vals_chunks: list[np.ndarray] = []
    edges_int = edges_array.astype(np.int64, copy=False)
    for ei in range(edges_int.shape[0]):
        u = int(edges_int[ei, 0])
        v = int(edges_int[ei, 1])
        adj_u = col_idx[row_ptr[u]:row_ptr[u + 1]]
        adj_v = col_idx[row_ptr[v]:row_ptr[v + 1]]
        # np.unique(concatenate) is faster than np.union1d on small arrays.
        adj = np.unique(np.concatenate([adj_u, adj_v]))
        if exclude_self:
            key = (min(u, v), max(u, v))
            self_t = edge_to_self_idx.get(key)
            if self_t is not None:
                adj = adj[adj != self_t]
        n_adj = adj.shape[0]
        if n_adj == 0:
            continue
        w = 1.0 / float(n_adj)
        rows_chunks.append(np.full(n_adj, ei, dtype=np.int64))
        cols_chunks.append(adj)
        vals_chunks.append(np.full(n_adj, w, dtype=np.float32))

    if not rows_chunks:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,)),
            (edges_array.shape[0], n_tuples),
        ).to(device)
    rows_np = np.concatenate(rows_chunks)
    cols_np = np.concatenate(cols_chunks)
    vals_np = np.concatenate(vals_chunks)
    idx = torch.from_numpy(np.stack([rows_np, cols_np])).to(device)
    v = torch.from_numpy(vals_np).to(device)
    return torch.sparse_coo_tensor(
        idx, v, (edges_array.shape[0], n_tuples),
    ).coalesce()


# Back-compat alias (k=2-specific wrapper).
_build_edge_incidence_k2 = _build_edge_incidence_vertex_adj


def _evaluate(model, per_arity_inputs, edges, signs, device):
    model.eval()
    e_t = torch.from_numpy(edges.astype(np.int64)).to(device)
    with torch.no_grad():
        edge_emb = model.encode_edges(per_arity_inputs, query_edges=e_t)
        # ensure indexing matches: edge_emb is over the train/val/test
        # edge tensor we passed M_e for. Caller wraps each split.
        logits = model.classifier(edge_emb).squeeze(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y = (signs == 1).astype(int)
    auc = (roc_auc_score(y, probs)
           if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m)


def run_one_mixed(dataset: str, seed: int,
                   hidden: int = 16, n_layers: int = 2, grid: int = 3,
                   n_epochs: int = 120, lr: float = 5e-2,
                   weight_decay: float = 1e-4, grad_clip: float = 1.0,
                   coef_smooth_lam: float = 0.010,
                   participation_lam: float = 0.05,
                   max_k4: int = 30000,
                   max_k3: int = 0,           # 0 = no subsample
                   max_k5: int = 30000,
                   max_per_arity: dict | None = None,
                   arities: tuple[int, ...] = (3, 4),
                   only_k3: bool = False,
                   early_stopping: bool = True,
                   class_weighted: bool = True,
                   val_every: int = 5,
                   cycle_batch_size: int | None = None,
                   spectral_init_eigvec=None,
                   directed: bool = False,
                   directed_m_e: bool | None = None,
                   feature_edges: str = "all",
                   lr_schedule: str = "fixed",
                   m_e_mode: str = "edge_in_cycle",
                   dedupe_pairs: bool = False,
                   dedupe_merge: str = "majority",
                   cycle_early_stop: bool = False,
                   balance_lambda: float = 0.0,
                   attention_m_e: bool = False,
                   multitask_lambda: float = 0.0,
                   direct_messaging: bool = False) -> dict:
    """``m_e_mode``       — how M_e (edge-cycle incidence) is built.
        - "edge_in_cycle" (default): M_e[query, t] = 1 iff query edge
          appears as one of cycle t's cycle-edges. The σ assignment of
          t therefore depends on the query edge's sign — *known
          σ-as-label leak* in transductive evaluation.
        - "vertex_adjacency": M_e[query, t] = 1/|N(query)| iff cycle t
          shares an endpoint with the query edge. The query edge is
          NEVER in cycle t's edge set, so its sign cannot enter t's σ
          pattern. Removes the σ-as-label leak structurally for k≥3
          (and matches the existing k=2 line-graph adjacency)."""
    """``directed``        — enumerate *directed* k-cycles. Each cycle's
                              σ assignment encodes the signs of its
                              specific directed edges, giving direction-
                              aware structural features.
    ``directed_m_e``      — if True, M_e[query, cycle] = 1 iff the
                              query directed edge ``(u, v)`` matches
                              a *directed* cycle edge in the same
                              direction. If False (default when
                              ``directed=True`` and ``feature_edges``
                              != "all"), M_e uses undirected sorted-
                              pair keys so transductive queries over
                              held-out edges still find incident
                              cycles. Defaults to ``directed`` when
                              ``feature_edges == 'all'`` (preserve the
                              old behaviour) and ``False`` otherwise.
    ``feature_edges``     — which edges feed cycle construction.
        - "all"       → cycles from g.edges (transductive, leaky)
        - "train_val" → cycles from train+val edges (test held out)
        - "train"     → cycles from train edges only (val+test held out)
    """
    """One mixed-arity run. ``only_k3=True`` falls back to single-arity
    k=3 (uses the same wrapper but with arities=(3,))."""
    if only_k3:
        arities = (3,)

    # M_e direction default: match `directed` ONLY when cycles are
    # built from the full graph (transductive leaky baseline). With
    # held-out test edges, strict-directed M_e collapses to AUC=0.5
    # because no remaining cycle contains the held-out directed edge,
    # so default to undirected query lookup.
    if directed_m_e is None:
        directed_m_e = directed and (feature_edges == "all")

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load(dataset)
    if dedupe_pairs:
        # Strict leak-free protocol: collapse duplicate (u,v) pairs
        # BEFORE splitting so no held-out pair survives in g_features.
        g = deduplicate_pairs(g, merge=dedupe_merge)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_va, s_va = g.edges[va_idx], g.signs[va_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    # Cycle-feature graph: optionally hold out val/test edges so the
    # cycles used to predict them don't structurally encode the answer.
    if feature_edges == "all":
        g_features = g
    else:
        from .datasets import SignedGraph
        if feature_edges == "train_val":
            feat_idx = np.concatenate([tr_idx, va_idx])
        elif feature_edges == "train":
            feat_idx = np.array(tr_idx)
        else:
            raise ValueError(
                f"feature_edges must be all|train_val|train, got "
                f"{feature_edges!r}"
            )
        g_features = SignedGraph(
            edges=g.edges[feat_idx],
            signs=g.signs[feat_idx],
            n_nodes=g.n_nodes,
        )

    # Per-arity tuple lists.
    # We pass ``max_cycles`` into ``construct_k`` so subsampling
    # happens BEFORE classification — for large graphs (Slashdot k=4)
    # materialising all cycles as SignedNTuple objects would OOM.
    per_arity_tuples = []
    for k in arities:
        cap = None
        if max_per_arity is not None and k in max_per_arity:
            cap = max_per_arity[k]
        elif k == 3 and max_k3 > 0:
            cap = max_k3
        elif k == 4:
            cap = max_k4
        elif k == 5:
            cap = max_k5
        if k == 2:
            t_k = construct_2(g_features)
        elif k == 3:
            # construct() is undirected-only; for directed k=3 we go
            # through construct_k which threads the flag.
            t_k = (construct(g_features) if not directed
                    else construct_k(g_features, k=3, max_cycles=cap,
                                       seed=seed, directed=True,
                                       early_stop=cycle_early_stop))
        else:
            t_k = construct_k(g_features, k=k, max_cycles=cap, seed=seed,
                                directed=directed,
                                early_stop=cycle_early_stop)
        # Re-apply subsample at SignedNTuple level for k=3 (which
        # doesn't go through construct_k) and as a safety net for any
        # cycles dropped during classification.
        if cap is not None and cap > 0 and len(t_k) > cap:
            t_k = subsample_tuples(t_k, cap, seed=seed)
        per_arity_tuples.append(t_k)

    # Drop arities that produced zero cycles (unsupported on this
    # graph). Without this, the per-arity input construction trips on
    # empty (n=0) triad arrays. Issue a hint so callers know.
    nonempty = [(k, t) for k, t in zip(arities, per_arity_tuples) if len(t) > 0]
    if len(nonempty) < len(arities):
        dropped = [k for k, t in zip(arities, per_arity_tuples) if len(t) == 0]
        print(f"[run_one_mixed] dropping arities with 0 cycles on this "
              f"graph: {dropped}")
    if not nonempty:
        raise ValueError(
            f"all arities {arities} have 0 cycles on dataset {dataset!r}; "
            f"check graph connectivity or pick smaller arities."
        )
    arities = tuple(k for k, _ in nonempty)
    per_arity_tuples = [t for _, t in nonempty]

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=n_layers,
            hidden_dim=hidden, grid=grid, k=3,    # base k=3 for default,
                                                   # the layer is arity-agnostic
            spline_kinds=["catmull_rom"] * n_layers,
            init_scale=0.05,
            pool_mode="sum",
            jk_mode="concat",
            layer_norm_between=True,
            share_weights=True,
            inner_skip="highway",
            outer_skip="none",
            use_residual=True,
            spectral_init_eigvec=(spectral_init_eigvec.to(device)
                                   if spectral_init_eigvec is not None
                                   else None),
        ),
        arities=arities,
        init_arity_logits=tuple([0.0] * len(arities)),
        cycle_batch_size=cycle_batch_size,
        attention_m_e=attention_m_e,
        direct_messaging=direct_messaging,
    )
    model = MixedAritySignedKAN(cfg).to(device)
    n_params = model.num_parameters()

    # Build per-sign signed adjacency (D^-1 A_signed-flavoured) from
    # the train edges, for the SGCN-style direct messaging path.
    if direct_messaging:
        n = g.n_nodes
        # Symmetrise: each train edge contributes to both directions.
        u_arr = e_tr[:, 0].astype(np.int64)
        v_arr = e_tr[:, 1].astype(np.int64)
        s_arr = s_tr.astype(np.int64)
        pos_mask = s_arr == 1
        neg_mask = s_arr == -1
        # Positive edges (symmetric).
        rows_p = np.concatenate([u_arr[pos_mask], v_arr[pos_mask]])
        cols_p = np.concatenate([v_arr[pos_mask], u_arr[pos_mask]])
        # Negative edges (symmetric).
        rows_n = np.concatenate([u_arr[neg_mask], v_arr[neg_mask]])
        cols_n = np.concatenate([v_arr[neg_mask], u_arr[neg_mask]])
        # Row-degree normalise (mean aggregation): D^-1 A.
        deg_p = np.zeros(n, dtype=np.float32)
        deg_n = np.zeros(n, dtype=np.float32)
        np.add.at(deg_p, rows_p, 1.0)
        np.add.at(deg_n, rows_n, 1.0)
        deg_p = np.maximum(deg_p, 1.0)
        deg_n = np.maximum(deg_n, 1.0)
        vals_p = 1.0 / deg_p[rows_p]
        vals_n = 1.0 / deg_n[rows_n]
        idx_p = torch.tensor(np.stack([rows_p, cols_p]),
                              dtype=torch.long, device=device)
        idx_n = torch.tensor(np.stack([rows_n, cols_n]),
                              dtype=torch.long, device=device)
        A_pos = torch.sparse_coo_tensor(
            idx_p, torch.from_numpy(vals_p).to(device),
            (n, n),
        ).coalesce()
        A_neg = torch.sparse_coo_tensor(
            idx_n, torch.from_numpy(vals_n).to(device),
            (n, n),
        ).coalesce()
        model.set_signed_adjacency(A_pos, A_neg)

    # Per-arity inputs (triad_v, triad_sigma, M_vt, M_edge_*).
    per_arity_train = []
    per_arity_val   = []
    per_arity_test  = []
    for ai, k in enumerate(arities):
        tuples = per_arity_tuples[ai]
        triad_v_np = np.array([t.v for t in tuples], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in tuples], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        n_tuples = len(tuples)
        M_vt = build_vertex_triad_incidence(
            triad_v_np, g.n_nodes, device, mode="sum",
        )
        if k == 2 or m_e_mode == "vertex_adjacency":
            # Line-graph-style incidence (vertex adjacency); for k=2
            # this is the existing path. For k≥3 with m_e_mode set,
            # it removes the σ-as-label leak by ensuring the query
            # edge is never in any cycle's σ-defining edge set.
            v2t = build_vertex_to_tuples(tuples)
            self_idx = {}
            for ti, t in enumerate(tuples):
                if len(t.v) == 2:
                    u, w = int(t.v[0]), int(t.v[1])
                    self_idx[(min(u, w), max(u, w))] = ti
            M_e_tr = _build_edge_incidence_vertex_adj_scipy(
                e_tr, v2t, self_idx, n_tuples, device,
                n_nodes=g.n_nodes,
            )
            M_e_va = _build_edge_incidence_vertex_adj_scipy(
                e_va, v2t, self_idx, n_tuples, device,
                n_nodes=g.n_nodes,
            )
            M_e_te = _build_edge_incidence_vertex_adj_scipy(
                e_te, v2t, self_idx, n_tuples, device,
                n_nodes=g.n_nodes,
            )
        else:
            edge_to_tuples = build_edge_to_tuples(tuples, directed=directed_m_e)
            M_e_tr = _build_edge_incidence(e_tr, edge_to_tuples, n_tuples, device, directed=directed_m_e)
            M_e_va = _build_edge_incidence(e_va, edge_to_tuples, n_tuples, device, directed=directed_m_e)
            M_e_te = _build_edge_incidence(e_te, edge_to_tuples, n_tuples, device, directed=directed_m_e)
        per_arity_train.append((triad_v, triad_sigma, M_vt, M_e_tr))
        per_arity_val.append(  (triad_v, triad_sigma, M_vt, M_e_va))
        per_arity_test.append( (triad_v, triad_sigma, M_vt, M_e_te))

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                            weight_decay=weight_decay)
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs,
        )
    elif lr_schedule == "fixed":
        sched = None
    else:
        raise ValueError(f"unknown lr_schedule: {lr_schedule}")

    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    if class_weighted:
        n_pos = int((s_tr ==  1).sum())
        n_neg = int((s_tr == -1).sum())
        pos_weight = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                                   device=device)
    else:
        pos_weight = None

    smooth_reg = (SplineSmoothRegulariser(coef_smooth_lam)
                   if coef_smooth_lam > 0 else None)
    if participation_lam > 0:
        part_reg = ParticipationRegulariser(lam=participation_lam).to(device)
        # Participation weights derived from k=3 triads (the structural
        # backbone). Mixed-arity does not redefine the per-vertex
        # degree.
        deg_np = triad_degree(per_arity_tuples[0], g.n_nodes)
        part_reg.set_degrees(deg_np)
    else:
        part_reg = None

    # Pre-compute query-edge tensor for attention M_e (no-op otherwise).
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)

    # Multi-task auxiliary targets: per-vertex signed degree
    # (n_pos - n_neg) / max(1, total) for each train edge endpoint.
    # Predicting this from edge_emb encourages structural awareness
    # of vertex sign-distributions.
    aux_head = None
    if multitask_lambda > 0.0:
        # Compute per-vertex signed-degree from train graph.
        n = g.n_nodes
        n_pos = np.zeros(n, dtype=np.float32)
        n_neg = np.zeros(n, dtype=np.float32)
        for (u, v), s in zip(e_tr, s_tr):
            if s > 0:
                n_pos[int(u)] += 1; n_pos[int(v)] += 1
            else:
                n_neg[int(u)] += 1; n_neg[int(v)] += 1
        sd = (n_pos - n_neg) / np.maximum(1.0, n_pos + n_neg)
        sd_t = torch.from_numpy(sd).to(device)
        # Per-edge targets: stack (sd[u], sd[v]).
        sd_u_tr = sd_t[e_tr_t[:, 0]]
        sd_v_tr = sd_t[e_tr_t[:, 1]]
        aux_target_tr = torch.stack([sd_u_tr, sd_v_tr], dim=-1)
        # Aux head: edge_emb → (2,) regression.
        d_jk = (cfg.base.hidden_dim * cfg.base.n_layers
                if cfg.base.jk_mode == "concat" else cfg.base.hidden_dim)
        aux_head = nn.Linear(d_jk, 2).to(device)
        # Re-create optimizer to include aux head params.
        opt = torch.optim.Adam(
            list(model.parameters()) + list(aux_head.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        if lr_schedule == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs,
            )

    best_val_auc, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        edge_emb = model.encode_edges(per_arity_train, query_edges=e_tr_t)
        logits = model.classifier(edge_emb).squeeze(-1)
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, target_tr, pos_weight=pos_weight,
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, target_tr)
        if smooth_reg is not None:
            loss = loss + smooth_reg(model.base)
        if part_reg is not None:
            loss = loss + part_reg(model.node_embed.weight)
        if balance_lambda > 0.0:
            # Cartwright-Harary balance loss: positive edges should have
            # close vertex embeddings (cos→1); negative edges should have
            # opposite vertex embeddings (cos→-1). Standard auxiliary loss
            # in signed-graph NNs (SGCN, SiGAT use variants).
            h = model.node_embed.weight
            u_idx = torch.from_numpy(e_tr[:, 0].astype(np.int64)).to(device)
            v_idx = torch.from_numpy(e_tr[:, 1].astype(np.int64)).to(device)
            h_u = h[u_idx]; h_v = h[v_idx]
            cos = F.cosine_similarity(h_u, h_v, dim=-1)
            sign = torch.from_numpy(s_tr.astype(np.float32)).to(device)
            # For sign=+1: minimise (1 - cos);  for sign=-1: minimise (1 + cos).
            l_balance = (1.0 - sign * cos).mean()
            loss = loss + balance_lambda * l_balance
        if aux_head is not None:
            # Multi-task auxiliary: predict signed-degree of each edge's
            # endpoints from edge_emb. MSE loss, weighted by
            # multitask_lambda.
            sd_pred = aux_head(edge_emb)            # (E_tr, 2)
            l_aux = F.mse_loss(sd_pred, aux_target_tr)
            loss = loss + multitask_lambda * l_aux
        opt.zero_grad(); loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            max_norm=grad_clip)
        opt.step()
        if sched is not None:
            sched.step()

        if early_stopping and (
            (epoch + 1) % val_every == 0 or epoch == n_epochs - 1
        ):
            v_auc, _ = _evaluate(model, per_arity_val, e_va, s_va, device)
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
    elapsed = time.time() - t0

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    test_auc, test_f1m = _evaluate(model, per_arity_test, e_te, s_te, device)

    alpha = model.alpha().detach().cpu().tolist()
    return dict(
        model="hsikan_mixed", dataset=dataset,
        hidden=hidden, n_layers=n_layers, grid=grid,
        seed=seed, n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, grad_clip=grad_clip,
        coef_smooth_lam=coef_smooth_lam,
        participation_lam=participation_lam,
        arities=list(arities),
        max_k4=max_k4,
        n_params=n_params, elapsed_s=elapsed,
        best_epoch=best_epoch, best_val_auc=best_val_auc,
        test_auc=test_auc, test_f1_macro=test_f1m,
        alpha=alpha,
        n_tuples_per_arity={str(k): len(tup)
                             for k, tup in zip(arities, per_arity_tuples)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--max_k4", type=int, default=30000)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase2_mixed.json")
    args = ap.parse_args()

    runs = []
    t_total = time.time()
    # k=3-only control + (k=3, k=4) mixed.
    for variant in ("k3_only", "k3_k4"):
        only_k3 = (variant == "k3_only")
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one_mixed(
                    dataset, seed,
                    hidden=args.hidden, n_layers=args.n_layers,
                    grid=args.grid, n_epochs=args.n_epochs,
                    max_k4=args.max_k4, only_k3=only_k3,
                )
                r["variant"] = variant
                print(f"  {variant:8s}  {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  "
                      f"F1m={r['test_f1_macro']:.4f}  "
                      f"params={r['n_params']:>7,}  "
                      f"alpha={r['alpha']}  "
                      f"{r['elapsed_s']:.1f}s")
                runs.append(r)

    summary = {}
    for variant in ("k3_only", "k3_k4"):
        for dataset in args.datasets:
            cell = [r for r in runs
                     if r["variant"] == variant and r["dataset"] == dataset]
            aucs = [r["test_auc"] for r in cell]
            f1ms = [r["test_f1_macro"] for r in cell]
            elap = [r["elapsed_s"] for r in cell]
            summary[f"{variant}|{dataset}"] = {
                "auc_med":   round(statistics.median(aucs), 4),
                "f1m_med":   round(statistics.median(f1ms), 4),
                "elapsed_med_s": round(statistics.median(elap), 2),
                "n_params":  cell[0]["n_params"],
                "auc_seeds": [round(a, 4) for a in aucs],
                "f1m_seeds": [round(f, 4) for f in f1ms],
            }
    out = {
        "runs": runs,
        "summary": summary,
        "wall_clock_s": round(time.time() - t_total, 1),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path} ({len(runs)} runs in "
          f"{out['wall_clock_s']:.1f}s)")


if __name__ == "__main__":
    main()
