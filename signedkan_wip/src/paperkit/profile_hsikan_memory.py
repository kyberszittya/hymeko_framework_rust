"""Stage-by-stage GPU memory profile of HSiKAN-on-Epinions.

Prints peak GPU memory after each setup phase (cycle enum,
M_e/M_vt construction, model init, first forward, first backward,
opt step) plus the top tensors by storage at peak.

Goal: localise the 5.7 GB working set to specific lines in the
training loop, so we can target Triton kernels / gradient
checkpointing / smaller dtype precisely.

Usage:
    HSIKAN_CYCLE_BATCH=10000 HSIKAN_ARITIES=2,3,4 \\
    HSIKAN_MAX_K2=80000 HSIKAN_MAX_K3=15000 \\
    python -m signedkan_wip.src.paperkit.profile_hsikan_memory \\
        --dataset epinions --hidden 16 --max-k4 30000 --seed 0
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def _mem(label: str):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"  [mem] {label:>32s}  alloc={alloc:5.2f} GiB  "
          f"reserved={reserved:5.2f} GiB  peak={peak:5.2f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="epinions")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--max-k4", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=3,
                    help="Few epochs is enough to expose peak memory.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARN: no CUDA available, profile will be empty")

    torch.cuda.reset_peak_memory_stats()
    _mem("startup")

    from ..datasets import load, split
    from ..hyperedges import construct
    from ..n_tuples import construct_k, construct_2
    from ..signedkan import build_vertex_triad_incidence
    from ..mixed_arity_signedkan import (MixedAritySignedKAN,
                                          MixedAritySignedKANConfig,
                                          subsample_tuples)
    from ..signedkan import MultiLayerSignedKANConfig

    g = load(args.dataset)
    print(f"  {args.dataset}: {g.n_nodes} nodes, "
          f"{g.edges.shape[0]} edges")
    tr_idx, _, te_idx = split(g, seed=args.seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    y_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)

    from ..runtime_config import get_runtime
    _train = get_runtime().training
    arities = _train.arities if _train.arities != (3,) else (3, 4)
    max_k2  = _train.max_k2
    max_k3  = _train.max_k3
    cap_dict = {2: max_k2, 3: max_k3,
                  4: args.max_k4, 5: args.max_k4}
    print(f"  arities={arities}  caps={cap_dict}  hidden={args.hidden}")

    _mem("after dataset load")

    # Cycle enumeration.
    per_arity_tr, per_arity_te = [], []
    for k in arities:
        t0 = time.time()
        if k == 2:
            t_k = construct_2(g)
        elif k == 3:
            t_k = construct(g)
        else:
            t_k = construct_k(g, k=k, max_cycles=cap_dict[k],
                               seed=args.seed)
        if not t_k:
            continue
        if len(t_k) > cap_dict[k]:
            t_k = subsample_tuples(t_k, cap_dict[k], seed=args.seed)
        triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in t_k],
                                    dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        n_t = len(t_k)
        print(f"  k={k}: {n_t} cycles ({time.time()-t0:.1f}s)")
        _mem(f"k={k} triad_v + triad_sigma")

        # M_vt
        M_vt = build_vertex_triad_incidence(triad_v_np, g.n_nodes,
                                              device, mode="sum")
        _mem(f"k={k} M_vt built")

        # M_e
        edge_to_tuples = {}
        edge_to_self_idx_k = {}
        for ti in range(n_t):
            cyc = triad_v_np[ti]
            if k == 2:
                key2 = (min(int(cyc[0]), int(cyc[1])),
                         max(int(cyc[0]), int(cyc[1])))
                edge_to_self_idx_k[key2] = ti
            for j in range(k):
                u_, v_ = int(cyc[j]), int(cyc[(j + 1) % k])
                key = (min(u_, v_), max(u_, v_))
                edge_to_tuples.setdefault(key, []).append(ti)

        def build_me(edges_arr):
            rows, cols, vals = [], [], []
            for ei, e in enumerate(edges_arr):
                u_, v_ = int(e[0]), int(e[1])
                key = (min(u_, v_), max(u_, v_))
                ids = edge_to_tuples.get(key, [])
                if k == 2:
                    self_t = edge_to_self_idx_k.get(key)
                    ids = [t for t in ids if t != self_t]
                if not ids:
                    continue
                w = 1.0 / float(len(ids))
                for t in ids:
                    rows.append(ei); cols.append(int(t)); vals.append(w)
            if not rows:
                return torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0,)),
                    (edges_arr.shape[0], n_t),
                ).to(device)
            idx = torch.tensor([rows, cols], dtype=torch.long,
                                device=device)
            v = torch.tensor(vals, dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(
                idx, v, (edges_arr.shape[0], n_t),
            ).coalesce()

        M_e_tr = build_me(e_tr)
        M_e_te = build_me(e_te)
        per_arity_tr.append((triad_v, triad_sigma, M_vt, M_e_tr))
        per_arity_te.append((triad_v, triad_sigma, M_vt, M_e_te))
        _mem(f"k={k} M_e built (nnz_tr={M_e_tr._nnz()})")

    cycle_batch = get_runtime().training.cycle_batch
    print(f"  cycle_batch={cycle_batch}")

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2, hidden_dim=args.hidden,
            grid=3, k=3, spline_kinds=["catmull_rom"]*2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True),
        arities=arities[:len(per_arity_tr)],
        init_arity_logits=tuple([0.0]*len(per_arity_tr)),
        cycle_batch_size=cycle_batch,
    )
    model = MixedAritySignedKAN(cfg).to(device)
    _mem("model init")

    import torch.nn as nn
    clf = nn.Linear(args.hidden * 2, 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()),
                            lr=5e-3)
    n_pos = int(y_tr.sum().item()); n_neg = int((1 - y_tr).sum().item())
    pw = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                       device=device)

    # Reset peak so we measure forward-pass-only peak.
    torch.cuda.reset_peak_memory_stats()
    _mem("right before training")

    for ep in range(args.n_epochs):
        model.train(); clf.train()
        torch.cuda.reset_peak_memory_stats()

        edge_emb = model.encode_edges(per_arity_tr)
        _mem(f"ep={ep} after forward")

        logits = clf(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr,
                                                    pos_weight=pw)
        _mem(f"ep={ep} after loss")

        opt.zero_grad()
        loss.backward()
        _mem(f"ep={ep} after backward")

        opt.step()
        _mem(f"ep={ep} after opt step")

    print("\n──────  per-tensor breakdown of resident GPU memory  ──────")
    # Top-N tensors by storage size.
    from collections import defaultdict
    by_size = []
    for obj in [model, clf]:
        for name, p in obj.named_parameters():
            by_size.append((p.element_size() * p.numel(), name,
                             tuple(p.shape), str(p.dtype)))
        for name, b in obj.named_buffers():
            by_size.append((b.element_size() * b.numel(),
                             f"(buf){name}", tuple(b.shape),
                             str(b.dtype)))
    # Per-arity buffers.
    for ai, (tv, ts, mvt, me) in enumerate(per_arity_tr):
        by_size.append((tv.element_size() * tv.numel(),
                         f"per_arity[{ai}].triad_v",
                         tuple(tv.shape), str(tv.dtype)))
        by_size.append((ts.element_size() * ts.numel(),
                         f"per_arity[{ai}].triad_sigma",
                         tuple(ts.shape), str(ts.dtype)))
        # Sparse buffers — approximate.
        if mvt.is_sparse_csr:
            sz = (mvt.crow_indices().element_size()
                  * mvt.crow_indices().numel()
                  + mvt.col_indices().element_size()
                  * mvt.col_indices().numel()
                  + mvt.values().element_size()
                  * mvt.values().numel())
            by_size.append((sz, f"per_arity[{ai}].M_vt(sparse_csr)",
                             tuple(mvt.shape), "sparse_csr"))
        if me.is_sparse:
            sz = (me._indices().element_size() * me._indices().numel()
                  + me._values().element_size() * me._values().numel())
            by_size.append((sz, f"per_arity[{ai}].M_e_tr(sparse_coo)",
                             tuple(me.shape), "sparse_coo"))

    by_size.sort(reverse=True)
    for sz, name, shape, dtype in by_size[:20]:
        mb = sz / (1024 ** 2)
        print(f"  {mb:8.2f} MiB  {name:>40s}  {shape}  {dtype}")


if __name__ == "__main__":
    main()
