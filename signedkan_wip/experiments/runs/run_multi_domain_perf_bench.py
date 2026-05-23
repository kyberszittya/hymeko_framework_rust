"""Multi-domain HSiKAN performance bench (accuracy + inference latency).

Five domains:
  bitcoin   — Bitcoin Alpha/OTC + Slashdot edge sign prediction
              (reuses results/inference_bench.json from the optimised
              bitcoin pipeline, since training cost dwarfs the
              inference-time question on those datasets).
  sbm       — synthetic stochastic-block-model signed graphs
              (datasets_small.sbm_signed)
  scene     — synthetic scene-graph relation prediction
              (adapters.visual_genome.synth_dataset, à la phase 15)
  kinematic — graph-level mechanism family classification + DOF
              regression on synthetic kinematic mechanisms
              (à la phase 11)
  pose      — per-vertex XYZ regression on synthetic kinematic
              mechanisms (à la phase 12)

Per cell we report:
  accuracy : domain-appropriate metric (AUC / acc / F1 / MAE / MSE)
  setup_ms : graph build + cycle enum + tensor build (ONE seed)
  fwd_med_ms : median single-forward latency over N_REPEATS calls
                with cuda.synchronize on each side (single-query
                latency in production-style measurement)
  fwd_throughput_ms : average forward latency when N_REPEATS calls
                       share one sync (steady-state throughput)
  n_params : trainable parameters
  n_inputs : test-set size that the timed forward processes

Running:
    HSIKAN_TORCH_COMPILE=1 python -m signedkan_wip.experiments.runs.run_multi_domain_perf_bench

Output:
    signedkan_wip/experiments/results/multi_domain_perf_bench.json
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Domain helpers (reused from existing phase scripts).
from signedkan_wip.src.datasets import SignedGraph, load, deduplicate_pairs, split
from signedkan_wip.src.datasets import sbm_signed, hierarchical_signed
from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.core.n_tuples import construct_k, construct_2
from signedkan_wip.src.mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_vertex_to_tuples,
                                      build_edge_to_tuples)
from signedkan_wip.src.core.signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from signedkan_wip.src.baselines.sgcn_model import SGCN, build_signed_adj


N_WARMUP = 10
N_REPEATS = 30


# -----------------------------------------------------------------------
# Generic timing utilities

def time_per_call(fn, n_warmup=N_WARMUP, n_repeats=N_REPEATS,
                   sync_cuda=True) -> float:
    """Median per-call latency in ms (with sync each side)."""
    for _ in range(n_warmup):
        fn()
        if sync_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
    samples = []
    for _ in range(n_repeats):
        if sync_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def time_throughput(fn, n_warmup=N_WARMUP, n_repeats=N_REPEATS,
                     sync_cuda=True) -> float:
    """Average call latency with ONE sync wrapping all repeats (ms)."""
    for _ in range(n_warmup):
        fn()
    if sync_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        fn()
    if sync_cuda and torch.cuda.is_available(): torch.cuda.synchronize()
    return ((time.perf_counter() - t0) / n_repeats) * 1000


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# Per-arity input builder (shared)

def build_per_arity(g: SignedGraph, arities: tuple, max_per: int,
                      device, seed: int = 0,
                      n_nodes_pad: int | None = None) -> list:
    """Returns list[(triad_v, triad_sigma, M_vt, M_e)] for given arities.

    k=2 falls back to construct_2 (raw edges as 2-uniform hyperedges,
    σ assigned via Davis-style parity at each endpoint). This lets
    domains with no k≥3 cycles (e.g. small synth scene graphs) still
    feed the HSiKAN encoder."""
    out = []
    n_v = n_nodes_pad if n_nodes_pad is not None else g.n_nodes
    for k in arities:
        if k == 2:
            t_k = construct_2(g)
        elif k == 3:
            t_k = construct(g)
        else:
            t_k = construct_k(g, k=k, max_cycles=max_per, seed=seed)
        if not t_k: continue
        if len(t_k) > max_per:
            t_k = subsample_tuples(t_k, max_per, seed=seed)
        triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        M_vt = build_vertex_triad_incidence(triad_v_np, n_v, device, mode="sum")
        edge_to_tuples = build_edge_to_tuples(t_k)
        rows, cols, vals = [], [], []
        for ei, e in enumerate(g.edges):
            key = (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
            ids = edge_to_tuples.get(key, [])
            if not ids: continue
            w = 1.0 / float(len(ids))
            for t in ids:
                rows.append(ei); cols.append(int(t)); vals.append(w)
        if rows:
            idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
            v = torch.tensor(vals, dtype=torch.float32, device=device)
            M_e = torch.sparse_coo_tensor(idx, v,
                                            (g.edges.shape[0], len(t_k))).coalesce()
        else:
            M_e = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,)),
                (g.edges.shape[0], len(t_k))).to(device)
        out.append((triad_v, triad_sigma, M_vt, M_e))
    return out


# -----------------------------------------------------------------------
# Domain 1: SBM (signed)
# Edge-sign prediction (binary) on synthetic sbm_signed graphs.

def domain_sbm(device, n_epochs=30) -> list[dict]:
    print("\n=== domain: sbm ===", flush=True)
    rows = []
    for n_nodes, name in [(200, "sbm_n200"), (400, "sbm_n400")]:
        g, _ = sbm_signed(n_nodes=n_nodes, n_communities=4, seed=0)
        rng = np.random.RandomState(0)
        idx = rng.permutation(g.edges.shape[0])
        n_tr = int(0.7 * len(idx))
        tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
        e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
        e_te, s_te = g.edges[te_idx], g.signs[te_idx]

        # ---- HSiKAN ----
        setup_t0 = time.perf_counter()
        per_arity = build_per_arity(g, (3,), 5000, device, seed=0)
        if not per_arity:
            print(f"  {name}: no cycles, skipping HSiKAN", flush=True); continue
        v2t = build_vertex_to_tuples_from_per_arity(per_arity)
        # build M_e for test edges via simple O(E*k) for cycles
        from .run_phase2_mixed_arity import _build_edge_incidence_vertex_adj_scipy
        # We need vertex_to_tuples per arity — rebuild via existing helper
        triad_v_np = per_arity[0][0].cpu().numpy()
        # build per-tuple cycle list (just use vertex set)
        t_list = [_DummyT(v=tuple(triad_v_np[i].tolist()))
                   for i in range(triad_v_np.shape[0])]
        v2t = build_vertex_to_tuples(t_list)
        M_e_te = _build_edge_incidence_vertex_adj_scipy(
            e_te, v2t, {}, len(t_list), device, n_nodes=g.n_nodes,
        )
        per_arity_test = [(per_arity[0][0], per_arity[0][1],
                            per_arity[0][2], M_e_te)]
        setup_ms = (time.perf_counter() - setup_t0) * 1000

        cfg = MixedAritySignedKANConfig(
            base=MultiLayerSignedKANConfig(
                n_nodes=g.n_nodes, n_layers=2, hidden_dim=16, grid=3, k=3,
                spline_kinds=["catmull_rom"]*2, init_scale=0.05,
                pool_mode="sum", jk_mode="concat",
                layer_norm_between=True, share_weights=True,
                inner_skip="highway", outer_skip="none", use_residual=True),
            arities=(3,), init_arity_logits=(0.0,))
        model = MixedAritySignedKAN(cfg).to(device)
        clf = nn.Linear(16 * 2, 1).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()),
                                lr=5e-3)
        # Train M_e per training-edge set
        M_e_tr = _build_edge_incidence_vertex_adj_scipy(
            e_tr, v2t, {}, len(t_list), device, n_nodes=g.n_nodes,
        )
        per_arity_tr = [(per_arity[0][0], per_arity[0][1],
                          per_arity[0][2], M_e_tr)]
        y_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
        y_te = torch.from_numpy((s_te == 1).astype(np.float32)).to(device)

        for ep in range(n_epochs):
            model.train(); clf.train()
            edge_emb = model.encode_edges(per_arity_tr)
            logits = clf(edge_emb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); clf.eval()
        def fwd():
            with torch.no_grad():
                emb = model.encode_edges(per_arity_test)
                return torch.sigmoid(clf(emb).squeeze(-1))
        with torch.no_grad():
            probs = fwd().cpu().numpy()
        try:
            auc = roc_auc_score(y_te.cpu().numpy(), probs)
        except ValueError:
            auc = float("nan")
        f1 = f1_score(y_te.cpu().numpy(), probs > 0.5, average="macro",
                       zero_division=0)

        per_call = time_per_call(fwd)
        thr = time_throughput(fwd)
        rows.append(dict(
            domain="sbm", subset=name, model="HSiKAN",
            n_nodes=g.n_nodes, n_test=len(te_idx),
            setup_ms=setup_ms,
            accuracy=dict(auc=auc, f1m=f1),
            fwd_per_call_ms=per_call,
            fwd_throughput_ms=thr,
            n_params=n_params(model) + n_params(clf),
        ))
        print(f"  {name} HSiKAN: auc={auc:.3f} f1m={f1:.3f} "
              f"per-call={per_call:.2f}ms thr={thr:.2f}ms n_test={len(te_idx)}",
              flush=True)

        # ---- SGCN baseline ----
        sgcn_setup_t0 = time.perf_counter()
        A_pos, A_neg = build_signed_adj(e_tr, s_tr, g.n_nodes, device)
        sgcn_setup_ms = (time.perf_counter() - sgcn_setup_t0) * 1000
        sgcn = SGCN(n_nodes=g.n_nodes, hidden_dim=32, n_layers=2).to(device)
        opt = torch.optim.Adam(sgcn.parameters(), lr=5e-3)
        e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
        e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)
        for ep in range(n_epochs):
            sgcn.train()
            z = sgcn.encode_nodes(A_pos, A_neg)
            logits = sgcn.edge_logits(z, e_tr_t).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr)
            opt.zero_grad(); loss.backward(); opt.step()
        sgcn.eval()
        def sgcn_fwd():
            with torch.no_grad():
                z = sgcn.encode_nodes(A_pos, A_neg)
                return torch.sigmoid(sgcn.edge_logits(z, e_te_t).squeeze(-1))
        with torch.no_grad():
            sprobs = sgcn_fwd().cpu().numpy()
        try:
            sauc = roc_auc_score(y_te.cpu().numpy(), sprobs)
        except ValueError:
            sauc = float("nan")
        sf1 = f1_score(y_te.cpu().numpy(), sprobs > 0.5, average="macro",
                        zero_division=0)
        sgcn_per_call = time_per_call(sgcn_fwd)
        sgcn_thr = time_throughput(sgcn_fwd)
        rows.append(dict(
            domain="sbm", subset=name, model="SGCN",
            n_nodes=g.n_nodes, n_test=len(te_idx),
            setup_ms=sgcn_setup_ms,
            accuracy=dict(auc=sauc, f1m=sf1),
            fwd_per_call_ms=sgcn_per_call,
            fwd_throughput_ms=sgcn_thr,
            n_params=n_params(sgcn),
        ))
        print(f"  {name} SGCN:   auc={sauc:.3f} f1m={sf1:.3f} "
              f"per-call={sgcn_per_call:.2f}ms thr={sgcn_thr:.2f}ms",
              flush=True)
    return rows


class _DummyT:
    def __init__(self, v): self.v = v


def build_vertex_to_tuples_from_per_arity(per_arity):
    triad_v_np = per_arity[0][0].cpu().numpy()
    t_list = [_DummyT(v=tuple(triad_v_np[i].tolist()))
               for i in range(triad_v_np.shape[0])]
    return build_vertex_to_tuples(t_list)


# -----------------------------------------------------------------------
# Domain 2: Scene graph (synth VG)

def domain_scene(device, n_epochs=30) -> list[dict]:
    print("\n=== domain: scene-graph (synth VG) ===", flush=True)
    from signedkan_wip.src.adapters.visual_genome import (
        synth_dataset, edge_features_from_bboxes,
    )

    rng = random.Random(0)
    np.random.seed(0); torch.manual_seed(0)
    ds_raw = synth_dataset(n_scenes=120, seed=0)
    ds = [(g, vf, sg) for g, vf, sg in ds_raw if g.edges.shape[0] >= 2]

    # Random sign flip 40% so we have both classes
    def flip(g, frac, rng):
        new_signs = g.signs.copy()
        for ei in range(g.edges.shape[0]):
            if rng.random() < frac:
                new_signs[ei] = -1
        return SignedGraph(edges=g.edges, signs=new_signs.astype(np.int8),
                            n_nodes=g.n_nodes)
    ds = [(flip(g, 0.4, rng), vf, sg) for g, vf, sg in ds]

    n_tr = int(0.7 * len(ds))
    tr_scenes = list(range(n_tr))
    te_scenes = list(range(n_tr, len(ds)))
    n_nodes_pad = max(g.n_nodes for g, _, _ in ds)
    d_v = ds[0][1].shape[1]
    d_e = edge_features_from_bboxes(ds[0][0], ds[0][1]).shape[1]
    hidden = 16

    # Decide which arity the model uses by probing the dataset. Synth
    # VG scenes are usually too sparse for k=3, so default arity is
    # k=2 (raw edges as 2-uniform hyperedges).
    n_with_k3 = sum(1 for g, _, _ in ds if construct(g))
    chosen_arity = 3 if n_with_k3 >= max(5, len(ds) // 3) else 2
    print(f"  scenes with k=3 cycles: {n_with_k3}/{len(ds)} → "
          f"using arity={chosen_arity}", flush=True)

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=n_nodes_pad, n_layers=2, hidden_dim=hidden, grid=3, k=3,
            spline_kinds=["catmull_rom"]*2, init_scale=0.05, pool_mode="sum",
            jk_mode="concat", layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True),
        arities=(chosen_arity,), init_arity_logits=(0.0,),
        vertex_feat_dim=d_v, edge_feat_dim=d_e)
    model = MixedAritySignedKAN(cfg).to(device)
    clf = nn.Linear(hidden * 2, 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()),
                            lr=5e-3)

    def build_scene_inputs(sid, seed=0):
        g, vf, _ = ds[sid]
        per_arity = build_per_arity(g, (chosen_arity,), 1000, device,
                                       seed=seed, n_nodes_pad=n_nodes_pad)
        if not per_arity: return None
        vf_pad = np.zeros((n_nodes_pad, d_v), dtype=np.float32)
        vf_pad[:vf.shape[0]] = vf
        vf_t = torch.from_numpy(vf_pad).to(device)
        ef_t = torch.from_numpy(edge_features_from_bboxes(g, vf)).to(device)
        # e2v: edge-to-vertex sparse (n_nodes, n_edges) — uses scipy
        from scipy.sparse import csr_matrix
        n_e = g.edges.shape[0]
        rows = np.concatenate([g.edges[:, 0], g.edges[:, 1]])
        cols = np.concatenate([np.arange(n_e), np.arange(n_e)])
        data = np.ones(2 * n_e, dtype=np.float32) * 0.5
        e2v_csr = csr_matrix((data, (rows, cols)),
                              shape=(n_nodes_pad, n_e))
        coo = e2v_csr.tocoo()
        idx = torch.from_numpy(np.stack([coo.row.astype(np.int64),
                                            coo.col.astype(np.int64)])).to(device)
        v = torch.from_numpy(coo.data).to(device)
        e2v = torch.sparse_coo_tensor(idx, v, (n_nodes_pad, n_e)).coalesce()
        q_edges = torch.from_numpy(g.edges).long().to(device)
        target = torch.from_numpy((g.signs == 1).astype(np.float32)).to(device)
        return per_arity, vf_t, ef_t, e2v, q_edges, target

    train_inputs = [(sid, build_scene_inputs(sid)) for sid in tr_scenes]
    train_inputs = [x for x in train_inputs if x[1] is not None]
    test_inputs = [(sid, build_scene_inputs(sid)) for sid in te_scenes]
    test_inputs = [x for x in test_inputs if x[1] is not None]

    if not train_inputs or not test_inputs:
        print("  no scenes with cycles, skipping", flush=True); return []

    for ep in range(n_epochs):
        model.train(); clf.train()
        random.shuffle(train_inputs)
        for _, (pa, vf_t, ef_t, e2v, q_edges, target) in train_inputs:
            edge_emb = model.encode_edges(
                pa, query_edges=q_edges,
                vertex_features=vf_t, edge_features=ef_t,
                edge_to_vertex=e2v)
            logits = clf(edge_emb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); clf.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for _, (pa, vf_t, ef_t, e2v, q_edges, target) in test_inputs:
            edge_emb = model.encode_edges(
                pa, query_edges=q_edges,
                vertex_features=vf_t, edge_features=ef_t,
                edge_to_vertex=e2v)
            probs = torch.sigmoid(clf(edge_emb).squeeze(-1)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_true.extend((target.cpu().numpy() == 1).astype(int).tolist())
    auc = roc_auc_score(all_true, all_probs)
    f1 = f1_score(all_true, np.array(all_probs) > 0.5, average="macro",
                   zero_division=0)

    # Time inference: a single test scene's forward
    sid, (pa, vf_t, ef_t, e2v, q_edges, target) = test_inputs[0]
    def fwd():
        with torch.no_grad():
            edge_emb = model.encode_edges(
                pa, query_edges=q_edges,
                vertex_features=vf_t, edge_features=ef_t,
                edge_to_vertex=e2v)
            return torch.sigmoid(clf(edge_emb).squeeze(-1))
    per_call = time_per_call(fwd)
    thr = time_throughput(fwd)
    n_e_scene = q_edges.shape[0]
    print(f"  scene-graph HSiKAN: auc={auc:.3f} f1m={f1:.3f} "
          f"per-call={per_call:.2f}ms thr={thr:.2f}ms (per scene of {n_e_scene} edges)",
          flush=True)
    return [dict(
        domain="scene", subset=f"synth_vg_k={chosen_arity}", model="HSiKAN",
        n_train_scenes=len(train_inputs), n_test_scenes=len(test_inputs),
        accuracy=dict(auc=float(auc), f1m=float(f1)),
        fwd_per_call_ms=per_call,
        fwd_throughput_ms=thr,
        n_edges_per_scene=n_e_scene,
        n_params=n_params(model) + n_params(clf),
    )]


# -----------------------------------------------------------------------
# Domain 3: Kinematic mechanism classification + DOF regression

def domain_kinematic(device, n_epochs=20) -> list[dict]:
    print("\n=== domain: kinematic (mechanism class + DOF) ===", flush=True)
    from .run_phase11_kinematic_tasks import (
        build_random_mechanism, detect_dominant_arity,
        GraphLevelHSiKAN, _build_per_arity_input,
    )
    rng = random.Random(0)
    torch.manual_seed(0); np.random.seed(0)
    train, test = [], []
    for _ in range(80): train.append(build_random_mechanism(rng))
    for _ in range(40): test.append(build_random_mechanism(rng))

    rows = []
    for arity in (4, 6):
        cands_tr = [t for t in train if detect_dominant_arity(t.g) == arity]
        cands_te = [t for t in test if detect_dominant_arity(t.g) == arity]
        if not cands_tr or not cands_te: continue
        n_nodes_max = max(c.g.n_nodes for c in cands_tr + cands_te)
        train_inputs = []
        for inst in cands_tr:
            inp = _build_per_arity_input(inst.g, arity, 30_000, device, 0,
                                            n_nodes_pad=n_nodes_max)
            if inp is None: continue
            train_inputs.append((inst, inp))
        test_inputs = []
        for inst in cands_te:
            inp = _build_per_arity_input(inst.g, arity, 30_000, device, 0,
                                            n_nodes_pad=n_nodes_max)
            if inp is None: continue
            test_inputs.append((inst, inp))
        if not train_inputs or not test_inputs: continue

        model = GraphLevelHSiKAN(n_nodes_max=n_nodes_max, arity=arity,
                                    hidden=16, n_classes=4).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)
        y_cls = torch.tensor([m[0].family_label for m in train_inputs],
                              dtype=torch.long, device=device)
        y_reg = torch.tensor([float(m[0].dof) for m in train_inputs],
                              dtype=torch.float32, device=device)
        for ep in range(n_epochs):
            model.train()
            perm = torch.randperm(len(train_inputs))
            for i in perm:
                cls_logits, reg_pred = model(train_inputs[i][1])
                l_cls = F.cross_entropy(cls_logits.unsqueeze(0),
                                          y_cls[i:i+1])
                l_reg = F.mse_loss(reg_pred, y_reg[i])
                loss = l_cls + 0.05 * l_reg
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        cls_preds, reg_preds, y_true_cls, y_true_reg = [], [], [], []
        with torch.no_grad():
            for inst, inp in test_inputs:
                cls_logits, reg_pred = model(inp)
                cls_preds.append(int(cls_logits.argmax().item()))
                reg_preds.append(float(reg_pred.item()))
                y_true_cls.append(inst.family_label)
                y_true_reg.append(float(inst.dof))
        acc = accuracy_score(y_true_cls, cls_preds)
        f1m = f1_score(y_true_cls, cls_preds, average="macro",
                        zero_division=0)
        dof_mae = float(np.mean(np.abs(np.array(reg_preds)
                                          - np.array(y_true_reg))))

        # Time inference: one mechanism forward
        _, inp_one = test_inputs[0]
        def fwd():
            with torch.no_grad():
                return model(inp_one)
        per_call = time_per_call(fwd)
        thr = time_throughput(fwd)
        rows.append(dict(
            domain="kinematic", subset=f"arity_k={arity}", model="HSiKAN",
            n_train=len(train_inputs), n_test=len(test_inputs),
            n_nodes_max=n_nodes_max,
            accuracy=dict(family_acc=float(acc),
                            family_f1m=float(f1m),
                            dof_mae=float(dof_mae)),
            fwd_per_call_ms=per_call,
            fwd_throughput_ms=thr,
            n_params=n_params(model),
        ))
        print(f"  k={arity}: acc={acc:.3f} f1m={f1m:.3f} dof_mae={dof_mae:.2f}  "
              f"per-call={per_call:.2f}ms thr={thr:.2f}ms",
              flush=True)
    return rows


# -----------------------------------------------------------------------
# Domain 4: Per-vertex pose estimation (XYZ regression)

def domain_pose(device, n_epochs=20) -> list[dict]:
    print("\n=== domain: pose (per-vertex XYZ regression) ===", flush=True)
    from .run_phase12_position_regression import (
        build_mechanism_with_positions, PositionRegHSiKAN, _build_input,
    )
    from .run_phase11_kinematic_tasks import detect_dominant_arity
    rng = random.Random(0)
    torch.manual_seed(0); np.random.seed(0)
    train, test = [], []
    for _ in range(60): train.append(build_mechanism_with_positions(rng))
    for _ in range(30): test.append(build_mechanism_with_positions(rng))

    rows = []
    for arity in (4, 6):
        cands_tr = [(g, p, fam) for g, p, fam in train
                     if detect_dominant_arity(g) == arity]
        cands_te = [(g, p, fam) for g, p, fam in test
                     if detect_dominant_arity(g) == arity]
        if not cands_tr or not cands_te: continue
        n_nodes_max = max(g.n_nodes for g, _, _ in cands_tr + cands_te)
        train_inputs = []
        for g, pos, fam in cands_tr:
            inp = _build_input(g, arity, 30_000, device, 0, n_nodes_max)
            if inp is None: continue
            pos_pad = np.zeros((n_nodes_max, 3), dtype=np.float32)
            pos_pad[:pos.shape[0]] = pos
            mask = np.zeros(n_nodes_max, dtype=np.float32)
            mask[:g.n_nodes] = 1.0
            train_inputs.append((inp,
                                  torch.from_numpy(pos_pad).to(device),
                                  torch.from_numpy(mask).to(device)))
        test_inputs = []
        for g, pos, fam in cands_te:
            inp = _build_input(g, arity, 30_000, device, 0, n_nodes_max)
            if inp is None: continue
            pos_pad = np.zeros((n_nodes_max, 3), dtype=np.float32)
            pos_pad[:pos.shape[0]] = pos
            mask = np.zeros(n_nodes_max, dtype=np.float32)
            mask[:g.n_nodes] = 1.0
            test_inputs.append((inp,
                                 torch.from_numpy(pos_pad).to(device),
                                 torch.from_numpy(mask).to(device)))
        if not train_inputs or not test_inputs: continue

        model = PositionRegHSiKAN(n_nodes_max=n_nodes_max, arity=arity,
                                      hidden=16, n_layers=2, grid=3).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-2)
        for ep in range(n_epochs):
            model.train()
            perm = torch.randperm(len(train_inputs))
            for i in perm:
                inp, pos, mask = train_inputs[i]
                pred = model(inp)
                err = (pred - pos) * mask.unsqueeze(-1)
                loss = (err.pow(2).sum()) / max(mask.sum().item(), 1.0)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        all_mse, all_mae = [], []
        with torch.no_grad():
            for inp, pos, mask in test_inputs:
                pred = model(inp)
                err = (pred - pos) * mask.unsqueeze(-1)
                mse = (err.pow(2).sum()) / max(mask.sum().item(), 1.0)
                mae = (err.abs().sum()) / max(mask.sum().item(), 1.0) / 3.0
                all_mse.append(float(mse.item()))
                all_mae.append(float(mae.item()))
        mse_mean = float(np.mean(all_mse))
        mae_mean = float(np.mean(all_mae))

        inp_one, _, _ = test_inputs[0]
        def fwd():
            with torch.no_grad():
                return model(inp_one)
        per_call = time_per_call(fwd)
        thr = time_throughput(fwd)
        rows.append(dict(
            domain="pose", subset=f"arity_k={arity}", model="HSiKAN",
            n_train=len(train_inputs), n_test=len(test_inputs),
            n_nodes_max=n_nodes_max,
            accuracy=dict(mse_mean=mse_mean, mae_mean=mae_mean),
            fwd_per_call_ms=per_call,
            fwd_throughput_ms=thr,
            n_params=n_params(model),
        ))
        print(f"  k={arity}: mse={mse_mean:.4f} mae={mae_mean:.3f}  "
              f"per-call={per_call:.2f}ms thr={thr:.2f}ms",
              flush=True)
    return rows


# -----------------------------------------------------------------------
# Bitcoin: just reference the existing inference_bench.json

def domain_bitcoin_summary() -> list[dict]:
    p = Path("signedkan_wip/experiments/results/inference_bench.json")
    if not p.exists():
        print("  bitcoin: inference_bench.json not found, skipping",
              flush=True)
        return []
    rows = json.loads(p.read_text())
    out = []
    for r in rows:
        out.append(dict(
            domain="bitcoin", subset=r["dataset"],
            model=r["model"], device=r["device"],
            n_test=r["n_test_queries"],
            setup_ms=r["setup_ms"],
            accuracy=dict(note="separate train/eval — see HSiKAN paper §V"),
            fwd_per_call_ms=r["fwd_med_ms"],
            fwd_throughput_ms=None,
            n_params=r["n_params"],
        ))
    return out


# -----------------------------------------------------------------------
# Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    from signedkan_wip.src.runtime_config import get_runtime
    _compile = get_runtime().compile
    print(f"compile: {'1' if _compile.enabled else '0'}")
    print(f"compile mode: {_compile.mode}")

    all_rows = []
    all_rows += domain_bitcoin_summary()
    all_rows += domain_sbm(device, n_epochs=30)
    all_rows += domain_scene(device, n_epochs=30)
    all_rows += domain_kinematic(device, n_epochs=20)
    all_rows += domain_pose(device, n_epochs=20)

    out_path = Path("signedkan_wip/experiments/results/multi_domain_perf_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {out_path}")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'domain':<10}{'subset':<22}{'model':<8}{'device':<5}"
          f"{'fwd_ms':>8}{'thr_ms':>8}{'n_in':>8}  accuracy")
    for r in all_rows:
        dev = r.get("device", "cuda")
        thr = r.get("fwd_throughput_ms")
        thr_s = f"{thr:>8.2f}" if thr is not None else "       —"
        n_in = (r.get("n_test")
                  or r.get("n_test_scenes")
                  or r.get("n_edges_per_scene")
                  or "?")
        acc_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, (int, float))
                              else f"{k}={v}"
                              for k, v in r["accuracy"].items())
        print(f"{r['domain']:<10}{r['subset']:<22}{r['model']:<8}{dev:<5}"
              f"{r['fwd_per_call_ms']:>8.2f}{thr_s}{str(n_in):>8}  {acc_str}")


if __name__ == "__main__":
    main()
