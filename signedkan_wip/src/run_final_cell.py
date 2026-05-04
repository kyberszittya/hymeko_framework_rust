"""Single (dataset, model, hidden) cell — runs in isolation and prints
ONE JSON line to stdout for assembly by ``run_final_table.sh``.

Each invocation = one fresh Python process = clean cudagraph cache.
This is the only way to get honest latency numbers across the
hidden_dim sweep on a Turing GPU where cudagraph cache eviction
between configs distorts in-process measurements.

Usage:
    HSIKAN_TORCH_COMPILE=1 python -m signedkan_wip.src.run_final_cell \
        --dataset bitcoin_alpha --model HSiKAN --hidden 16

Recognised models:
    HSiKAN, HSiKAN-mixed, SGCN, HSiKAN-graphlevel, HSiKAN-pose
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score


def time_per_call(fn, n_warmup=15, n_repeats=40, sync=True):
    cuda_sync = sync and torch.cuda.is_available()
    for _ in range(n_warmup):
        fn()
        if cuda_sync: torch.cuda.synchronize()
    samples = []
    for _ in range(n_repeats):
        if cuda_sync: torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda_sync: torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def n_params(*modules) -> int:
    return sum(sum(p.numel() for p in m.parameters() if p.requires_grad)
                 for m in modules)


# ----- Edge-sign-prediction (Bitcoin / SBM / Slashdot) -----

def cell_signed_graph(dataset: str, model_name: str, hidden: int,
                        n_epochs: int, max_k4: int, device, seed: int = 0):
    from .datasets import load, deduplicate_pairs, split
    from .datasets_small import sbm_signed
    from .hyperedges import construct
    from .n_tuples import construct_k
    from .mixed_arity_signedkan import (MixedAritySignedKAN,
                                          MixedAritySignedKANConfig,
                                          subsample_tuples,
                                          build_vertex_to_tuples,
                                          build_edge_to_tuples)
    from .signedkan import (MultiLayerSignedKANConfig,
                             build_vertex_triad_incidence)
    from .baselines.sgcn_model import SGCN, build_signed_adj
    from .run_phase2_mixed_arity import _build_edge_incidence

    torch.manual_seed(seed); np.random.seed(seed)
    # Load
    if dataset.startswith("sbm_n"):
        n_nodes = int(dataset.split("_n")[1])
        g, _ = sbm_signed(n_nodes=n_nodes, n_communities=4, seed=seed)
    else:
        g = load(dataset)
        # NOTE: published Slashdot SOTA script (run_phase7_slashdot_pruning.py)
        # does NOT call deduplicate_pairs — match exactly.
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    y_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    y_te = (s_te == 1).astype(np.float32)
    n_pos = int(y_tr.sum().item()); n_neg = int((1 - y_tr).sum().item())
    pw = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                       device=device)

    if model_name == "SGCN":
        A_pos, A_neg = build_signed_adj(e_tr, s_tr, g.n_nodes, device)
        sgcn = SGCN(n_nodes=g.n_nodes, hidden_dim=hidden,
                     n_layers=2).to(device)
        opt = torch.optim.Adam(sgcn.parameters(), lr=5e-3)
        e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
        e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)
        for _ in range(n_epochs):
            sgcn.train()
            z = sgcn.encode_nodes(A_pos, A_neg)
            logits = sgcn.edge_logits(z, e_tr_t).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr,
                                                        pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sgcn.eval()
        def fwd():
            with torch.no_grad():
                z = sgcn.encode_nodes(A_pos, A_neg)
                return sgcn.edge_logits(z, e_te_t)
        with torch.no_grad():
            probs = torch.sigmoid(fwd().squeeze(-1)).cpu().numpy()
        auc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else float("nan")
        f1 = f1_score(y_te, probs > 0.5, average="macro", zero_division=0)
        lat = time_per_call(fwd)
        return dict(dataset=dataset, model="SGCN", hidden=hidden,
                      n_test=int(len(te_idx)),
                      auc=float(auc), f1m=float(f1),
                      fwd_per_call_ms=lat,
                      n_params=int(n_params(sgcn)))

    # HSiKAN arities — Slashdot SOTA-config uses k=(3,4,5) (αₖ shows
    # k=5 carries 54% of signal); other datasets use k=(3,4).
    # Override via env var HSIKAN_ARITIES=2,3,4,5 etc.
    arities_env = os.environ.get("HSIKAN_ARITIES")
    if arities_env:
        arities = tuple(int(x) for x in arities_env.split(","))
    elif dataset == "slashdot":
        arities = (3, 4, 5)
    else:
        arities = (3, 4)
    cap_dict = {2: 1_000_000, 3: 30_000,
                  4: max_k4, 5: max_k4, 6: max_k4}

    from .n_tuples import construct_2
    per_arity_tr, per_arity_te = [], []
    for k in arities:
        if k == 2:
            t_k = construct_2(g)
        elif k == 3:
            t_k = construct(g)
        else:
            # Published Slashdot SOTA (phase7_slashdot_pruning.py:65) uses
            # plain construct_k WITHOUT early_stop — reservoir-sampled
            # cycles, unbiased. early_stop biases toward small-vertex
            # cycles which apparently hurts AUC noticeably on Slashdot.
            t_k = construct_k(g, k=k, max_cycles=cap_dict[k], seed=seed)
        if not t_k: continue
        if len(t_k) > cap_dict[k]:
            t_k = subsample_tuples(t_k, cap_dict[k], seed=seed)
        triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        M_vt = build_vertex_triad_incidence(triad_v_np, g.n_nodes, device,
                                              mode="sum")
        n_t = len(t_k)
        edge_to_tuples = {}
        # k=2 self-tuple lookup: for each undirected edge key, the
        # tuple-index whose vertex set IS that edge. Used to exclude
        # the answer-leaking row from M_e at query time (n_tuples.py:215).
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
                if not ids: continue
                w = 1.0 / float(len(ids))
                for t in ids:
                    rows.append(ei); cols.append(int(t)); vals.append(w)
            if not rows:
                return torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0,)),
                    (edges_arr.shape[0], n_t),
                ).to(device)
            idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
            v = torch.tensor(vals, dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(
                idx, v, (edges_arr.shape[0], n_t),
            ).coalesce()

        M_e_tr = build_me(e_tr)
        M_e_te = build_me(e_te)
        per_arity_tr.append((triad_v, triad_sigma, M_vt, M_e_tr))
        per_arity_te.append((triad_v, triad_sigma, M_vt, M_e_te))
    arities_used = arities[:len(per_arity_tr)]
    cycle_batch = 10000 if (dataset == "slashdot") else None
    is_slashdot = (dataset == "slashdot")

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2, hidden_dim=hidden,
            grid=3, k=3, spline_kinds=["catmull_rom"]*2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True),
        arities=arities_used,
        init_arity_logits=tuple([0.0]*len(arities_used)),
        cycle_batch_size=cycle_batch)
    model = MixedAritySignedKAN(cfg).to(device)

    if is_slashdot:
        # Match the published Slashdot SOTA config exactly
        # (run_phase7_slashdot_pruning.py:247-249): no external clf,
        # no pos_weight, no weight_decay, no grad_clip, no
        # smooth/participation regs. The model has its own
        # `model.classifier` which we use directly.
        opt = torch.optim.Adam(model.parameters(), lr=5e-2,
                                weight_decay=0.0)
        for _ in range(n_epochs):
            model.train()
            edge_emb = model.encode_edges(per_arity_tr)
            logits = model.classifier(edge_emb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_tr)
            opt.zero_grad(); loss.backward(); opt.step()
        # Eval: also use model.classifier
        model.eval()
        def fwd():
            with torch.no_grad():
                return model.classifier(model.encode_edges(per_arity_te))
        with torch.no_grad():
            probs = torch.sigmoid(fwd().squeeze(-1)).cpu().numpy()
        auc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else float("nan")
        f1 = f1_score(y_te, probs > 0.5, average="macro", zero_division=0)
        lat = time_per_call(fwd)
        return dict(dataset=dataset, model="HSiKAN-mixed", hidden=hidden,
                      n_test=int(len(te_idx)),
                      arities=list(arities_used),
                      auc=float(auc), f1m=float(f1),
                      fwd_per_call_ms=lat,
                      n_params=int(n_params(model)))

    # Non-Slashdot: keep the external clf path with class balancing
    # (works for Bitcoin, SBM where the published configs use it).
    clf = nn.Linear(hidden * 2, 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()),
                            lr=5e-3)
    for _ in range(n_epochs):
        model.train(); clf.train()
        edge_emb = model.encode_edges(per_arity_tr)
        logits = clf(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr,
                                                    pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); clf.eval()
    def fwd():
        with torch.no_grad():
            return clf(model.encode_edges(per_arity_te))
    with torch.no_grad():
        probs = torch.sigmoid(fwd().squeeze(-1)).cpu().numpy()
    auc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else float("nan")
    f1 = f1_score(y_te, probs > 0.5, average="macro", zero_division=0)
    lat = time_per_call(fwd)
    return dict(dataset=dataset, model="HSiKAN-mixed", hidden=hidden,
                  n_test=int(len(te_idx)),
                  arities=list(arities_used),
                  auc=float(auc), f1m=float(f1),
                  fwd_per_call_ms=lat,
                  n_params=int(n_params(model, clf)))


# ----- Graph-level kinematic / pose -----

def cell_kinematic(arity: int, hidden: int, n_epochs: int, device):
    from .run_phase11_kinematic_tasks import (
        build_random_mechanism, detect_dominant_arity,
        GraphLevelHSiKAN, _build_per_arity_input,
    )
    rng = random.Random(0)
    torch.manual_seed(0); np.random.seed(0)
    train, test = [], []
    for _ in range(120): train.append(build_random_mechanism(rng))
    for _ in range(40): test.append(build_random_mechanism(rng))
    cands_tr = [t for t in train if detect_dominant_arity(t.g) == arity]
    cands_te = [t for t in test if detect_dominant_arity(t.g) == arity]
    if not cands_tr or not cands_te:
        return None
    n_nodes_max = max(c.g.n_nodes for c in cands_tr + cands_te)
    train_inputs = []
    for inst in cands_tr:
        inp = _build_per_arity_input(inst.g, arity, 30000, device, 0,
                                        n_nodes_pad=n_nodes_max)
        if inp: train_inputs.append((inst, inp))
    test_inputs = []
    for inst in cands_te:
        inp = _build_per_arity_input(inst.g, arity, 30000, device, 0,
                                        n_nodes_pad=n_nodes_max)
        if inp: test_inputs.append((inst, inp))
    model = GraphLevelHSiKAN(n_nodes_max=n_nodes_max, arity=arity,
                                hidden=hidden, n_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    y_cls = torch.tensor([m[0].family_label for m in train_inputs],
                          dtype=torch.long, device=device)
    y_reg = torch.tensor([float(m[0].dof) for m in train_inputs],
                          dtype=torch.float32, device=device)
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(len(train_inputs))
        for i in perm:
            cls_logits, reg_pred = model(train_inputs[i][1])
            l_cls = F.cross_entropy(cls_logits.unsqueeze(0), y_cls[i:i+1])
            l_reg = F.mse_loss(reg_pred, y_reg[i])
            (l_cls + 0.05 * l_reg).backward()
            opt.step(); opt.zero_grad()
    model.eval()
    cls_preds, reg_preds, y_t_cls, y_t_reg = [], [], [], []
    with torch.no_grad():
        for inst, inp in test_inputs:
            c, r = model(inp)
            cls_preds.append(int(c.argmax().item()))
            reg_preds.append(float(r.item()))
            y_t_cls.append(inst.family_label); y_t_reg.append(float(inst.dof))
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_t_cls, cls_preds)
    f1m = f1_score(y_t_cls, cls_preds, average="macro", zero_division=0)
    dof_mae = float(np.mean(np.abs(np.array(reg_preds) - np.array(y_t_reg))))
    _, inp_one = test_inputs[0]
    def fwd():
        with torch.no_grad(): return model(inp_one)
    lat = time_per_call(fwd)
    return dict(dataset=f"kinematic_k{arity}", model="HSiKAN-graphlevel",
                  hidden=hidden, n_test=len(test_inputs),
                  family_acc=float(acc), family_f1m=float(f1m),
                  dof_mae=float(dof_mae),
                  fwd_per_call_ms=lat,
                  n_params=int(n_params(model)))


def cell_pose(arity: int, hidden: int, n_epochs: int, device):
    from .run_phase12_position_regression import (
        build_mechanism_with_positions, PositionRegHSiKAN, _build_input,
    )
    from .run_phase11_kinematic_tasks import detect_dominant_arity
    rng = random.Random(0)
    torch.manual_seed(0); np.random.seed(0)
    train, test = [], []
    for _ in range(150): train.append(build_mechanism_with_positions(rng))
    for _ in range(50): test.append(build_mechanism_with_positions(rng))
    cands_tr = [(g, p, fam) for g, p, fam in train
                 if detect_dominant_arity(g) == arity]
    cands_te = [(g, p, fam) for g, p, fam in test
                 if detect_dominant_arity(g) == arity]
    if not cands_tr or not cands_te:
        return None
    n_nodes_max = max(g.n_nodes for g, _, _ in cands_tr + cands_te)
    train_inputs, test_inputs = [], []
    for g, pos, _ in cands_tr:
        inp = _build_input(g, arity, 30000, device, 0, n_nodes_max)
        if not inp: continue
        pp = np.zeros((n_nodes_max, 3), dtype=np.float32); pp[:pos.shape[0]] = pos
        m = np.zeros(n_nodes_max, dtype=np.float32); m[:g.n_nodes] = 1.0
        train_inputs.append((inp, torch.from_numpy(pp).to(device),
                                torch.from_numpy(m).to(device)))
    for g, pos, _ in cands_te:
        inp = _build_input(g, arity, 30000, device, 0, n_nodes_max)
        if not inp: continue
        pp = np.zeros((n_nodes_max, 3), dtype=np.float32); pp[:pos.shape[0]] = pos
        m = np.zeros(n_nodes_max, dtype=np.float32); m[:g.n_nodes] = 1.0
        test_inputs.append((inp, torch.from_numpy(pp).to(device),
                              torch.from_numpy(m).to(device)))
    model = PositionRegHSiKAN(n_nodes_max=n_nodes_max, arity=arity,
                                  hidden=hidden, n_layers=2,
                                  grid=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    for _ in range(n_epochs):
        model.train()
        perm = torch.randperm(len(train_inputs))
        for i in perm:
            inp, pos, mask = train_inputs[i]
            pred = model(inp)
            err = (pred - pos) * mask.unsqueeze(-1)
            loss = err.pow(2).sum() / max(mask.sum().item(), 1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    all_mse, all_mae = [], []
    with torch.no_grad():
        for inp, pos, mask in test_inputs:
            pred = model(inp)
            err = (pred - pos) * mask.unsqueeze(-1)
            all_mse.append(float((err.pow(2).sum() / max(mask.sum().item(), 1.0)).item()))
            all_mae.append(float((err.abs().sum() / max(mask.sum().item(), 1.0) / 3.0).item()))
    inp_one, _, _ = test_inputs[0]
    def fwd():
        with torch.no_grad(): return model(inp_one)
    lat = time_per_call(fwd)
    return dict(dataset=f"pose_k{arity}", model="HSiKAN-pose",
                  hidden=hidden, n_test=len(test_inputs),
                  mse=float(np.mean(all_mse)),
                  mae=float(np.mean(all_mae)),
                  fwd_per_call_ms=lat,
                  n_params=int(n_params(model)))


# ----- Scene graph (k=2 fallback) -----

def cell_scene(hidden: int, n_epochs: int, device):
    from .adapters.visual_genome import (synth_dataset,
                                            edge_features_from_bboxes)
    from .datasets import SignedGraph
    from .hyperedges import construct
    from .n_tuples import construct_2
    from .mixed_arity_signedkan import (MixedAritySignedKAN,
                                          MixedAritySignedKANConfig,
                                          build_edge_to_tuples)
    from .signedkan import (MultiLayerSignedKANConfig,
                             build_vertex_triad_incidence)
    from scipy.sparse import csr_matrix as _csr
    rng = random.Random(0); np.random.seed(0); torch.manual_seed(0)
    ds_raw = synth_dataset(n_scenes=200, seed=0)
    ds = [(g, vf, sg) for g, vf, sg in ds_raw if g.edges.shape[0] >= 2]
    def flip(g, frac, rng):
        s = g.signs.copy()
        for ei in range(g.edges.shape[0]):
            if rng.random() < frac: s[ei] = -1
        return SignedGraph(edges=g.edges, signs=s.astype(np.int8),
                            n_nodes=g.n_nodes)
    ds = [(flip(g, 0.4, rng), vf, sg) for g, vf, sg in ds]
    n_tr = int(0.7 * len(ds))
    tr_scenes = list(range(n_tr)); te_scenes = list(range(n_tr, len(ds)))
    n_pad = max(g.n_nodes for g, _, _ in ds)
    d_v = ds[0][1].shape[1]
    d_e = edge_features_from_bboxes(ds[0][0], ds[0][1]).shape[1]

    def build_inputs(sid):
        g, vf, _ = ds[sid]
        t_k = construct_2(g)
        if not t_k: return None
        triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        M_vt = build_vertex_triad_incidence(triad_v_np, n_pad, device,
                                              mode="sum")
        edge_to_tuples = build_edge_to_tuples(t_k)
        rows, cols, vals = [], [], []
        for ei, e in enumerate(g.edges):
            key = (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
            for t in edge_to_tuples.get(key, []):
                rows.append(ei); cols.append(int(t)); vals.append(1.0)
        if rows:
            idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
            v = torch.tensor(vals, dtype=torch.float32, device=device)
            M_e = torch.sparse_coo_tensor(
                idx, v, (g.edges.shape[0], len(t_k))).coalesce()
        else:
            M_e = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,)),
                (g.edges.shape[0], len(t_k))).to(device)
        vf_pad = np.zeros((n_pad, d_v), dtype=np.float32); vf_pad[:vf.shape[0]] = vf
        vf_t = torch.from_numpy(vf_pad).to(device)
        ef_t = torch.from_numpy(edge_features_from_bboxes(g, vf)).to(device)
        n_e = g.edges.shape[0]
        rs = np.concatenate([g.edges[:, 0], g.edges[:, 1]])
        cs = np.concatenate([np.arange(n_e), np.arange(n_e)])
        d_ = np.ones(2*n_e, dtype=np.float32) * 0.5
        e2v_csr = _csr((d_, (rs, cs)), shape=(n_pad, n_e))
        coo = e2v_csr.tocoo()
        e2v = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([coo.row.astype(np.int64),
                                          coo.col.astype(np.int64)])).to(device),
            torch.from_numpy(coo.data).to(device),
            (n_pad, n_e)).coalesce()
        q_edges = torch.from_numpy(g.edges).long().to(device)
        target = torch.from_numpy((g.signs == 1).astype(np.float32)).to(device)
        return [(triad_v, triad_sigma, M_vt, M_e)], vf_t, ef_t, e2v, q_edges, target

    train_inputs = [(s, build_inputs(s)) for s in tr_scenes]
    train_inputs = [x for x in train_inputs if x[1] is not None]
    test_inputs = [(s, build_inputs(s)) for s in te_scenes]
    test_inputs = [x for x in test_inputs if x[1] is not None]
    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=n_pad, n_layers=2, hidden_dim=hidden, grid=3, k=3,
            spline_kinds=["catmull_rom"]*2, init_scale=0.05, pool_mode="sum",
            jk_mode="concat", layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True),
        arities=(2,), init_arity_logits=(0.0,),
        vertex_feat_dim=d_v, edge_feat_dim=d_e)
    model = MixedAritySignedKAN(cfg).to(device)
    clf = nn.Linear(hidden * 2, 1).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()),
                            lr=5e-3)
    for _ in range(n_epochs):
        model.train(); clf.train()
        random.shuffle(train_inputs)
        for _, (pa, vf_t, ef_t, e2v, q, target) in train_inputs:
            edge_emb = model.encode_edges(pa, query_edges=q,
                                              vertex_features=vf_t,
                                              edge_features=ef_t,
                                              edge_to_vertex=e2v)
            logits = clf(edge_emb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); clf.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for _, (pa, vf_t, ef_t, e2v, q, target) in test_inputs:
            edge_emb = model.encode_edges(pa, query_edges=q,
                                              vertex_features=vf_t,
                                              edge_features=ef_t,
                                              edge_to_vertex=e2v)
            p = torch.sigmoid(clf(edge_emb).squeeze(-1)).cpu().numpy()
            all_p.extend(p.tolist())
            all_t.extend((target.cpu().numpy() == 1).astype(int).tolist())
    auc = roc_auc_score(all_t, all_p)
    f1 = f1_score(all_t, np.array(all_p) > 0.5, average="macro",
                   zero_division=0)
    sid, (pa, vf_t, ef_t, e2v, q, target) = test_inputs[0]
    def fwd():
        with torch.no_grad():
            return clf(model.encode_edges(pa, query_edges=q,
                                              vertex_features=vf_t,
                                              edge_features=ef_t,
                                              edge_to_vertex=e2v))
    lat = time_per_call(fwd)
    return dict(dataset="scene_synth_vg_k2", model="HSiKAN-scene",
                  hidden=hidden, n_test_scenes=len(test_inputs),
                  auc=float(auc), f1m=float(f1),
                  fwd_per_call_ms=lat,
                  n_params=int(n_params(model, clf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["bitcoin_alpha", "bitcoin_otc", "slashdot",
                              "sbm_n200", "sbm_n400",
                              "kinematic_k4", "kinematic_k6",
                              "pose_k4", "pose_k6",
                              "scene"])
    ap.add_argument("--model", default="HSiKAN")
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--max-k4", type=int, default=200000,
                    help="Slashdot mixed-arity k=4 cap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset.startswith(("bitcoin", "slashdot", "sbm")):
        out = cell_signed_graph(args.dataset, args.model, args.hidden,
                                  args.n_epochs, args.max_k4, device,
                                  seed=args.seed)
        if out is not None:
            out["seed"] = args.seed
    elif args.dataset.startswith("kinematic_k"):
        arity = int(args.dataset.split("_k")[1])
        out = cell_kinematic(arity, args.hidden, args.n_epochs, device)
    elif args.dataset.startswith("pose_k"):
        arity = int(args.dataset.split("_k")[1])
        out = cell_pose(arity, args.hidden, args.n_epochs, device)
    elif args.dataset == "scene":
        out = cell_scene(args.hidden, args.n_epochs, device)
    else:
        out = None
    print(json.dumps(out))


if __name__ == "__main__":
    main()
