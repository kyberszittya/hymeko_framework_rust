"""Multi-domain HSiKAN deep bench (3 seeds, longer training, real
accuracy + latency).

Companion to ``run_multi_domain_perf_bench.py`` — same domains but
with realistic training budgets, multiple seeds for variance, and
class-balanced loss where applicable. The shorter bench was for
"do the wires connect"; this is for "what's the actual accuracy".

Domains:
  sbm       — n=200 + n=400, 200 epochs, pos_weight'd BCE
  scene     — synth VG, 80 epochs, k=2 fallback
  pose      — k=4 + k=6, 150 train + 50 test, 100 epochs

Bitcoin and kinematic are skipped — Bitcoin is already covered by
inference_bench.json + the published-paper accuracy table; kinematic
saturated at 100% accuracy in the quick bench so longer training
adds nothing.

Running:
    HSIKAN_TORCH_COMPILE=1 python -m signedkan_wip.experiments.runs.run_multi_domain_perf_deep

Output:
    signedkan_wip/experiments/results/multi_domain_perf_deep.json
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from signedkan_wip.src.datasets import SignedGraph
from signedkan_wip.src.datasets import sbm_signed
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
from .run_phase2_mixed_arity import (_build_edge_incidence_vertex_adj_scipy,
                                       _build_edge_incidence)
from .run_multi_domain_perf_bench import (build_per_arity, time_per_call,
                                            time_throughput, n_params,
                                            build_vertex_to_tuples_from_per_arity)

SEEDS = (0, 1, 2)


def mean_std(xs: list[float]) -> dict:
    if not xs: return dict(mean=float("nan"), std=float("nan"), n=0)
    return dict(
        mean=float(np.mean(xs)),
        std=float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0),
        n=len(xs),
    )


# -----------------------------------------------------------------------
# SBM with class-balanced loss + longer training

def domain_sbm_deep(device, n_epochs=200) -> list[dict]:
    print("\n=== domain: sbm (deep) ===", flush=True)
    out = []
    for n_nodes, name in [(200, "sbm_n200"), (400, "sbm_n400")]:
        hsk_aucs, hsk_f1s, hsk_lat, hsk_n = [], [], [], 0
        sgcn_aucs, sgcn_f1s, sgcn_lat = [], [], []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            g, _ = sbm_signed(n_nodes=n_nodes, n_communities=4, seed=seed)
            rng = np.random.RandomState(seed)
            idx = rng.permutation(g.edges.shape[0])
            n_tr = int(0.7 * len(idx))
            tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
            e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
            e_te, s_te = g.edges[te_idx], g.signs[te_idx]

            # Class balance (pos_weight for BCE)
            n_pos = int((s_tr == 1).sum()); n_neg = len(s_tr) - n_pos
            pos_w = max(1.0, float(n_neg) / max(1, n_pos))
            print(f"  {name} seed={seed} pos_w={pos_w:.2f} "
                  f"(neg/pos={n_neg}/{n_pos})", flush=True)

            # ---- HSiKAN (mixed k=3+k=4 — matches phase 6 / paper config) ----
            arities_used = []
            per_arity_full = []
            for k_arity in (3, 4):
                pa_k = build_per_arity(g, (k_arity,), 5000, device, seed=seed)
                if not pa_k: continue
                arities_used.append(k_arity)
                per_arity_full.extend(pa_k)
            if not per_arity_full:
                print(f"  {name}: no cycles at any arity seed={seed}, skipping",
                      flush=True); continue
            # Build per-arity train and test M_e tensors. Using
            # ``_build_edge_incidence`` (mode="edge_in_cycle") to match
            # the published phase-6 / phase-2 SBM protocol — M_e[q, t]
            # = 1 iff query edge q is one of the cycle-edges of t.
            # This includes the σ-as-label leak that's known and
            # documented in run_phase2_mixed_arity.py docstring; it is
            # the canonical comparison protocol against published SGCN
            # numbers, not a strict-Derr no-leak test.
            per_arity_tr, per_arity_te = [], []
            for (tv, ts, M_vt, _M_e_full) in per_arity_full:
                triad_v_np = tv.cpu().numpy()
                n_t = triad_v_np.shape[0]
                # Build edge_to_tuples dict: each cycle's k consecutive
                # vertex-pairs are its "edges in the cycle".
                k_arity = triad_v_np.shape[1]
                edge_to_tuples = {}
                for ti in range(n_t):
                    cyc = triad_v_np[ti]
                    for j in range(k_arity):
                        u_, v_ = int(cyc[j]), int(cyc[(j + 1) % k_arity])
                        key = (min(u_, v_), max(u_, v_))
                        edge_to_tuples.setdefault(key, []).append(ti)
                M_e_tr = _build_edge_incidence(e_tr, edge_to_tuples, n_t,
                                                  device, directed=False)
                M_e_te = _build_edge_incidence(e_te, edge_to_tuples, n_t,
                                                  device, directed=False)
                per_arity_tr.append((tv, ts, M_vt, M_e_tr))
                per_arity_te.append((tv, ts, M_vt, M_e_te))
            cfg = MixedAritySignedKANConfig(
                base=MultiLayerSignedKANConfig(
                    n_nodes=g.n_nodes, n_layers=2, hidden_dim=16, grid=3, k=3,
                    spline_kinds=["catmull_rom"]*2, init_scale=0.05,
                    pool_mode="sum", jk_mode="concat",
                    layer_norm_between=True, share_weights=True,
                    inner_skip="highway", outer_skip="none", use_residual=True),
                arities=tuple(arities_used),
                init_arity_logits=tuple([0.0]*len(arities_used)))
            model = MixedAritySignedKAN(cfg).to(device)
            clf = nn.Linear(16 * 2, 1).to(device)
            opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()),
                                    lr=5e-3)
            y_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
            y_te_np = (s_te == 1).astype(np.float32)
            pw = torch.tensor(pos_w, device=device)
            for ep in range(n_epochs):
                model.train(); clf.train()
                logits = clf(model.encode_edges(per_arity_tr)).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_tr,
                                                            pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval(); clf.eval()
            def hsk_fwd():
                with torch.no_grad():
                    return torch.sigmoid(clf(model.encode_edges(per_arity_te)).squeeze(-1))
            with torch.no_grad():
                probs = hsk_fwd().cpu().numpy()
            try: auc = roc_auc_score(y_te_np, probs)
            except ValueError: auc = float("nan")
            f1 = f1_score(y_te_np, probs > 0.5, average="macro",
                           zero_division=0)
            lat = time_per_call(hsk_fwd)
            hsk_aucs.append(auc); hsk_f1s.append(f1); hsk_lat.append(lat)
            hsk_n = len(te_idx)
            print(f"    HSiKAN seed={seed}: auc={auc:.3f} f1m={f1:.3f} "
                  f"lat={lat:.2f}ms", flush=True)

            # ---- SGCN ----
            A_pos, A_neg = build_signed_adj(e_tr, s_tr, g.n_nodes, device)
            sgcn = SGCN(n_nodes=g.n_nodes, hidden_dim=32, n_layers=2).to(device)
            opt = torch.optim.Adam(sgcn.parameters(), lr=5e-3)
            e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
            e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)
            for ep in range(n_epochs):
                sgcn.train()
                z = sgcn.encode_nodes(A_pos, A_neg)
                logits = sgcn.edge_logits(z, e_tr_t).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_tr,
                                                            pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
            sgcn.eval()
            def sgcn_fwd():
                with torch.no_grad():
                    z = sgcn.encode_nodes(A_pos, A_neg)
                    return torch.sigmoid(sgcn.edge_logits(z, e_te_t).squeeze(-1))
            with torch.no_grad():
                sprobs = sgcn_fwd().cpu().numpy()
            try: sauc = roc_auc_score(y_te_np, sprobs)
            except ValueError: sauc = float("nan")
            sf1 = f1_score(y_te_np, sprobs > 0.5, average="macro",
                            zero_division=0)
            slat = time_per_call(sgcn_fwd)
            sgcn_aucs.append(sauc); sgcn_f1s.append(sf1); sgcn_lat.append(slat)
            print(f"    SGCN   seed={seed}: auc={sauc:.3f} f1m={sf1:.3f} "
                  f"lat={slat:.2f}ms", flush=True)

        out.append(dict(
            domain="sbm", subset=name, model="HSiKAN", n_test=hsk_n,
            accuracy=dict(auc=mean_std(hsk_aucs), f1m=mean_std(hsk_f1s)),
            fwd_per_call_ms=mean_std(hsk_lat),
            seeds=list(SEEDS),
        ))
        out.append(dict(
            domain="sbm", subset=name, model="SGCN", n_test=hsk_n,
            accuracy=dict(auc=mean_std(sgcn_aucs), f1m=mean_std(sgcn_f1s)),
            fwd_per_call_ms=mean_std(sgcn_lat),
            seeds=list(SEEDS),
        ))
    return out


# -----------------------------------------------------------------------
# Pose: bigger training set, longer epochs, both arities, 3 seeds

def domain_pose_deep(device, n_train=150, n_test=50, n_epochs=100) -> list[dict]:
    print("\n=== domain: pose (deep) ===", flush=True)
    from .run_phase12_position_regression import (
        build_mechanism_with_positions, PositionRegHSiKAN, _build_input,
    )
    from .run_phase11_kinematic_tasks import detect_dominant_arity

    rows = []
    for arity in (4, 6):
        mse_runs, mae_runs, lat_runs = [], [], []
        n_match_tr_runs, n_match_te_runs = [], []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            rng = random.Random(seed)
            train, test = [], []
            for _ in range(n_train): train.append(build_mechanism_with_positions(rng))
            for _ in range(n_test): test.append(build_mechanism_with_positions(rng))
            cands_tr = [(g, p, fam) for g, p, fam in train
                         if detect_dominant_arity(g) == arity]
            cands_te = [(g, p, fam) for g, p, fam in test
                         if detect_dominant_arity(g) == arity]
            if not cands_tr or not cands_te:
                print(f"  k={arity} seed={seed}: no matching mechanisms, skip",
                      flush=True); continue
            n_nodes_max = max(g.n_nodes for g, _, _ in cands_tr + cands_te)
            train_inputs, test_inputs = [], []
            for g, pos, _ in cands_tr:
                inp = _build_input(g, arity, 30_000, device, seed, n_nodes_max)
                if inp is None: continue
                pos_pad = np.zeros((n_nodes_max, 3), dtype=np.float32)
                pos_pad[:pos.shape[0]] = pos
                mask = np.zeros(n_nodes_max, dtype=np.float32)
                mask[:g.n_nodes] = 1.0
                train_inputs.append((inp,
                                       torch.from_numpy(pos_pad).to(device),
                                       torch.from_numpy(mask).to(device)))
            for g, pos, _ in cands_te:
                inp = _build_input(g, arity, 30_000, device, seed, n_nodes_max)
                if inp is None: continue
                pos_pad = np.zeros((n_nodes_max, 3), dtype=np.float32)
                pos_pad[:pos.shape[0]] = pos
                mask = np.zeros(n_nodes_max, dtype=np.float32)
                mask[:g.n_nodes] = 1.0
                test_inputs.append((inp,
                                      torch.from_numpy(pos_pad).to(device),
                                      torch.from_numpy(mask).to(device)))
            if not train_inputs or not test_inputs:
                print(f"  k={arity} seed={seed}: build_input returned empty",
                      flush=True); continue

            n_match_tr_runs.append(len(train_inputs))
            n_match_te_runs.append(len(test_inputs))

            model = PositionRegHSiKAN(n_nodes_max=n_nodes_max, arity=arity,
                                          hidden=16, n_layers=2,
                                          grid=3).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=5e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs)
            best_mse = float("inf")
            for ep in range(n_epochs):
                model.train()
                perm = torch.randperm(len(train_inputs))
                for i in perm:
                    inp, pos, mask = train_inputs[i]
                    pred = model(inp)
                    err = (pred - pos) * mask.unsqueeze(-1)
                    loss = (err.pow(2).sum()) / max(mask.sum().item(), 1.0)
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()

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
            lat = time_per_call(fwd)
            mse_runs.append(mse_mean); mae_runs.append(mae_mean); lat_runs.append(lat)
            print(f"  k={arity} seed={seed}: mse={mse_mean:.4f} mae={mae_mean:.3f} "
                  f"n_tr={len(train_inputs)} n_te={len(test_inputs)} "
                  f"lat={lat:.2f}ms", flush=True)

        if mse_runs:
            rows.append(dict(
                domain="pose", subset=f"arity_k={arity}", model="HSiKAN",
                n_train_matched_mean=float(np.mean(n_match_tr_runs)),
                n_test_matched_mean=float(np.mean(n_match_te_runs)),
                accuracy=dict(mse=mean_std(mse_runs),
                                mae=mean_std(mae_runs)),
                fwd_per_call_ms=mean_std(lat_runs),
                seeds=list(SEEDS),
            ))
    return rows


# -----------------------------------------------------------------------
# Scene-graph deep (k=2 fallback already, just longer training + 3 seeds)

def domain_scene_deep(device, n_epochs=80, n_scenes=200) -> list[dict]:
    print("\n=== domain: scene-graph (deep) ===", flush=True)
    from signedkan_wip.src.adapters.visual_genome import (
        synth_dataset, edge_features_from_bboxes,
    )
    aucs, f1s, lats = [], [], []
    n_e_per_scene = None
    for seed in SEEDS:
        rng = random.Random(seed)
        np.random.seed(seed); torch.manual_seed(seed)
        ds_raw = synth_dataset(n_scenes=n_scenes, seed=seed)
        ds = [(g, vf, sg) for g, vf, sg in ds_raw if g.edges.shape[0] >= 2]
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

        n_with_k3 = sum(1 for g, _, _ in ds if construct(g))
        chosen_arity = 3 if n_with_k3 >= max(5, len(ds) // 3) else 2
        print(f"  seed={seed}: scenes with k=3 cycles {n_with_k3}/{len(ds)} "
              f"→ arity={chosen_arity}", flush=True)

        cfg = MixedAritySignedKANConfig(
            base=MultiLayerSignedKANConfig(
                n_nodes=n_nodes_pad, n_layers=2, hidden_dim=16, grid=3, k=3,
                spline_kinds=["catmull_rom"]*2, init_scale=0.05,
                pool_mode="sum", jk_mode="concat",
                layer_norm_between=True, share_weights=True,
                inner_skip="highway", outer_skip="none", use_residual=True),
            arities=(chosen_arity,), init_arity_logits=(0.0,),
            vertex_feat_dim=d_v, edge_feat_dim=d_e)
        model = MixedAritySignedKAN(cfg).to(device)
        clf = nn.Linear(16 * 2, 1).to(device)
        opt = torch.optim.Adam(
            list(model.parameters()) + list(clf.parameters()), lr=5e-3)

        def build_scene_inputs(sid, seed=0):
            g, vf, _ = ds[sid]
            per_arity = build_per_arity(g, (chosen_arity,), 1000, device,
                                           seed=seed, n_nodes_pad=n_nodes_pad)
            if not per_arity: return None
            vf_pad = np.zeros((n_nodes_pad, d_v), dtype=np.float32)
            vf_pad[:vf.shape[0]] = vf
            vf_t = torch.from_numpy(vf_pad).to(device)
            ef_t = torch.from_numpy(edge_features_from_bboxes(g, vf)).to(device)
            from scipy.sparse import csr_matrix
            n_e = g.edges.shape[0]
            rows_ = np.concatenate([g.edges[:, 0], g.edges[:, 1]])
            cols_ = np.concatenate([np.arange(n_e), np.arange(n_e)])
            data_ = np.ones(2 * n_e, dtype=np.float32) * 0.5
            e2v_csr = csr_matrix((data_, (rows_, cols_)),
                                  shape=(n_nodes_pad, n_e))
            coo = e2v_csr.tocoo()
            idx = torch.from_numpy(np.stack([coo.row.astype(np.int64),
                                                coo.col.astype(np.int64)])).to(device)
            v = torch.from_numpy(coo.data).to(device)
            e2v = torch.sparse_coo_tensor(idx, v, (n_nodes_pad, n_e)).coalesce()
            q_edges = torch.from_numpy(g.edges).long().to(device)
            target = torch.from_numpy((g.signs == 1).astype(np.float32)).to(device)
            return per_arity, vf_t, ef_t, e2v, q_edges, target

        train_inputs = [(sid, build_scene_inputs(sid, seed=seed))
                          for sid in tr_scenes]
        train_inputs = [x for x in train_inputs if x[1] is not None]
        test_inputs = [(sid, build_scene_inputs(sid, seed=seed))
                         for sid in te_scenes]
        test_inputs = [x for x in test_inputs if x[1] is not None]

        if not train_inputs or not test_inputs:
            print(f"  seed={seed}: no usable scenes, skipping", flush=True)
            continue

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
        try:
            auc = roc_auc_score(all_true, all_probs)
        except ValueError:
            auc = float("nan")
        f1 = f1_score(all_true, np.array(all_probs) > 0.5, average="macro",
                       zero_division=0)
        sid, (pa, vf_t, ef_t, e2v, q_edges, target) = test_inputs[0]
        n_e_per_scene = q_edges.shape[0]
        def fwd():
            with torch.no_grad():
                edge_emb = model.encode_edges(
                    pa, query_edges=q_edges,
                    vertex_features=vf_t, edge_features=ef_t,
                    edge_to_vertex=e2v)
                return torch.sigmoid(clf(edge_emb).squeeze(-1))
        lat = time_per_call(fwd)
        aucs.append(auc); f1s.append(f1); lats.append(lat)
        print(f"  seed={seed}: auc={auc:.3f} f1m={f1:.3f} lat={lat:.2f}ms",
              flush=True)

    return [dict(
        domain="scene", subset=f"synth_vg_k=2", model="HSiKAN",
        accuracy=dict(auc=mean_std(aucs), f1m=mean_std(f1s)),
        fwd_per_call_ms=mean_std(lats),
        n_edges_per_scene=n_e_per_scene,
        seeds=list(SEEDS),
    )]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    from signedkan_wip.src.runtime_config import get_runtime
    print(f"compile: {'1' if get_runtime().compile.enabled else '0'}")

    all_rows = []
    all_rows += domain_sbm_deep(device, n_epochs=200)
    all_rows += domain_scene_deep(device, n_epochs=80, n_scenes=200)
    all_rows += domain_pose_deep(device, n_train=150, n_test=50, n_epochs=100)

    out_path = Path("signedkan_wip/experiments/results/multi_domain_perf_deep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {out_path}")

    print("\n--- Deep summary (mean ± std over 3 seeds) ---")
    for r in all_rows:
        acc_str = ", ".join(
            f"{k}={v['mean']:.3f}±{v['std']:.3f}"
            for k, v in r["accuracy"].items()
        )
        lat = r["fwd_per_call_ms"]
        print(f"  {r['domain']:<10}{r['subset']:<22}{r['model']:<8}  "
              f"lat={lat['mean']:5.2f}±{lat['std']:.2f}ms  {acc_str}")


if __name__ == "__main__":
    main()
