"""Phase 14 — HSiKAN action recognition on synthetic NTU skeleton data.

End-to-end validation of the per-vertex + per-edge continuous-feature
pathway. The task: predict action class from a skeleton SignedGraph
(static topology) + per-frame joint positions/velocities (per-vertex)
+ per-frame bone vectors/lengths (per-edge).

Dataset: 8-class synthetic NTU substitute (160 samples, 30 frames each)
from `signedkan_wip.src.adapters.ntu_skeleton.synth_ntu_dataset`.

Three configurations compared:
  1. **MLP baseline**: flat per-sample feature vector → MLP → class.
     Uses the same vertex+edge features but no graph structure.
  2. **HSiKAN structural-only**: cycle-pool features alone (no
     continuous features). Validates that the static skeleton has
     class-discriminative cycle structure.
  3. **HSiKAN + features**: full pathway — cycle pool + per-vertex +
     per-edge continuous features.

Expected: HSiKAN+features > MLP > HSiKAN-structural-only on a
skeleton task where graph topology is fixed (so cycle-only features
can't distinguish classes; the per-vertex motion features carry the
signal).
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from signedkan_wip.src.adapters.ntu_skeleton import (
    NTUSample, NTU_BONES_25, build_ntu_signed_graph, synth_ntu_dataset,
    aggregate_per_vertex_features, aggregate_per_edge_features,
)
from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_edge_to_tuples)
from signedkan_wip.src.core.n_tuples import construct_k
from signedkan_wip.src.core.signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)


def _build_per_arity(g, arity, max_per, device, n_nodes_pad, seed=0):
    if arity == 3:
        t_k = construct(g)
    else:
        t_k = construct_k(g, k=arity, max_cycles=max_per, seed=seed)
    if not t_k:
        return None
    if len(t_k) > max_per:
        t_k = subsample_tuples(t_k, max_per, seed=seed)
    triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
    triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
    triad_v = torch.from_numpy(triad_v_np).to(device)
    triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
    M_vt = build_vertex_triad_incidence(triad_v_np, n_nodes_pad, device, mode="sum")
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
        M_e = torch.sparse_coo_tensor(idx, v, (g.edges.shape[0], len(t_k))).coalesce()
    else:
        M_e = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,)),
            (g.edges.shape[0], len(t_k))).to(device)
    return [(triad_v, triad_sigma, M_vt, M_e)]


def _build_edge_to_vertex(g, n_nodes, device):
    """Build the V × E sparse matrix mapping each edge to its endpoints
    (each endpoint contributes 0.5 weight)."""
    e_arr = g.edges
    rows = np.concatenate([e_arr[:, 0], e_arr[:, 1]])
    cols = np.concatenate([np.arange(e_arr.shape[0]), np.arange(e_arr.shape[0])])
    vals = np.full(2 * e_arr.shape[0], 0.5, dtype=np.float32)
    idx = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    v = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, v, (n_nodes, e_arr.shape[0])).coalesce()


class GraphActionHSiKAN(nn.Module):
    def __init__(self, n_nodes: int, arity: int,
                  hidden: int = 16, n_layers: int = 2, grid: int = 5,
                  vertex_feat_dim: int = 0, edge_feat_dim: int = 0,
                  n_classes: int = 8):
        super().__init__()
        cfg = MixedAritySignedKANConfig(
            base=MultiLayerSignedKANConfig(
                n_nodes=n_nodes, n_layers=n_layers,
                hidden_dim=hidden, grid=grid, k=3,
                spline_kinds=["catmull_rom"] * n_layers,
                init_scale=0.05, pool_mode="sum", jk_mode="concat",
                layer_norm_between=True, share_weights=True,
                inner_skip="highway", outer_skip="none", use_residual=True,
            ),
            arities=(arity,), init_arity_logits=(0.0,),
            vertex_feat_dim=vertex_feat_dim,
            edge_feat_dim=edge_feat_dim,
        )
        self.backbone = MixedAritySignedKAN(cfg)
        d_jk = hidden * n_layers
        self.cls_head = nn.Linear(d_jk, n_classes)

    def forward(self, per_arity_inputs, vf=None, ef=None, e2v=None):
        graph_emb = self.backbone.encode_graph(
            per_arity_inputs,
            vertex_features=vf,
            edge_features=ef,
            edge_to_vertex=e2v,
        )
        return self.cls_head(graph_emb)


class MLPBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _flat_features(sample: NTUSample) -> np.ndarray:
    vf = aggregate_per_vertex_features(sample, pool="stats").flatten()
    ef = aggregate_per_edge_features(sample, pool="stats").flatten()
    return np.concatenate([vf, ef])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_classes", type=int, default=8)
    ap.add_argument("--n_per_class", type=int, default=80)
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=100)
    args = ap.parse_args()

    print("=== Phase 14 — HSiKAN action recognition on synthetic NTU ===",
          flush=True)
    print(f"  classes={args.n_classes}  n_per_class={args.n_per_class}  "
          f"frames={args.n_frames}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}", flush=True)

    g = build_ntu_signed_graph()
    print(f"  skeleton graph: {g.stats()}", flush=True)
    cycles_3 = construct(g)
    cycles_4 = construct_k(g, k=4, max_cycles=10000)
    print(f"  k=3 cycles in skeleton: {len(cycles_3)}  "
          f"k=4 cycles: {len(cycles_4)}", flush=True)

    if len(cycles_3) == 0 and len(cycles_4) == 0:
        print("  NOTE: NTU skeleton is a tree (0 cycles natively); "
              "we add a closing bone (head→l_foot) to give HSiKAN's "
              "cycle-pool a single long cycle to operate on. The "
              "structural prior is therefore weak on NTU (no real "
              "cycles in skeletons), but the per-vertex + per-edge "
              "feature pathway gets exercised end-to-end.",
              flush=True)
    run_hsikan = True   # always try with closing-bone trick

    aggregate = {"mlp_baseline": [], "hsikan_features": []}
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        all_samples = synth_ntu_dataset(
            n_classes=args.n_classes, n_per_class=args.n_per_class,
            n_frames=args.n_frames, seed=seed,
        )
        n = len(all_samples)
        n_train = int(0.7 * n); n_test = n - n_train
        train_samples = all_samples[:n_train]
        test_samples = all_samples[n_train:]

        # MLP baseline.
        X_tr = np.stack([_flat_features(s) for s in train_samples])
        y_tr = np.array([s.action_label for s in train_samples])
        X_te = np.stack([_flat_features(s) for s in test_samples])
        y_te = np.array([s.action_label for s in test_samples])
        Xt = torch.from_numpy(X_tr.astype(np.float32)).to(device)
        Yt = torch.from_numpy(y_tr).to(device)
        Xe = torch.from_numpy(X_te.astype(np.float32)).to(device)
        Ye = torch.from_numpy(y_te).to(device)
        mlp = MLPBaseline(X_tr.shape[1], hidden=128,
                            n_classes=args.n_classes).to(device)
        opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        for ep in range(args.n_epochs):
            mlp.train()
            logits = mlp(Xt)
            loss = F.cross_entropy(logits, Yt)
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        with torch.no_grad():
            preds = mlp(Xe).argmax(-1).cpu().numpy()
        acc_mlp = accuracy_score(y_te, preds)
        f1_mlp = f1_score(y_te, preds, average="macro", zero_division=0)
        aggregate["mlp_baseline"].append((acc_mlp, f1_mlp))
        print(f"  seed={seed}  MLP_baseline: acc={acc_mlp:.3f}  f1m={f1_mlp:.3f}",
              flush=True)

        # HSiKAN with per-vertex + per-edge features.
        # Skeleton tree has no cycles → add a closing bone (head→l_foot)
        # to create artificial cycles for the cycle-pool to enumerate.
        from signedkan_wip.src.datasets import SignedGraph
        closing_edge = np.array([[3, 19]], dtype=np.int64)
        new_edges = np.concatenate([g.edges, closing_edge], axis=0)
        new_signs = np.concatenate([g.signs, np.array([+1], dtype=np.int8)])
        g_closed = SignedGraph(edges=new_edges, signs=new_signs, n_nodes=25)
        # The closing bone creates one long cycle; smaller arities likely
        # 0. Probe a few k's to find one that works.
        chosen_arity = None
        cycles_in_closed = []
        for try_k in (4, 5, 6, 7, 8, 9, 10, 12, 14):
            try:
                cs = construct_k(g_closed, k=try_k, max_cycles=10000)
                if len(cs) > 0:
                    chosen_arity = try_k
                    cycles_in_closed = cs
                    break
            except Exception:
                continue
        if seed == args.seeds[0]:
            print(f"  closing bone gives chosen_arity={chosen_arity}  "
                  f"cycles={len(cycles_in_closed)}", flush=True)
        if chosen_arity is None:
            print(f"  seed={seed}  no cycles even with closing bone; "
                  f"skipping HSiKAN", flush=True)
            continue

        n_nodes_pad = 25
        per_arity_inp = _build_per_arity(g_closed, chosen_arity, 1000, device, n_nodes_pad, seed=seed)
        if per_arity_inp is None:
            continue
        e2v = _build_edge_to_vertex(g_closed, 25, device)

        vf_dim = aggregate_per_vertex_features(train_samples[0], "stats").shape[1]
        ef_dim = aggregate_per_edge_features(train_samples[0], "stats").shape[1] + 1  # +1 for closing-bone padding
        # Actually: edge features are per-original-edge. For the closed
        # graph we pad with zeros for the new closing bone.
        ef_dim_orig = aggregate_per_edge_features(train_samples[0], "stats").shape[1]

        model = GraphActionHSiKAN(
            n_nodes=n_nodes_pad, arity=chosen_arity, hidden=16, n_layers=2, grid=5,
            vertex_feat_dim=vf_dim, edge_feat_dim=ef_dim_orig,
            n_classes=args.n_classes,
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        for ep in range(args.n_epochs):
            model.train()
            perm = torch.randperm(len(train_samples))
            loss_total = 0.0
            for i in perm:
                s = train_samples[i]
                vf = torch.from_numpy(
                    aggregate_per_vertex_features(s, "stats")
                ).to(device)
                ef_orig = aggregate_per_edge_features(s, "stats")
                ef_padded = np.zeros((g_closed.edges.shape[0], ef_dim_orig), dtype=np.float32)
                ef_padded[:ef_orig.shape[0]] = ef_orig
                ef = torch.from_numpy(ef_padded).to(device)
                y = torch.tensor([s.action_label], dtype=torch.long, device=device)
                logits = model(per_arity_inp, vf=vf, ef=ef, e2v=e2v)
                loss = F.cross_entropy(logits.unsqueeze(0), y)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_total += loss.item()
        model.eval()
        cls_preds = []
        with torch.no_grad():
            for s in test_samples:
                vf = torch.from_numpy(aggregate_per_vertex_features(s, "stats")).to(device)
                ef_orig = aggregate_per_edge_features(s, "stats")
                ef_padded = np.zeros((g_closed.edges.shape[0], ef_dim_orig), dtype=np.float32)
                ef_padded[:ef_orig.shape[0]] = ef_orig
                ef = torch.from_numpy(ef_padded).to(device)
                logits = model(per_arity_inp, vf=vf, ef=ef, e2v=e2v)
                cls_preds.append(int(logits.argmax().item()))
        acc_h = accuracy_score(y_te, cls_preds)
        f1_h = f1_score(y_te, cls_preds, average="macro", zero_division=0)
        aggregate["hsikan_features"].append((acc_h, f1_h))
        print(f"  seed={seed}  HSiKAN+vf+ef: acc={acc_h:.3f}  f1m={f1_h:.3f}",
              flush=True)

    print("\n=== Aggregated (median across seeds) ===", flush=True)
    for cell, results in aggregate.items():
        if not results: continue
        accs = [r[0] for r in results]
        f1ms = [r[1] for r in results]
        print(f"  {cell:<20s}  acc_med={statistics.median(accs):.3f}  "
              f"f1m_med={statistics.median(f1ms):.3f}", flush=True)


if __name__ == "__main__":
    main()
