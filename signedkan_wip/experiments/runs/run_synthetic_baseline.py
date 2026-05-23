"""Run HSiKAN edge-sign prediction on synthetic signed-graph tests.

Validates architectural levers (sparse attention, learnable
incidence) against ground-truth labels before deploying on real
datasets.  Edge-level link-sign prediction exercises the
``attention_m_e`` path; toggle dense vs top-K via the
``HSIKAN_SPARSE_ATTN_K`` env var.

Quick smoke (CPU OK for small synthetic graphs):
    python -m signedkan_wip.experiments.runs.run_synthetic_baseline \\
        --generator easy_sbm --n-epochs 30
    HSIKAN_SPARSE_ATTN_K=8 python -m signedkan_wip.experiments.runs.run_synthetic_baseline \\
        --generator needle_in_haystack --n-epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from signedkan_wip.src.synthetic_signed_graphs import GENERATORS
from signedkan_wip.src.mixed_arity_signedkan import (
    MixedAritySignedKAN, MixedAritySignedKANConfig,
    MultiLayerSignedKANConfig,
)
from .run_tabular_smoke import build_M_vt, build_M_e, build_per_arity


def _build_per_arity_inputs_with_M_e(g, arities, max_k, seed,
                                       device, query_edges_np):
    """Build per-arity inputs including a real M_e linking each query
    edge to the cycles whose EDGE SET contains it (proven pattern
    from run_tabular_smoke.build_M_e)."""
    per_arity_tuples, arities_used = build_per_arity(
        g, arities, max_k=max_k, seed=seed,
    )
    n_query = query_edges_np.shape[0]
    inputs = []
    for k_v, triad_v, triad_sigma in per_arity_tuples:
        triad_v_t = torch.from_numpy(triad_v).to(device)
        triad_sigma_t = torch.from_numpy(triad_sigma).to(device)
        M_vt = build_M_vt(triad_v, g.n_nodes, device)
        M_e = build_M_e(triad_v, k_v, query_edges_np, n_query,
                          device, is_walk=False)
        inputs.append((triad_v_t, triad_sigma_t, M_vt, M_e))
    return inputs, arities_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="easy_sbm",
                    choices=list(GENERATORS.keys()))
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-k", type=int, default=2000)
    ap.add_argument("--attention", choices=["off", "dot"],
                    default="dot")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s = GENERATORS[args.generator](seed=args.seed)
    n = s.graph.n_nodes
    e_total = s.graph.edges.shape[0]
    from signedkan_wip.src.runtime_config import get_runtime
    sparse_K = get_runtime().training.sparse_attn_k
    print(f"[{s.name}] n_nodes={n}, n_edges={e_total}, "
          f"attention={args.attention}, sparse_K={sparse_K} "
          f"(0 = dense)")

    # Edge-level link-sign prediction: 80/20 train/test split on edges.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(e_total)
    n_test = max(1, e_total // 5)
    te_idx = perm[:n_test]
    tr_idx = perm[n_test:]
    e_tr = s.graph.edges[tr_idx]
    s_tr = s.graph.signs[tr_idx]
    e_te = s.graph.edges[te_idx]
    s_te = s.graph.signs[te_idx]

    # Build train/test inputs with their own M_e.
    arities = (3, 4)
    tr_inputs, arities_used = _build_per_arity_inputs_with_M_e(
        s.graph, arities, args.max_k, args.seed, device, e_tr,
    )
    te_inputs, _ = _build_per_arity_inputs_with_M_e(
        s.graph, arities, args.max_k, args.seed, device, e_te,
    )

    # Vertex features (used as side info).
    if s.features is None:
        deg_pos = np.zeros(n, dtype=np.float32)
        deg_neg = np.zeros(n, dtype=np.float32)
        for (u, v), sgn in zip(s.graph.edges, s.graph.signs):
            if sgn > 0:
                deg_pos[u] += 1; deg_pos[v] += 1
            else:
                deg_neg[u] += 1; deg_neg[v] += 1
        feats = np.stack([deg_pos, deg_neg], axis=1)
    else:
        feats = s.features
    feats_t = torch.tensor(feats, dtype=torch.float32, device=device)
    feat_dim = feats_t.shape[1]

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=n, n_layers=2, hidden_dim=args.hidden,
            grid=3, k=3,
            spline_kinds=["catmull_rom"] * 2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none",
            use_residual=True),
        arities=tuple(arities_used),
        init_arity_logits=tuple([0.0] * len(arities_used)),
        vertex_feat_dim=feat_dim,
        attention_m_e=(args.attention == "dot"),
        attention_m_e_kind="dot",
    )
    model = MixedAritySignedKAN(cfg).to(device)

    e_tr_t = torch.tensor(e_tr, dtype=torch.long, device=device)
    e_te_t = torch.tensor(e_te, dtype=torch.long, device=device)
    y_tr = torch.tensor((s_tr > 0).astype(np.float32), device=device)
    y_te = (s_te > 0).astype(np.float32)

    use_attn = args.attention == "dot"
    q_tr = e_tr_t if use_attn else None
    q_te = e_te_t if use_attn else None

    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    t0 = time.time()
    for ep in range(args.n_epochs):
        model.train()
        edge_emb = model.encode_edges(tr_inputs, query_edges=q_tr,
                                        vertex_features=feats_t)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    train_s = time.time() - t0

    model.eval()
    with torch.no_grad():
        edge_emb_te = model.encode_edges(te_inputs, query_edges=q_te,
                                            vertex_features=feats_t)
        probs = torch.sigmoid(
            model.classifier(edge_emb_te).squeeze(-1)
        ).cpu().numpy()

    auc = float(roc_auc_score(y_te, probs)) if len(set(y_te.tolist())) > 1 else float("nan")
    out = dict(
        generator=s.name, seed=args.seed, n_nodes=n, n_edges=e_total,
        attention=args.attention, sparse_K=sparse_K,
        auc=round(auc, 4),
        n_params=sum(p.numel() for p in model.parameters()),
        train_s=round(train_s, 1),
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
