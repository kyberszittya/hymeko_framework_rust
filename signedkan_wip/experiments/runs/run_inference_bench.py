"""Inference-time benchmark: HSiKAN vs SGCN forward-pass latency.

Reports per-query latency on the test split for each (model, dataset, device)
cell, after a warmup pass and over N repeats. We measure ONLY the forward
pass — graph load and cycle enumeration are amortised at fit time and
reported separately as "setup".

Usage:
    python -m signedkan_wip.experiments.runs.run_inference_bench
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from signedkan_wip.src.datasets import load, split, deduplicate_pairs
from signedkan_wip.src.hyperedges import construct
from signedkan_wip.src.n_tuples import construct_k
from signedkan_wip.src.mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_vertex_to_tuples)
from signedkan_wip.src.signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from signedkan_wip.src.baselines.sgcn_model import SGCN, build_signed_adj
from .run_phase2_mixed_arity import _build_edge_incidence_vertex_adj_scipy


N_REPEATS = 20
WARMUP = 5      # torch.compile needs ≥3 calls to fully warm — use 5 for safety

# Per-dataset SOTA configurations (matched to the headline-table runs).
CONFIGS = {
    "bitcoin_alpha": dict(
        arities=(3, 4, 5), max_k=dict(k3=0, k4=20_000, k5=5_000),
        h_hsikan=16, h_sgcn=32, n_layers_sgcn=2,
    ),
    "bitcoin_otc": dict(
        arities=(3, 4, 5), max_k=dict(k3=0, k4=20_000, k5=5_000),
        h_hsikan=16, h_sgcn=32, n_layers_sgcn=2,
    ),
    "slashdot": dict(
        arities=(3, 4), max_k=dict(k3=0, k4=100_000, k5=0),
        h_hsikan=16, h_sgcn=32, n_layers_sgcn=2,
    ),
}


def _time_block(fn, n_repeats=N_REPEATS, warmup=WARMUP, sync_cuda=False):
    for _ in range(warmup):
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
    samples = []
    for _ in range(n_repeats):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return samples


def build_hsikan_inputs(dataset, cfg, seed, device):
    g = load(dataset)
    g = deduplicate_pairs(g, merge="majority")
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_te = g.edges[te_idx]
    g_features = g  # we time inference; cycle pool from full graph is fine.

    arities = cfg["arities"]
    per_arity_tuples = []
    setup_t0 = time.perf_counter()
    for k in arities:
        cap = cfg["max_k"].get(f"k{k}", None)
        if cap is None or cap == 0:
            continue
        if k == 3:
            t_k = construct(g_features)
        else:
            t_k = construct_k(g_features, k=k, max_cycles=cap, seed=seed)
        if cap and len(t_k) > cap:
            t_k = subsample_tuples(t_k, cap, seed=seed)
        if len(t_k) > 0:
            per_arity_tuples.append((k, t_k))
    arities_used = tuple(k for k, _ in per_arity_tuples)
    tuples_lists = [t for _, t in per_arity_tuples]

    per_arity_test = []
    for k, tuples in zip(arities_used, tuples_lists):
        triad_v_np = np.array([t.v for t in tuples], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in tuples], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        n_tuples = len(tuples)
        M_vt = build_vertex_triad_incidence(
            triad_v_np, g.n_nodes, device, mode="sum",
        )
        v2t = build_vertex_to_tuples(tuples)
        M_e_te = _build_edge_incidence_vertex_adj_scipy(
            e_te, v2t, {}, n_tuples, device, n_nodes=g.n_nodes,
        )
        per_arity_test.append((triad_v, triad_sigma, M_vt, M_e_te))
    setup_s = time.perf_counter() - setup_t0

    model_cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2, hidden_dim=cfg["h_hsikan"],
            grid=3, k=3, spline_kinds=["catmull_rom"] * 2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True,
        ),
        arities=arities_used,
        init_arity_logits=tuple([0.0] * len(arities_used)),
    )
    model = MixedAritySignedKAN(model_cfg).to(device)
    model.eval()
    return model, per_arity_test, e_te.shape[0], setup_s, g.n_nodes, len(arities_used)


def build_sgcn_inputs(dataset, cfg, seed, device):
    g = load(dataset)
    g = deduplicate_pairs(g, merge="majority")
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te = g.edges[te_idx]

    setup_t0 = time.perf_counter()
    A_pos, A_neg = build_signed_adj(e_tr, s_tr, g.n_nodes, device)
    setup_s = time.perf_counter() - setup_t0

    model = SGCN(n_nodes=g.n_nodes, hidden_dim=cfg["h_sgcn"],
                  n_layers=cfg["n_layers_sgcn"]).to(device)
    model.eval()
    edges_t = torch.tensor(e_te, dtype=torch.long, device=device)
    return model, (A_pos, A_neg, edges_t), e_te.shape[0], setup_s, g.n_nodes


def hsikan_forward(model, per_arity_test):
    with torch.no_grad():
        edge_emb = model.encode_edges(per_arity_test)
        return model.classifier(edge_emb)


def sgcn_forward(model, A_pos, A_neg, edges_t):
    with torch.no_grad():
        z = model.encode_nodes(A_pos, A_neg)
        return model.edge_logits(z, edges_t)


def main():
    out_path = Path("signedkan_wip/experiments/results/inference_bench.json")
    rows = []
    for dataset, cfg in CONFIGS.items():
        for device_name in ("cpu", "cuda" if torch.cuda.is_available() else None):
            if device_name is None:
                continue
            device = torch.device(device_name)
            print(f"\n=== {dataset} on {device_name} ===")

            # SGCN
            sgcn_model, (A_pos, A_neg, edges_t), n_te, setup_sgcn, n_nodes = \
                build_sgcn_inputs(dataset, cfg, seed=0, device=device)
            samples = _time_block(
                lambda: sgcn_forward(sgcn_model, A_pos, A_neg, edges_t),
                sync_cuda=(device_name == "cuda"),
            )
            sgcn_med = statistics.median(samples)
            sgcn_per_q_us = sgcn_med * 1e6 / n_te
            n_params_sgcn = sgcn_model.num_parameters()
            print(f"  SGCN  setup={setup_sgcn*1000:.1f}ms  fwd={sgcn_med*1000:.2f}ms  "
                  f"per-query={sgcn_per_q_us:.2f}us  params={n_params_sgcn}")
            rows.append(dict(
                dataset=dataset, device=device_name, model="SGCN",
                n_test_queries=n_te, n_nodes=int(n_nodes),
                setup_ms=setup_sgcn * 1000,
                fwd_med_ms=sgcn_med * 1000,
                per_query_us=sgcn_per_q_us,
                n_params=n_params_sgcn,
            ))
            del sgcn_model, A_pos, A_neg, edges_t
            if device_name == "cuda":
                torch.cuda.empty_cache()

            # HSiKAN
            hk_model, per_arity_test, n_te, setup_hk, n_nodes, n_arities = \
                build_hsikan_inputs(dataset, cfg, seed=0, device=device)
            samples = _time_block(
                lambda: hsikan_forward(hk_model, per_arity_test),
                sync_cuda=(device_name == "cuda"),
            )
            hk_med = statistics.median(samples)
            hk_per_q_us = hk_med * 1e6 / n_te
            n_params_hk = hk_model.num_parameters()
            n_cycles = sum(t[0].shape[0] for t in per_arity_test)
            print(f"  HSiKAN setup={setup_hk*1000:.1f}ms  fwd={hk_med*1000:.2f}ms  "
                  f"per-query={hk_per_q_us:.2f}us  params={n_params_hk}  "
                  f"cycles={n_cycles}  arities={n_arities}")
            rows.append(dict(
                dataset=dataset, device=device_name, model="HSiKAN",
                n_test_queries=n_te, n_nodes=int(n_nodes),
                setup_ms=setup_hk * 1000,
                fwd_med_ms=hk_med * 1000,
                per_query_us=hk_per_q_us,
                n_params=n_params_hk,
                n_cycles=n_cycles, n_arities=n_arities,
                max_k=cfg["max_k"],
            ))
            del hk_model, per_arity_test
            if device_name == "cuda":
                torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")
    print("\n--- Summary ---")
    print(f"{'dataset':<16} {'device':<6} {'model':<7} "
          f"{'fwd(ms)':>9} {'per-q(us)':>10} {'×SGCN':>7}")
    by_key = {}
    for r in rows:
        by_key[(r["dataset"], r["device"], r["model"])] = r
    for (dataset, _) in CONFIGS.items().__iter__() if False else [(d, c) for d, c in CONFIGS.items()]:
        for device_name in ("cpu", "cuda"):
            if (dataset, device_name, "SGCN") not in by_key:
                continue
            sg = by_key[(dataset, device_name, "SGCN")]
            hk = by_key[(dataset, device_name, "HSiKAN")]
            ratio = hk["fwd_med_ms"] / sg["fwd_med_ms"]
            print(f"{dataset:<16} {device_name:<6} SGCN    "
                  f"{sg['fwd_med_ms']:>9.2f} {sg['per_query_us']:>10.2f} "
                  f"{'1.00×':>7}")
            print(f"{dataset:<16} {device_name:<6} HSiKAN  "
                  f"{hk['fwd_med_ms']:>9.2f} {hk['per_query_us']:>10.2f} "
                  f"{ratio:>6.2f}×")


if __name__ == "__main__":
    main()
