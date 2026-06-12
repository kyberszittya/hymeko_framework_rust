"""Inference-time benchmark: HSiKAN vs SGCN forward-pass latency.

Reports per-query latency on the test split for each (model, dataset, device)
cell, after a warmup pass and over N repeats. We measure ONLY the forward
pass — graph load and cycle enumeration are amortised at fit time and
reported separately as "setup".

Two HSiKAN *width* variants are timed per cell — **lean** (hidden 4) and
**joint** (hidden 16) — so the intra-family cost of width is *measured*, not
asserted (this is the slide-18 "lean vs joint" gap). The two variants differ
only parametrically, so width is a config axis, not a separate model class
(CLAUDE.md §6.5 #8); the width-independent cycle pool is built once and reused.

Honesty (INFERENCE_DEMOS Demo 2): against SGCN, HSiKAN's *absolute* forward is
heavier. The defensible claims are accuracy-per-parameter and the within-family
lean-vs-joint gap — not "faster than SGCN". The summary prints all three side
by side and says so.

Usage:
    python -m signedkan_wip.experiments.runs.run_inference_bench
    python -m signedkan_wip.experiments.runs.run_inference_bench --datasets bitcoin_alpha bitcoin_otc slashdot
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from signedkan_wip.src.datasets import load, split, deduplicate_pairs
from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.core.n_tuples import construct_k
from signedkan_wip.src.mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_vertex_to_tuples)
from signedkan_wip.src.core.signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from signedkan_wip.src.baselines.sgcn_model import SGCN, build_signed_adj
from signedkan_wip.src.baselines.sigat_model import (SiGATAttn,
                                      build_neighbour_lists)
from signedkan_wip.src.baselines.sgt import SGT, build_signed_neighbours
from signedkan_wip.src.baselines.mlp_gcn import MLPEdge
from signedkan_wip.demos.seminar.resources import peak_rss_bytes, fmt_bytes
from .run_phase2_mixed_arity import _build_edge_incidence_vertex_adj_scipy


N_REPEATS = 20
WARMUP = 5      # torch.compile needs ≥3 calls to fully warm — use 5 for safety
RSS_CAP_BYTES = 16 * 1024 ** 3      # CLAUDE.md §4 hard cap

# HSiKAN intra-family width variants. Difference is purely parametric (hidden
# width), so it is a config axis, not a separate model class (§6.5 #8). These
# are the two bars the slide-18 chart contrasts.
HSIKAN_VARIANTS = {"lean": 4, "joint": 16}

# The plan's demo cells (docs/plans/2026-06-10-seminar-demo-program): the two
# Bitcoin graphs at production scale. slashdot (100k k4 cycles) is heavier and
# opt-in via --datasets so the default run stays inside the <120 s / <3 GB
# budget.
DEFAULT_DATASETS = ("bitcoin_alpha", "bitcoin_otc")

# Per-dataset SOTA configurations (matched to the headline-table runs). The
# h_hsikan field is overridden per width variant; it stays only as a documented
# default for callers that do not sweep variants. h_baseline=32 matches the
# phase8 panel (sgcn_balance / sigat_attn at h32) so the here-measured latency
# pairs with the committed accuracy at the same width.
CONFIGS = {
    "bitcoin_alpha": dict(
        arities=(3, 4, 5), max_k=dict(k3=0, k4=20_000, k5=5_000),
        h_hsikan=16, h_sgcn=32, h_baseline=32, n_layers_sgcn=2,
    ),
    "bitcoin_otc": dict(
        arities=(3, 4, 5), max_k=dict(k3=0, k4=20_000, k5=5_000),
        h_hsikan=16, h_sgcn=32, h_baseline=32, n_layers_sgcn=2,
    ),
    "slashdot": dict(
        arities=(3, 4), max_k=dict(k3=0, k4=100_000, k5=0),
        h_hsikan=16, h_sgcn=32, h_baseline=32, n_layers_sgcn=2,
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


def _summarize(samples, n_te):
    """Latency statistics from per-call wall samples (seconds).

    CLAUDE.md §3 requires median, IQR, and worst case — not a single median.

    Preconditions: ``samples`` non-empty; ``n_te`` > 0.
    Postconditions: all times in ms, non-negative; ``fwd_worst_ms`` >=
    ``fwd_med_ms``; ``per_query_us`` is the median forward divided across the
    ``n_te`` test queries.
    """
    assert samples, "no timing samples"
    assert n_te > 0, "n_te must be positive"
    med = statistics.median(samples)
    q1, q3 = (float(x) for x in np.percentile(samples, [25, 75]))
    return dict(
        fwd_med_ms=med * 1e3,
        fwd_iqr_ms=(q3 - q1) * 1e3,
        fwd_worst_ms=max(samples) * 1e3,
        per_query_us=med * 1e6 / n_te,
    )


def build_hsikan_data(dataset, cfg, seed, device):
    """Build the width-independent HSiKAN cycle pool + per-arity incidence.

    Reused across every width variant for one (dataset, device) so the O(cycles)
    enumeration is paid once. Returns ``(per_arity_test, n_te, setup_s, n_nodes,
    arities_used)``.

    Preconditions: ``dataset`` in CONFIGS; ``cfg`` carries ``arities`` and
    ``max_k``. Postconditions: ``per_arity_test`` has one tuple per used arity;
    ``setup_s`` >= 0.
    """
    g = load(dataset)
    g = deduplicate_pairs(g, merge="majority")
    _tr_idx, _va_idx, te_idx = split(g, seed=seed)
    e_te = g.edges[te_idx]
    g_features = g  # we time inference; cycle pool from full graph is fine.

    per_arity_tuples = []
    setup_t0 = time.perf_counter()
    for k in cfg["arities"]:
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

    per_arity_test = []
    for _k, tuples in per_arity_tuples:
        triad_v_np = np.array([t.v for t in tuples], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(
            np.array([t.sigma for t in tuples], dtype=np.int64)).to(device)
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
    return per_arity_test, e_te.shape[0], setup_s, g.n_nodes, arities_used


def build_hsikan_model(arities_used, n_nodes, h_hsikan, device):
    """Construct an eval-mode HSiKAN of the given hidden width.

    Width is the only axis that separates lean from joint — a parametric
    difference, so config rather than a class (CLAUDE.md §6.5 #8).

    Preconditions: ``arities_used`` non-empty; ``h_hsikan`` >= 1.
    Postconditions: returned model is on ``device`` and in ``eval`` mode.
    """
    model_cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=n_nodes, n_layers=2, hidden_dim=h_hsikan,
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
    return model


def hsikan_forward(model, per_arity_test):
    with torch.no_grad():
        edge_emb = model.encode_edges(per_arity_test)
        return model.classifier(edge_emb)


# --------------------------------------------------------------------------- #
# Baseline registry (Strategy). Every external baseline shares the same shape:  #
# load+split → build a forward thunk → time it. A registry of builders replaces #
# the per-model near-copies (CLAUDE.md §6.5 #1/#9). Each builder returns a      #
# BenchUnit; the bench loop is model-agnostic.                                  #
# --------------------------------------------------------------------------- #
@dataclass
class BenchUnit:
    """One timed forward: a zero-arg thunk + the metadata for its result row."""
    forward: Callable[[], object]
    n_te: int
    setup_s: float
    n_nodes: int
    n_params: int


def _load_split_edges(dataset, seed, device):
    """Shared load → dedup → split, returning train/test edges+signs + edges_t."""
    g = load(dataset)
    g = deduplicate_pairs(g, merge="majority")
    tr_idx, _va_idx, te_idx = split(g, seed=seed)
    e_te = g.edges[te_idx]
    edges_t = torch.tensor(e_te, dtype=torch.long, device=device)
    return g, tr_idx, te_idx, edges_t


def _no_grad_edge_logits(model, ctx, edges_t):
    """Uniform encode→edge_logits forward, used by the node-encoding baselines."""
    def thunk():
        with torch.no_grad():
            return model.edge_logits(model.encode_nodes(*ctx), edges_t)
    return thunk


def build_sgcn_unit(dataset, cfg, seed, device):
    g, tr_idx, _te_idx, edges_t = _load_split_edges(dataset, seed, device)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    setup_t0 = time.perf_counter()
    A_pos, A_neg = build_signed_adj(e_tr, s_tr, g.n_nodes, device)
    setup_s = time.perf_counter() - setup_t0
    model = SGCN(n_nodes=g.n_nodes, hidden_dim=cfg["h_sgcn"],
                 n_layers=cfg["n_layers_sgcn"]).to(device).eval()
    return BenchUnit(_no_grad_edge_logits(model, (A_pos, A_neg), edges_t),
                     edges_t.shape[0], setup_s, g.n_nodes, model.num_parameters())


def build_sigat_unit(dataset, cfg, seed, device):
    g, tr_idx, _te_idx, edges_t = _load_split_edges(dataset, seed, device)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    setup_t0 = time.perf_counter()
    pos_buckets, neg_buckets = build_neighbour_lists(e_tr, s_tr, g.n_nodes)
    setup_s = time.perf_counter() - setup_t0
    model = SiGATAttn(n_nodes=g.n_nodes, hidden_dim=cfg["h_baseline"]).to(device).eval()
    return BenchUnit(_no_grad_edge_logits(model, (pos_buckets, neg_buckets), edges_t),
                     edges_t.shape[0], setup_s, g.n_nodes, model.num_parameters())


def build_sgt_unit(dataset, cfg, seed, device):
    g, tr_idx, _te_idx, edges_t = _load_split_edges(dataset, seed, device)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    setup_t0 = time.perf_counter()
    nbrs, sgns = build_signed_neighbours(e_tr, s_tr, g.n_nodes)
    setup_s = time.perf_counter() - setup_t0
    model = SGT(n_nodes=g.n_nodes, hidden_dim=cfg["h_baseline"]).to(device).eval()
    return BenchUnit(_no_grad_edge_logits(model, (nbrs, sgns), edges_t),
                     edges_t.shape[0], setup_s, g.n_nodes, model.num_parameters())


def build_mlp_unit(dataset, cfg, seed, device):
    g, _tr_idx, _te_idx, edges_t = _load_split_edges(dataset, seed, device)
    model = MLPEdge(n_nodes=g.n_nodes, hidden_dim=cfg["h_baseline"]).to(device).eval()

    def thunk():
        with torch.no_grad():
            return model(edges_t)
    return BenchUnit(thunk, edges_t.shape[0], 0.0, g.n_nodes,
                     model.num_parameters())


# Display name → builder. GCN-blind needs an unsigned-adjacency builder that is
# not on disk, so it is Pareto/table-only (accuracy+params) and absent here.
BASELINES = {
    "SGCN": build_sgcn_unit,
    "SiGAT": build_sigat_unit,
    "SGT": build_sgt_unit,
    "MLP-blind": build_mlp_unit,
}


def _bench_baselines(dataset, cfg, device_name, device):
    """Time every registered baseline for one cell; return their result rows."""
    cell_rows = []
    for name, builder in BASELINES.items():
        unit = builder(dataset, cfg, seed=0, device=device)
        stats = _summarize(_time_block(
            unit.forward, sync_cuda=(device_name == "cuda"),
        ), unit.n_te)
        print(f"  {name:<11} setup={unit.setup_s*1000:7.1f}ms  "
              f"fwd={stats['fwd_med_ms']:7.2f}ms  "
              f"per-query={stats['per_query_us']:7.2f}us  params={unit.n_params}")
        cell_rows.append(dict(
            dataset=dataset, device=device_name, model=name,
            n_test_queries=unit.n_te, n_nodes=int(unit.n_nodes),
            setup_ms=unit.setup_s * 1000, **stats, n_params=unit.n_params,
        ))
        if device_name == "cuda":
            torch.cuda.empty_cache()
    return cell_rows


def _bench_hsikan(dataset, cfg, device_name, device):
    """Time every HSiKAN width variant for one cell; return their result rows.

    The width-independent cycle pool is built once and reused across variants;
    the per-variant model is local to the loop body so it frees each iteration.
    """
    per_arity_test, n_te, setup_hk, n_nodes, arities_used = \
        build_hsikan_data(dataset, cfg, seed=0, device=device)
    n_cycles = sum(t[0].shape[0] for t in per_arity_test)
    cell_rows = []
    for variant, h in HSIKAN_VARIANTS.items():
        hk_model = build_hsikan_model(arities_used, n_nodes, h, device)
        stats = _summarize(_time_block(
            lambda model=hk_model: hsikan_forward(model, per_arity_test),
            sync_cuda=(device_name == "cuda"),
        ), n_te)
        n_params_hk = hk_model.num_parameters()
        print(f"  HSiKAN-{variant:<5} setup={setup_hk*1000:7.1f}ms  "
              f"fwd={stats['fwd_med_ms']:7.2f}ms  "
              f"per-query={stats['per_query_us']:7.2f}us  params={n_params_hk}  "
              f"cycles={n_cycles}  h={h}")
        cell_rows.append(dict(
            dataset=dataset, device=device_name, model=f"HSiKAN-{variant}",
            variant=variant, hidden_dim=h,
            n_test_queries=n_te, n_nodes=int(n_nodes),
            setup_ms=setup_hk * 1000, **stats, n_params=n_params_hk,
            n_cycles=n_cycles, n_arities=len(arities_used), max_k=cfg["max_k"],
        ))
        if device_name == "cuda":
            torch.cuda.empty_cache()
    return cell_rows


def _bench_cell(dataset, cfg, device_name, rows):
    """Bench every registered baseline + HSiKAN width variant for one cell.

    Appends one row per model to ``rows`` and prints a per-model line.
    """
    device = torch.device(device_name)
    print(f"\n=== {dataset} on {device_name} ===")
    rows.extend(_bench_baselines(dataset, cfg, device_name, device))
    if device_name == "cuda":
        torch.cuda.empty_cache()
    rows.extend(_bench_hsikan(dataset, cfg, device_name, device))
    if device_name == "cuda":
        torch.cuda.empty_cache()


def _print_summary(rows):
    """All benched models per cell, with the within-family gap + honest note."""
    by_key = {(r["dataset"], r["device"], r["model"]): r for r in rows}
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    # Stable print order: baselines, then HSiKAN width variants.
    order = list(BASELINES) + ["HSiKAN-lean", "HSiKAN-joint"]
    print("\n--- Summary (median forward ms) ---")
    print(f"{'dataset':<14} {'dev':<5} {'model':<13} {'fwd(ms)':>9} "
          f"{'IQR':>7} {'worst':>8} {'per-q(us)':>10} {'params':>8}")
    for dataset in datasets:
        for device_name in ("cpu", "cuda"):
            for model in order:
                r = by_key.get((dataset, device_name, model))
                if r is None:
                    continue
                print(f"{r['dataset']:<14} {device_name:<5} {r['model']:<13} "
                      f"{r['fwd_med_ms']:>9.2f} {r['fwd_iqr_ms']:>7.2f} "
                      f"{r['fwd_worst_ms']:>8.2f} {r['per_query_us']:>10.2f} "
                      f"{r['n_params']:>8}")
            lean = by_key.get((dataset, device_name, "HSiKAN-lean"))
            joint = by_key.get((dataset, device_name, "HSiKAN-joint"))
            if lean and joint:
                gap = joint["fwd_med_ms"] / lean["fwd_med_ms"]
                print(f"  -> {dataset}/{device_name}: joint/lean = {gap:.1f}× "
                      f"(within-family cost of width 16 vs 4)")
    # Honest note (INFERENCE_DEMOS Demo 2): SGCN/MLP-blind are the light ones;
    # SiGAT and SGT are heavier than HSiKAN-lean. The HSiKAN claim is not raw
    # forward speed.
    print("\nHonest note: SGCN and MLP-blind have the lightest absolute forward "
          "(SiGAT/SGT are heavier than HSiKAN-lean). HSiKAN's claims are "
          "accuracy-per-parameter and the lean-vs-joint gap above, not raw speed.")


def main(datasets=DEFAULT_DATASETS, out_path=None):
    """Run the bench over ``datasets`` and emit the json.

    Preconditions: every name in ``datasets`` is a key of CONFIGS.
    Postconditions: writes the json; prints peak RSS + wall; asserts RSS cap.
    """
    bad = [d for d in datasets if d not in CONFIGS]
    assert not bad, f"unknown datasets {bad}; known: {list(CONFIGS)}"
    out_path = Path(out_path) if out_path else Path(
        "signedkan_wip/experiments/results/inference_bench.json")
    print(f"Datasets: {list(datasets)}  (slashdot opt-in via --datasets)")

    wall_t0 = time.perf_counter()
    rows = []
    for dataset in datasets:
        cfg = CONFIGS[dataset]
        for device_name in ("cpu", "cuda" if torch.cuda.is_available() else None):
            if device_name is None:
                continue
            _bench_cell(dataset, cfg, device_name, rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out_path}")
    _print_summary(rows)

    peak = peak_rss_bytes()
    wall = time.perf_counter() - wall_t0
    print(f"\nPeak RSS: {fmt_bytes(peak)}  |  wall: {wall:.1f}s")
    assert peak is None or peak < RSS_CAP_BYTES, \
        f"peak RSS {fmt_bytes(peak)} exceeds 16 GB cap"
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HSiKAN vs SGCN forward bench")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS),
                        choices=list(CONFIGS),
                        help="datasets to bench (default: the two Bitcoin graphs)")
    parser.add_argument("--out", default=None, help="output json path")
    a = parser.parse_args()
    main(datasets=tuple(a.datasets), out_path=a.out)
