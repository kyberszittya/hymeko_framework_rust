"""Structural channel pruning sweep — Slashdot, MixedAritySignedKAN
SOTA-config edition.

Companion to ``run_structural_prune.py``. The base script uses single-
arity SignedKAN which doesn't train on Slashdot (collapses to majority,
AUC ~0.56). The published HSiKAN-on-Slashdot setup uses:

  - Mixed arities k=3 + k=4 (αₖ-mixed), per `project_phase9_k45_sweet_spot`
  - 100k k=4 cycles cap (early-stop sampler)
  - ``edge_in_cycle`` M_e mode (the canonical published-comparison
    protocol with the σ-as-label leak — same as `run_phase2_mixed_arity`
    default)
  - Class-balanced BCE (Slashdot is ~95% positive)
  - n_layers=2, share_weights, JK-concat, highway inner skip
  - 100+ epochs

This script replicates that recipe, then sweeps hidden_dim and reports
the structural-pruning Pareto.

Usage:
    HSIKAN_TORCH_COMPILE=1 python -m signedkan_wip.experiments.runs.run_structural_prune_slashdot

Output:
    signedkan_wip/experiments/results/structural_prune_slashdot_sota.json
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

from signedkan_wip.src.datasets import load, split, deduplicate_pairs
from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.core.n_tuples import construct_k
from signedkan_wip.src.mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_vertex_to_tuples,
                                      build_edge_to_tuples)
from signedkan_wip.src.core.signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from .run_phase2_mixed_arity import _build_edge_incidence
from signedkan_wip.src.core.entropy_reg import SplineSmoothRegulariser
from signedkan_wip.src.core.participation_reg import ParticipationRegulariser, triad_degree


def time_fwd(model, per_arity_te, device, n_warmup=15, n_repeats=40):
    sync = torch.cuda.is_available() and device.type == "cuda"
    def fwd():
        with torch.no_grad():
            return model.encode_edges(per_arity_te)
    for _ in range(n_warmup):
        fwd()
        if sync: torch.cuda.synchronize()
    samples = []
    for _ in range(n_repeats):
        if sync: torch.cuda.synchronize()
        t0 = time.perf_counter()
        fwd()
        if sync: torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def build_inputs_for_seed(g, e_subset, arities, max_per_arity_dict,
                              device, seed):
    """Build (per_arity_inputs, n_tuples_total) where M_e is built
    per-arity using edge_in_cycle mode against e_subset.

    ``max_per_arity_dict``: dict[int, int] — per-k cap, e.g.
    {3: 30000, 4: 200000} (matches published phase-7 Slashdot config)."""
    per_arity = []
    for k in arities:
        cap_k = max_per_arity_dict.get(k, 30000)
        if k == 3:
            # Slashdot has 580k k=3 triads — cap aggressively for memory.
            t_k = construct(g)
        else:
            t_k = construct_k(g, k=k, max_cycles=cap_k, seed=seed,
                                early_stop=True)
        if not t_k: continue
        if len(t_k) > cap_k:
            t_k = subsample_tuples(t_k, cap_k, seed=seed)
        triad_v_np = np.array([t.v for t in t_k], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in t_k], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        M_vt = build_vertex_triad_incidence(triad_v_np, g.n_nodes, device,
                                              mode="sum")
        # edge_in_cycle M_e: M_e[q, t] = 1/|cycle-edges(t)| iff query
        # edge q is one of cycle t's k cycle-edges.
        n_t = len(t_k)
        edge_to_tuples = {}
        for ti in range(n_t):
            cyc = triad_v_np[ti]
            for j in range(k):
                u_, v_ = int(cyc[j]), int(cyc[(j + 1) % k])
                key = (min(u_, v_), max(u_, v_))
                edge_to_tuples.setdefault(key, []).append(ti)
        M_e = _build_edge_incidence(e_subset, edge_to_tuples, n_t, device,
                                      directed=False)
        per_arity.append((triad_v, triad_sigma, M_vt, M_e))
    return per_arity


def train_one(g, hidden, seed, n_epochs, device, arities=(3, 4),
                max_per_arity_dict=None,
                cycle_batch_size: int | None = None):
    if max_per_arity_dict is None:
        max_per_arity_dict = {3: 30000, 4: 200000}
    torch.manual_seed(seed); np.random.seed(seed)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    per_arity_tr = build_inputs_for_seed(g, e_tr, arities,
                                            max_per_arity_dict, device, seed)
    per_arity_te = build_inputs_for_seed(g, e_te, arities,
                                            max_per_arity_dict, device, seed)
    if not per_arity_tr or not per_arity_te:
        return None
    arities_used = tuple(arities[:len(per_arity_tr)])

    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=2, hidden_dim=hidden,
            grid=3, k=3, spline_kinds=["catmull_rom"]*2,
            init_scale=0.05, pool_mode="sum", jk_mode="concat",
            layer_norm_between=True, share_weights=True,
            inner_skip="highway", outer_skip="none", use_residual=True),
        arities=arities_used,
        init_arity_logits=tuple([0.0]*len(arities_used)),
        cycle_batch_size=cycle_batch_size)
    model = MixedAritySignedKAN(cfg).to(device)
    clf = nn.Linear(hidden * 2, 1).to(device)
    # Match phase-7 published Slashdot config: lr=5e-2, weight_decay=1e-4,
    # grad_clip=1.0, coef_smooth_lam=0.010, participation_lam=0.05.
    opt = torch.optim.Adam(list(model.parameters()) + list(clf.parameters()),
                            lr=5e-2, weight_decay=1e-4)

    y_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    y_te = (s_te == 1).astype(np.float32)
    n_pos = int(y_tr.sum().item()); n_neg = int((1 - y_tr).sum().item())
    pw = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                       device=device)

    # Regularisers (the published phase-7 defaults).
    smooth_reg = SplineSmoothRegulariser(0.010)
    part_reg = ParticipationRegulariser(lam=0.05).to(device)
    # Participation degrees from the first arity's triads (k=3 if
    # present, else k=4). triad_v_np matches construct_k output.
    first_triad_v = per_arity_tr[0][0].cpu().numpy()
    # Convert (T, k) array → list of dummy SignedNTuple-like objects
    # for triad_degree compatibility.
    class _T:
        __slots__ = ("v",)
        def __init__(self, v): self.v = v
    triad_list = [_T(v=tuple(first_triad_v[i].tolist()))
                   for i in range(first_triad_v.shape[0])]
    deg_np = triad_degree(triad_list, g.n_nodes)
    part_reg.set_degrees(deg_np)

    grad_clip = 1.0
    t0 = time.time()
    for ep in range(n_epochs):
        model.train(); clf.train()
        edge_emb = model.encode_edges(per_arity_tr)
        logits = clf(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_tr,
                                                    pos_weight=pw)
        loss = loss + smooth_reg(model.base)
        loss = loss + part_reg(model.node_embed.weight)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(clf.parameters()), grad_clip)
        opt.step()
    train_time = time.time() - t0

    model.eval(); clf.eval()
    with torch.no_grad():
        probs = torch.sigmoid(clf(model.encode_edges(per_arity_te)).squeeze(-1)).cpu().numpy()
    try: auc = roc_auc_score(y_te, probs)
    except ValueError: auc = float("nan")
    f1m = f1_score(y_te, probs > 0.5, average="macro", zero_division=0)
    lat = time_fwd(model, per_arity_te, device)
    n_params = (sum(p.numel() for p in model.parameters() if p.requires_grad)
                  + sum(p.numel() for p in clf.parameters() if p.requires_grad))
    return dict(hidden=hidden, train_time_s=train_time,
                  auc=float(auc), f1m=float(f1m),
                  fwd_latency_ms=lat, n_params=n_params,
                  arities=list(arities_used))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--hidden-sweep", nargs="+", type=int,
                    default=[32, 16, 8, 4])
    ap.add_argument("--n-epochs", type=int, default=60)
    ap.add_argument("--max-k3", type=int, default=30000)
    ap.add_argument("--max-k4", type=int, default=200000)
    ap.add_argument("--cycle-batch-size", type=int, default=10000,
                    help="Mini-batches forward over cycles to bound peak "
                         "activation memory. Matches phase-7 Slashdot config.")
    ap.add_argument("--arities", nargs="+", type=int, default=[3, 4],
                    help="Arities to mix. Slashdot published phase-7 SOTA "
                         "uses (3, 4) with max_k3=30k, max_k4=200k, "
                         "cycle_batch_size=10k.")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/structural_prune_slashdot_sota.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Loading slashdot ...")
    g = load("slashdot")
    g = deduplicate_pairs(g, merge="majority")
    print(f"  n_nodes={g.n_nodes}  n_edges={len(g.edges)}  "
          f"pos_frac={float((g.signs == 1).mean()):.3f}")

    rows = []
    for h in args.hidden_sweep:
        for seed in args.seeds:
            print(f"\n  h={h}  seed={seed}", flush=True)
            r = train_one(g, hidden=h, seed=seed,
                            n_epochs=args.n_epochs, device=device,
                            arities=tuple(args.arities),
                            max_per_arity_dict={3: args.max_k3,
                                                  4: args.max_k4},
                            cycle_batch_size=args.cycle_batch_size)
            if r is None:
                print(f"    ! skipped (no per-arity inputs)", flush=True)
                continue
            print(f"    AUC={r['auc']:.4f} F1m={r['f1m']:.4f} "
                  f"fwd={r['fwd_latency_ms']:.2f}ms params={r['n_params']} "
                  f"arities={r['arities']} train={r['train_time_s']:.1f}s",
                  flush=True)
            r["seed"] = seed
            rows.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out} ({len(rows)} rows)")

    print(f"\n--- Slashdot SOTA-config Pareto (median over {len(args.seeds)} seeds) ---")
    print(f"{'h':>4}{'auc':>8}{'f1m':>8}{'fwd_ms':>9}{'params':>11}")
    for h in args.hidden_sweep:
        cell = [r for r in rows if r["hidden"] == h]
        if not cell: continue
        auc_med = statistics.median(r["auc"] for r in cell)
        f1_med = statistics.median(r["f1m"] for r in cell)
        lat_med = statistics.median(r["fwd_latency_ms"] for r in cell)
        n_p = cell[0]["n_params"]
        print(f"{h:>4d}{auc_med:>8.4f}{f1_med:>8.4f}{lat_med:>9.2f}{n_p:>11d}")


if __name__ == "__main__":
    main()
