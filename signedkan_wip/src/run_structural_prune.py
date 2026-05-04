"""Structural channel pruning for SignedKAN.

Mask-based pruning (see ``run_prune_distill.py``) zeros coefficients
without changing the coef tensor shape, so kernel cost stays constant.
This script demonstrates the structural variant: physically reduce
``hidden_dim`` (= n_channels of the spline activations) and measure
both the latency drop AND the accuracy retention.

Two-step protocol:

  Step 1 — train a teacher at full ``hidden_dim_full`` (e.g., 32).
           Measure per-(branch, channel) L2 activity. Decide
           ``hidden_dim_kept`` from the activity quantiles (informed
           Pareto choice, not a hyperparameter guess).

  Step 2 — train students at hidden_dim ∈ a sweep around the
           informed point. Report (AUC, F1m, fwd_latency) for each
           student to expose the Pareto curve. Mask-based pruning
           is the flat baseline: same accuracy at the matching
           prune fraction, latency unchanged. Structural pruning
           gets latency down proportionally to the channel reduction.

Output:
  signedkan_wip/experiments/results/structural_prune.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .datasets import load, split
from .hyperedges import construct
from .signedkan import SignedKAN, SignedKANConfig
from .train import build_edge_to_triads
from .run_compare import build_edge_incidence
from .prune_distill import measure_activity


def time_fwd(model, triad_v, triad_sigma, M, device,
              n_warmup=15, n_repeats=40):
    sync = torch.cuda.is_available() and device.type == "cuda"
    def fwd():
        with torch.no_grad():
            te = model.encode_triads(triad_v, triad_sigma)
            ee = torch.sparse.mm(M, te)
            return model.classifier(ee)
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


def evaluate(model, triad_v, triad_sigma, M_test, signs):
    model.eval()
    with torch.no_grad():
        te = model.encode_triads(triad_v, triad_sigma)
        ee = torch.sparse.mm(M_test, te)
        logits = model.classifier(ee).squeeze(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (signs == 1).astype(int)
    preds = (probs > 0.5).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m)


def train_one(g, hidden, seed, n_epochs, device, spline_kind="catmull_rom",
                max_triads: int | None = None):
    torch.manual_seed(seed); np.random.seed(seed)
    triads = construct(g)
    if max_triads is not None and len(triads) > max_triads:
        rng = np.random.RandomState(seed)
        keep = rng.choice(len(triads), size=max_triads, replace=False)
        triads = [triads[int(i)] for i in keep]
    triad_v = torch.tensor([t.v for t in triads], dtype=torch.long).to(device)
    triad_sigma = torch.tensor([t.sigma for t in triads],
                                dtype=torch.long).to(device)
    edge_to_triads = build_edge_to_triads(triads)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    n_triads = triad_v.shape[0]
    M_train = build_edge_incidence(e_tr, edge_to_triads, n_triads, device)
    M_test  = build_edge_incidence(e_te, edge_to_triads, n_triads, device)
    cfg = SignedKANConfig(n_nodes=g.n_nodes, hidden_dim=hidden,
                           grid=5, k=3, spline_kind=spline_kind)
    model = SignedKAN(cfg).to(device)
    target = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    n_pos = int((s_tr == 1).sum()); n_neg = int((s_tr == -1).sum())
    pos_w = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                          device=device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)
    t0 = time.time()
    for _ in range(n_epochs):
        model.train()
        te_ = model.encode_triads(triad_v, triad_sigma)
        ee_ = torch.sparse.mm(M_train, te_)
        logits = model.classifier(ee_).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target,
                                                    pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0
    auc, f1m = evaluate(model, triad_v, triad_sigma, M_test, s_te)
    lat = time_fwd(model, triad_v, triad_sigma, M_test, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(model=model, hidden=hidden, train_time_s=train_time,
                  auc=auc, f1m=f1m, fwd_latency_ms=lat,
                  n_params=n_params,
                  triad_v=triad_v, triad_sigma=triad_sigma, M_test=M_test,
                  s_te=s_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["bitcoin_alpha"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden-sweep", nargs="+", type=int,
                    default=[32, 24, 16, 12, 8, 4])
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--spline-kind", default="catmull_rom")
    ap.add_argument("--max-triads", type=int, default=None,
                    help="Subsample triads for large datasets (e.g., 30000 "
                         "on Slashdot which has 580k triads).")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/structural_prune.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    rows = []
    for dataset in args.datasets:
        g = load(dataset)
        # ----- Step 1: teacher activity → informed h_kept choice -----
        print(f"\n=== {dataset} — Step 1: train teacher (h={args.hidden_sweep[0]}) "
              f"to measure activity ===")
        teacher = train_one(g, hidden=args.hidden_sweep[0], seed=0,
                              n_epochs=args.n_epochs, device=device,
                              spline_kind=args.spline_kind,
                              max_triads=args.max_triads)
        print(f"  teacher: AUC={teacher['auc']:.4f} F1m={teacher['f1m']:.4f} "
              f"fwd={teacher['fwd_latency_ms']:.2f}ms params={teacher['n_params']}")
        # Per-channel activity = max over branches of L2 norm.
        # Channel c is "alive" if EITHER branch has activity > τ.
        inner_act = measure_activity(teacher["model"].layer.inner)  # (S, C)
        outer_act = measure_activity(teacher["model"].layer.outer)
        per_ch_inner = inner_act.max(dim=0).values             # (C,)
        per_ch_outer = outer_act.max(dim=0).values             # (C,)
        per_ch = torch.maximum(per_ch_inner, per_ch_outer)     # (C,)
        # Sort thresholds for the recommended-keep curve.
        sorted_act, _ = torch.sort(per_ch, descending=True)
        # Recommend hidden_dim_kept = #channels with activity > 0.5 of max
        thresh = float(sorted_act[0]) * 0.1
        h_kept_recommended = int((per_ch > thresh).sum().item())
        h_kept_recommended = max(4, min(args.hidden_sweep[0],
                                          h_kept_recommended))
        print(f"  per-channel activity quantiles (top 10): "
              f"{[round(float(x), 3) for x in sorted_act[:10].tolist()]}")
        print(f"  recommended h_kept (activity > 0.1 × max): "
              f"{h_kept_recommended}")
        del teacher  # free GPU mem

        # ----- Step 2: sweep — train students at each h, time + eval -----
        for h in args.hidden_sweep:
            for seed in args.seeds:
                print(f"\n  h={h}  seed={seed}", end=" ")
                r = train_one(g, hidden=h, seed=seed,
                                n_epochs=args.n_epochs,
                                device=device,
                                spline_kind=args.spline_kind,
                                max_triads=args.max_triads)
                print(f"AUC={r['auc']:.4f} F1m={r['f1m']:.4f} "
                      f"fwd={r['fwd_latency_ms']:.2f}ms "
                      f"params={r['n_params']} "
                      f"train={r['train_time_s']:.1f}s", flush=True)
                rows.append(dict(
                    dataset=dataset, seed=seed, hidden=h,
                    auc=r["auc"], f1m=r["f1m"],
                    fwd_latency_ms=r["fwd_latency_ms"],
                    n_params=r["n_params"],
                    train_time_s=r["train_time_s"],
                    h_kept_recommended=h_kept_recommended,
                ))
                del r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out} ({len(rows)} rows)")

    # Pareto summary
    print(f"\n--- Pareto summary (median over {len(args.seeds)} seeds) ---")
    print(f"{'dataset':<14}{'h':>4}{'auc':>8}{'f1m':>8}{'fwd_ms':>9}"
          f"{'params':>9}")
    for dataset in args.datasets:
        for h in args.hidden_sweep:
            cell = [r for r in rows
                     if r["dataset"] == dataset and r["hidden"] == h]
            if not cell: continue
            auc_med = statistics.median(r["auc"] for r in cell)
            f1_med = statistics.median(r["f1m"] for r in cell)
            lat_med = statistics.median(r["fwd_latency_ms"] for r in cell)
            n_p = cell[0]["n_params"]
            print(f"{dataset:<14}{h:>4d}{auc_med:>8.4f}{f1_med:>8.4f}"
                  f"{lat_med:>9.2f}{n_p:>9d}")


if __name__ == "__main__":
    main()
