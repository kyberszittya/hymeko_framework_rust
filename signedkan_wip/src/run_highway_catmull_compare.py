"""Highway-SignedKAN: B-spline vs Catmull-Rom comparison with full
cost-capacity instrumentation.

Reports per (recipe, dataset, seed):
  - Test AUC, macro-$F_1$
  - Test BCE loss (the actual classification loss the model optimised)
  - Parameter count
  - Total training wall-clock (200 epochs with early stopping)
  - Inference wall-clock (forward pass over the full test split, mean
    over 10 trials with GPU sync)
  - Best-validation epoch (early-stopping landing point)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .datasets import load, split
from .hyperedges import construct
from .highway_signedkan import HighwaySignedKAN, HighwaySignedKANConfig
from .signedkan import build_vertex_triad_incidence
from .train import build_edge_to_triads
from .run_compare import build_edge_incidence
from .entropy_reg import EntropyRegulariser, EntropyRegConfig
from .participation_reg import ParticipationRegulariser, triad_degree


def _time_inference(model, triad_v_dev, triad_sigma_dev, M_test, M_vt,
                    device, n_trials: int = 10) -> float:
    """Median wall-clock for one full forward + classifier on the test
    edges. GPU-synced, so the timing reflects compute not async-launch."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_trials):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            triad_emb = model.encode_triads(triad_v_dev, triad_sigma_dev,
                                             M_vt)
            edge_emb = torch.sparse.mm(M_test, triad_emb)
            _ = model.classifier(edge_emb).squeeze(-1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _eval(model, triad_v_dev, triad_sigma_dev, edges, signs, M, M_vt, device):
    model.eval()
    with torch.no_grad():
        triad_emb = model.encode_triads(triad_v_dev, triad_sigma_dev, M_vt)
        edge_emb = torch.sparse.mm(M, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1)
        target = torch.from_numpy(
            (signs == 1).astype(np.float32)).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, target).item()
        probs = torch.sigmoid(logits).cpu().numpy()
    y = (signs == 1).astype(int)
    preds = (probs > 0.5).astype(int)
    auc = (roc_auc_score(y, probs)
           if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m), float(loss)


def run_one(spline_kind: str, dataset: str, seed: int,
             n_epochs: int = 200, hidden_dim: int = 32,
             val_every: int = 5):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load(dataset)
    triads = construct(g)
    triad_v = torch.tensor([t.v for t in triads], dtype=torch.long).to(device)
    triad_sigma = torch.tensor([t.sigma for t in triads],
                                 dtype=torch.long).to(device)
    edge_to_triads = build_edge_to_triads(triads)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_va, s_va = g.edges[va_idx], g.signs[va_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]
    n_triads = triad_v.shape[0]
    M_train = build_edge_incidence(e_tr, edge_to_triads, n_triads, device)
    M_val   = build_edge_incidence(e_va, edge_to_triads, n_triads, device)
    M_test  = build_edge_incidence(e_te, edge_to_triads, n_triads, device)
    M_vt = build_vertex_triad_incidence(triad_v.cpu().numpy(), g.n_nodes,
                                          device, mode="sum")

    cfg = HighwaySignedKANConfig(n_nodes=g.n_nodes, hidden_dim=hidden_dim,
                                   spline_kind=spline_kind)
    model = HighwaySignedKAN(cfg).to(device)
    n_pos = int((s_tr ==  1).sum()); n_neg = int((s_tr == -1).sum())
    pos_w = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                          device=device)
    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)
    ereg = EntropyRegulariser(EntropyRegConfig(
        lam_0=0.01, lam_a=1.0, lam_b=1.0, eta=5.0, target_entropy=0.5,
        kl_normalized=True, momentum=0.9))
    part_reg = ParticipationRegulariser(lam=0.05).to(device)
    deg_np = triad_degree(triads, g.n_nodes)
    part_reg.set_degrees(deg_np)

    best_val_auc = -1.0; best_state = None; best_epoch = -1
    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        triad_emb = model.encode_triads(triad_v, triad_sigma, M_vt)
        edge_emb = torch.sparse.mm(M_train, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits, target_tr, pos_weight=pos_w)
        loss = loss + ereg(model.node_embed.weight)
        loss = loss + part_reg(model.node_embed.weight)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % val_every == 0 or epoch == n_epochs - 1:
            v_auc, _, _ = _eval(model, triad_v, triad_sigma,
                                 e_va, s_va, M_val, M_vt, device)
            if v_auc > best_val_auc:
                best_val_auc = v_auc
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    test_auc, test_f1m, test_loss = _eval(
        model, triad_v, triad_sigma, e_te, s_te, M_test, M_vt, device)
    inf_ms = 1000.0 * _time_inference(
        model, triad_v, triad_sigma, M_test, M_vt, device, n_trials=10)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(
        spline_kind=spline_kind, dataset=dataset, seed=seed,
        test_auc=test_auc, test_f1m=test_f1m, test_loss=test_loss,
        n_params=n_params, train_seconds=train_time,
        inference_ms=inf_ms, best_epoch=best_epoch,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/highway_catmull.json")
    args = ap.parse_args()

    results = []
    for spline_kind in ("bspline", "catmull_rom"):
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_one(spline_kind, dataset, seed,
                             n_epochs=args.n_epochs)
                tag = f"HSiKAN-{'BS' if spline_kind=='bspline' else 'CR'}"
                r["cfg"] = tag
                print(f"  {tag:12s} {dataset:14s} seed={seed}  "
                      f"AUC={r['test_auc']:.4f}  F1m={r['test_f1m']:.4f}  "
                      f"loss={r['test_loss']:.4f}  "
                      f"params={r['n_params']:,}  "
                      f"train={r['train_seconds']:.1f}s  "
                      f"inf={r['inference_ms']:.2f}ms")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
