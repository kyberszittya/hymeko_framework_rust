"""Mixed-arity SignedKAN with learned per-arity mixture α_k.

Trains HSiKAN on the union of k=3, k=4, k=5 Davis-1967 weakly-balanced
hyperedges. The shared :class:`SignedKANLayer` (arity-agnostic forward)
processes each arity independently and emits a per-arity hyperedge
embedding; per-arity edge-incidence matrices map those into per-arity
edge embeddings; a learned softmax-mixture $\\alpha = \\mathrm{softmax}(\\theta)$
over arities forms the final edge embedding for classification:

    edge_emb = sum_k softmax(theta)_k * (M_edge^{(k)} @ h_e^{(k)})

The mixture is the *only* arity-specific parameter — the spline trunk
is shared, so the model has the same parameter count as single-arity
HSiKAN plus three scalars. This isolates "does higher-arity carry
complementary signal?" from "does extra capacity help?".

Storyline: single-arity k=4 was the falsification (k=4 alone matches
k=3 on Alpha, 2026-05-01). The mixture is the experiment that, if it
beats k=3, demonstrates higher-arity cycles carry complementary
information that the αₖ gate can selectively emphasise.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from signedkan_wip.src.datasets import load, split
from signedkan_wip.src.n_tuples import construct_k, stats as ntuple_stats


_CACHE_DIR = Path("data/ntuples_cache")


def _construct_k_cached(g, dataset: str, k: int):
    """Cache n-tuple enumeration (seed-independent, expensive for k>=5)."""
    import pickle
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{dataset}_k{k}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    tuples = construct_k(g, k)
    with cache_path.open("wb") as f:
        pickle.dump(tuples, f)
    return tuples
from signedkan_wip.src.signedkan import (SignedKAN, SignedKANConfig, SignedKANLayer,
                         build_vertex_triad_incidence)
from .run_compare import build_edge_incidence
from .run_ntuples import build_edge_to_ntuples


@dataclass
class MixedAritySignedKANConfig:
    n_nodes: int
    arities: tuple[int, ...] = (3, 4, 5)
    hidden_dim: int = 16
    grid: int = 5
    spline_kind: str = "catmull_rom"
    init_scale: float = 0.05
    use_minus_branch: bool = True


class MixedAritySignedKAN(nn.Module):
    """Single-layer SignedKAN with learned αₖ mixture across arities.

    Pieces:
      - one ``node_embed`` (shared across arities)
      - one ``SignedKANLayer`` (arity-agnostic forward; shared)
      - one ``classifier`` head
      - one learnable ``alpha_logits`` of length |arities|
    """

    def __init__(self, cfg: MixedAritySignedKANConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.node_embed = nn.Embedding(cfg.n_nodes, d)
        nn.init.uniform_(self.node_embed.weight, -cfg.init_scale, cfg.init_scale)
        layer_cfg = SignedKANConfig(
            n_nodes=cfg.n_nodes, hidden_dim=d, grid=cfg.grid, k=3,
            use_minus_branch=cfg.use_minus_branch,
            inner_skip="highway", outer_skip="none",
            spline_kind=cfg.spline_kind, init_scale=cfg.init_scale,
        )
        self.layer = SignedKANLayer(layer_cfg)
        self.classifier = nn.Linear(d, 1)
        self.alpha_logits = nn.Parameter(torch.zeros(len(cfg.arities)))

    def encode_per_arity(self, triad_v_per_k, triad_sigma_per_k):
        x = self.node_embed.weight                       # (V, d)
        h_e_per_k = []
        for tv, ts in zip(triad_v_per_k, triad_sigma_per_k):
            h_e_per_k.append(self.layer(x, tv, ts))      # (T_k, d)
        return h_e_per_k

    def predict(self, triad_v_per_k, triad_sigma_per_k, M_edge_per_k):
        h_e_per_k = self.encode_per_arity(triad_v_per_k, triad_sigma_per_k)
        alpha = F.softmax(self.alpha_logits, dim=0)      # (K,)
        edge_emb = None
        for a, M_e, h_e in zip(alpha, M_edge_per_k, h_e_per_k):
            ee_k = torch.sparse.mm(M_e, h_e)             # (E, d)
            edge_emb = a * ee_k if edge_emb is None else edge_emb + a * ee_k
        logits = self.classifier(edge_emb).squeeze(-1)
        return logits


def _eval(model, tv_per_k, ts_per_k, M_edge_per_k, signs):
    model.eval()
    with torch.no_grad():
        logits = model.predict(tv_per_k, ts_per_k, M_edge_per_k).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int) * 2 - 1
    y = signs.astype(int); y01 = (y == 1).astype(int)
    return {
        "auc": float(roc_auc_score(y01, probs)),
        "f1_pos": float(f1_score(y, preds, pos_label=1, zero_division=0)),
        "f1_macro": float(
            f1_score(y, preds, labels=[-1, 1], average="macro", zero_division=0)
        ),
    }


def run_mixed(dataset: str, arities: tuple[int, ...], seed: int,
               n_epochs: int = 200, lr: float = 5e-2, hidden: int = 16,
               max_tuples_per_k: int = 40000) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = load(dataset)
    print(f"  dataset={dataset} |V|={g.n_nodes} |E|={g.edges.shape[0]} arities={arities}")

    # Build per-arity n-tuple sets (with subsample cap).
    tv_per_k, ts_per_k, n_per_k, bal_per_k = [], [], [], []
    e2t_per_k = []
    for k in arities:
        t0 = time.time()
        tuples = _construct_k_cached(g, dataset, k)
        s = ntuple_stats(tuples)
        if max_tuples_per_k is not None and len(tuples) > max_tuples_per_k:
            rng = np.random.default_rng(seed + k * 17)
            keep = rng.choice(len(tuples), size=max_tuples_per_k, replace=False)
            tuples = [tuples[i] for i in keep]
        print(f"    k={k}: {len(tuples):>6d} cycles ({time.time()-t0:.1f}s, "
              f"balanced={s['balanced_frac']:.3f})")
        tv = torch.tensor([t.v for t in tuples], dtype=torch.long)
        ts = torch.tensor([t.sigma for t in tuples], dtype=torch.long)
        e2t = build_edge_to_ntuples(tuples)
        tv_per_k.append(tv.to(device))
        ts_per_k.append(ts.to(device))
        e2t_per_k.append(e2t)
        n_per_k.append(len(tuples))
        bal_per_k.append(s["balanced_frac"])

    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_va, s_va = g.edges[va_idx], g.signs[va_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    # Per-arity edge-incidence matrices.
    def _edges(edges_arr):
        return [build_edge_incidence(edges_arr, e2t, n, device)
                for e2t, n in zip(e2t_per_k, n_per_k)]
    M_edge_train = _edges(e_tr)
    M_edge_val   = _edges(e_va)
    M_edge_test  = _edges(e_te)

    cfg = MixedAritySignedKANConfig(
        n_nodes=g.n_nodes, arities=tuple(arities), hidden_dim=hidden,
        grid=5, spline_kind="catmull_rom", init_scale=0.05,
    )
    model = MixedAritySignedKAN(cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = float((s_tr == +1).sum()); n_neg = float((s_tr == -1).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    y_tr01 = torch.tensor((s_tr == +1).astype(np.float32), device=device)

    best_val_auc = -1.0; best_state = None; best_epoch = 0; patience = 0
    train_t0 = time.time()
    for ep in range(n_epochs):
        model.train(); opt.zero_grad()
        logits_tr = model.predict(tv_per_k, ts_per_k, M_edge_train)
        loss = bce(logits_tr, y_tr01)
        loss.backward(); opt.step()
        if (ep + 1) % 5 == 0 or ep == n_epochs - 1:
            v = _eval(model, tv_per_k, ts_per_k, M_edge_val, s_va)
            if v["auc"] > best_val_auc:
                best_val_auc = v["auc"]; best_epoch = ep + 1
                best_state = {k_: v_.detach().clone()
                              for k_, v_ in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break
    train_s = time.time() - train_t0
    if best_state is not None:
        model.load_state_dict(best_state)
    test = _eval(model, tv_per_k, ts_per_k, M_edge_test, s_te)
    alpha = F.softmax(model.alpha_logits.detach(), dim=0).cpu().numpy().tolist()
    return {
        "dataset": dataset, "arities": list(arities), "seed": seed,
        "n_per_k": n_per_k, "balanced_per_k": bal_per_k,
        "alpha": alpha, "best_epoch": best_epoch,
        "test_auc": test["auc"], "test_f1_pos": test["f1_pos"],
        "test_f1_macro": test["f1_macro"], "elapsed_s": round(train_s, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["bitcoin_alpha"])
    ap.add_argument("--arities", nargs="+", type=int, default=[3, 4, 5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n_epochs", type=int, default=200)
    ap.add_argument("--max_tuples_per_k", type=int, default=40000)
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/ntuples_mixed.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for seed in args.seeds:
            r = run_mixed(dataset, tuple(args.arities), seed,
                           n_epochs=args.n_epochs,
                           max_tuples_per_k=args.max_tuples_per_k)
            print(f"  {dataset:14s} arities={args.arities} seed={seed} "
                  f"AUC={r['test_auc']:.4f} F1m={r['test_f1_macro']:.4f} "
                  f"α={['%.2f'%x for x in r['alpha']]} {r['elapsed_s']:.1f}s")
            results.append(r)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
