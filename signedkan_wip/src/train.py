"""SignedKAN — Phase 3: training loop + evaluation.

Minimal training harness for link sign prediction. Each epoch trains
on the full set of triads (constructed once at startup), evaluates on
val + test edge splits.

Run:
    python3 -m src.train --dataset bitcoin_alpha --hidden 32 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .datasets import SignedGraph, load, split
from .hyperedges import SignedTriad, construct
from .signedkan import SignedKAN, SignedKANConfig


@dataclass
class TrainConfig:
    dataset: str = "bitcoin_alpha"
    hidden: int = 32
    grid: int = 5
    k: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 100
    batch_size: int = 256        # candidate edges per minibatch
    seed: int = 0
    log_every: int = 10
    recompute_every_k_batches: int = 10
    out_dir: Path = Path("signedkan_wip/experiments/results")


def build_edge_to_triads(triads: list[SignedTriad]) -> dict[tuple[int,int], list[int]]:
    """Map each undirected edge (u, v) → list of triad IDs that include
    it. The classifier mean-pools over these to predict sign(u, v)."""
    out: dict[tuple[int,int], list[int]] = defaultdict(list)
    for ti, t in enumerate(triads):
        i, j, k = t.v
        for a, b in [(i, j), (j, k), (i, k)]:
            out[(min(a, b), max(a, b))].append(ti)
    return dict(out)


def evaluate(model: SignedKAN, triad_v: torch.Tensor, triad_sigma: torch.Tensor,
             edges: np.ndarray, signs: np.ndarray,
             M: torch.Tensor, device: torch.device) -> dict:
    """AUC + Binary-F1 + Macro-F1 on a held-out edge set, vectorised
    via a precomputed sparse edge↔triad incidence matrix `M`."""
    model.eval()
    with torch.no_grad():
        triad_emb = model.encode_triads(triad_v.to(device), triad_sigma.to(device))
        edge_emb = torch.sparse.mm(M, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int) * 2 - 1     # → {-1, +1}
    y = signs.astype(int)
    # AUC: treat +1 as positive class
    y01 = (y == 1).astype(int)
    auc = roc_auc_score(y01, probs) if len(np.unique(y01)) > 1 else float("nan")
    f1_bin = f1_score(y01, (preds == 1).astype(int), average="binary",
                       zero_division=0)
    f1_mac = f1_score(y01, (preds == 1).astype(int), average="macro",
                       zero_division=0)
    return dict(auc=auc, f1_binary=f1_bin, f1_macro=f1_mac)


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Data + hyperedges ─────────────────────────────────────────
    g = load(cfg.dataset)
    print(f"[data] {cfg.dataset}: {g.stats()}")
    triads = construct(g)
    print(f"[triads] {len(triads)} sign-balance triads")
    triad_v = torch.tensor([t.v for t in triads], dtype=torch.long)
    triad_sigma = torch.tensor([t.sigma for t in triads], dtype=torch.long)
    edge_to_triads = build_edge_to_triads(triads)

    # ── Splits ────────────────────────────────────────────────────
    tr_idx, va_idx, te_idx = split(g, seed=cfg.seed)
    edges_train = g.edges[tr_idx]; signs_train = g.signs[tr_idx]
    edges_val   = g.edges[va_idx]; signs_val   = g.signs[va_idx]
    edges_test  = g.edges[te_idx]; signs_test  = g.signs[te_idx]

    # ── Model ─────────────────────────────────────────────────────
    model_cfg = SignedKANConfig(n_nodes=g.n_nodes, hidden_dim=cfg.hidden,
                                 grid=cfg.grid, k=cfg.k)
    model = SignedKAN(model_cfg).to(device)
    print(f"[model] {model.num_parameters():,} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)

    # Triad tensors are static across the run — move to device once.
    triad_v_dev = triad_v.to(device)
    triad_sigma_dev = triad_sigma.to(device)

    # Build a sparse edge↔triad incidence matrix M_train: shape
    # (n_train_edges, n_triads), where M[e, t] = 1/|triads(e)| iff
    # triad t covers edge e. Then per-edge embeddings are
    #     edge_emb = M @ triad_emb       (one matmul; fully vectorised)
    # and logits = classifier(edge_emb).
    #
    # This replaces the Python loop in `predict_edge_sign` with a
    # single mat-mul, eliminating the per-edge autograd overhead that
    # dominated the previous implementation's runtime.
    def build_edge_incidence(edges_array: np.ndarray, n_triads: int) -> torch.Tensor:
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for ei, e in enumerate(edges_array):
            tri_ids = edge_to_triads.get(
                (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))), [],
            )
            if not tri_ids:
                continue
            w = 1.0 / float(len(tri_ids))
            for t in tri_ids:
                rows.append(ei)
                cols.append(int(t))
                vals.append(w)
        if not rows:
            return torch.zeros((edges_array.shape[0], n_triads), device=device)
        idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
        v = torch.tensor(vals, dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            idx, v, (edges_array.shape[0], n_triads)
        ).coalesce()

    n_triads = triad_v.shape[0]
    M_train = build_edge_incidence(edges_train, n_triads)
    M_val = build_edge_incidence(edges_val, n_triads)
    M_test = build_edge_incidence(edges_test, n_triads)

    # ── Training loop (vectorised, full-batch) ───────────────────────
    history = {"epoch": [], "train_loss": [], "val_auc": [],
               "val_f1_bin": [], "val_f1_mac": []}
    t0 = time.time()
    target_train = torch.from_numpy(
        (signs_train == 1).astype(np.float32)
    ).to(device)
    for epoch in range(cfg.n_epochs):
        model.train()
        # Single full-batch step per epoch. encode_triads is called
        # exactly once per epoch; predict is one sparse mat-mul plus one
        # linear layer.
        triad_emb = model.encode_triads(triad_v_dev, triad_sigma_dev)
        edge_emb = torch.sparse.mm(M_train, triad_emb)         # (E, d)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss = float(loss.item())
        n_batches = 1

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            val_metrics = evaluate(model, triad_v, triad_sigma,
                                    edges_val, signs_val,
                                    M_val, device)
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(epoch_loss)
            history["val_auc"].append(val_metrics["auc"])
            history["val_f1_bin"].append(val_metrics["f1_binary"])
            history["val_f1_mac"].append(val_metrics["f1_macro"])
            print(f"  epoch {epoch+1:>3}  loss={epoch_loss:.4f}  "
                  f"val_auc={val_metrics['auc']:.4f}  "
                  f"val_f1_bin={val_metrics['f1_binary']:.4f}  "
                  f"val_f1_mac={val_metrics['f1_macro']:.4f}")

    # ── Test ──────────────────────────────────────────────────────
    test_metrics = evaluate(model, triad_v, triad_sigma,
                             edges_test, signs_test,
                             M_test, device)
    elapsed = time.time() - t0
    result = dict(
        config=asdict(cfg),
        n_params=model.num_parameters(),
        n_triads=len(triads),
        elapsed_s=elapsed,
        history=history,
        test=test_metrics,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{cfg.dataset}_h{cfg.hidden}_seed{cfg.seed}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[final] test_auc={test_metrics['auc']:.4f}  "
          f"f1_bin={test_metrics['f1_binary']:.4f}  "
          f"f1_mac={test_metrics['f1_macro']:.4f}  "
          f"({elapsed:.1f}s)")
    print(f"[saved] {out_path}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_alpha",
                    choices=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--grid", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--recompute-every-k", type=int, default=10,
                    help="Recompute triad embeddings every K minibatches (default 10). "
                         "Higher values = faster training, more stale gradients.")
    args = ap.parse_args()
    cfg = TrainConfig(
        dataset=args.dataset, hidden=args.hidden, grid=args.grid, k=args.k,
        lr=args.lr, weight_decay=args.weight_decay,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        seed=args.seed, log_every=args.log_every,
        recompute_every_k_batches=args.recompute_every_k,
    )
    train(cfg)


if __name__ == "__main__":
    main()
