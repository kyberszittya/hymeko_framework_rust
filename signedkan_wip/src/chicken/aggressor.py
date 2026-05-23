"""Per-vertex aggressor classifier built on HSiKAN cycle embeddings.

Architecture:

    signed graph (vertices = birds, edges = interactions)
         │
         ├─→ k-cycle enumeration (k ∈ {3, 4})
         ├─→ SignedKAN encoder           # h_t per cycle
         ├─→ M_vt · h_t                  # vertex pool — h_v per bird
         └─→ Linear → sigmoid → P(aggressor | bird)

The encoder reuses the published SignedKAN code; only the head is new
(Linear over per-vertex embedding instead of per-edge).  Trained with
BCE on ground-truth aggressor labels (synthetic data) or labels
provided by Éva (real data).

The minimum API:

    >>> g_signed = trajectories_to_signed_graph(traj, ...)[0]
    >>> y_aggressor = np.array([...], dtype=np.int64)   # 0/1 per bird
    >>> probs, model = train_aggressor_classifier(
    ...     g_signed, y_aggressor, hidden=8, n_epochs=100)
    >>> aggressor_predictions = (probs > 0.5).astype(int)

Returns the per-bird P(aggressor) array + the trained model for
inspection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from ..datasets import SignedGraph
from ..core.hyperedges import construct
from ..core.n_tuples import construct_k
from ..core.signedkan import (SignedKAN, SignedKANConfig,
                          build_vertex_triad_incidence)
from ..core.train import build_edge_to_triads


@dataclass
class AggressorTrainResult:
    auc:    float
    f1m:    float
    probs:  np.ndarray   # (n_nodes,) P(aggressor)
    train_time_s: float
    n_params: int


class HSiKANAggressorClassifier(nn.Module):
    """SignedKAN encoder + per-vertex linear head.

    Reuses ``SignedKAN.encode_triads`` to get per-cycle embeddings,
    pools them to per-vertex via ``M_vt`` (mean-aggregation), then
    runs a Linear → sigmoid head.  Total parameter count is the
    SignedKAN's plus a tiny linear (d → 1).
    """

    def __init__(self, n_nodes: int, hidden_dim: int = 8,
                 spline_kind: str = "catmull_rom"):
        super().__init__()
        cfg = SignedKANConfig(n_nodes=n_nodes, hidden_dim=hidden_dim,
                                grid=5, k=3, spline_kind=spline_kind)
        self.encoder = SignedKAN(cfg)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, triad_v: torch.Tensor,
                triad_sigma: torch.Tensor,
                M_vt: torch.Tensor) -> torch.Tensor:
        h_t = self.encoder.encode_triads(triad_v, triad_sigma)  # (T, d)
        # M_vt: (n_nodes, T) sparse CSR; mean-pools cycles to vertices.
        h_v = torch.sparse.mm(M_vt, h_t)                         # (V, d)
        return self.head(h_v).squeeze(-1)                        # (V,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


def _all_cycles(g: SignedGraph, arities=(3, 4),
                 max_cycles_per_arity: int = 5000, seed: int = 0):
    """Concatenate cycles across arities into a single (T_total, k_max)
    triad_v / triad_sigma pair via padding.  For multi-arity flow we
    pad shorter cycles to k_max — the encoder treats padded entries as
    repeated vertices, which is harmless (their σ contribution is 0
    once σ is normalised by occupancy)."""
    all_cycles = []
    for k in arities:
        if k == 3:
            tk = construct(g)
        else:
            tk = construct_k(g, k=k, max_cycles=max_cycles_per_arity,
                              seed=seed)
        if not tk:
            continue
        all_cycles.extend(tk)
    if not all_cycles:
        return None, None
    # Use the smallest fixed k for now (avoid mixed-arity padding
    # complications in this scaffold — extend later).
    k_use = min(arities)
    same_k = [t for t in all_cycles if len(t.v) == k_use]
    if not same_k:
        same_k = all_cycles      # fall through; may have mixed k
    triad_v = np.array([t.v for t in same_k], dtype=np.int64)
    triad_sigma = np.array([t.sigma for t in same_k], dtype=np.int64)
    return triad_v, triad_sigma


def train_aggressor_classifier(
    g: SignedGraph,
    y_aggressor: np.ndarray,
    hidden: int = 8,
    n_epochs: int = 100,
    lr: float = 5e-2,
    arities=(3, 4),
    seed: int = 0,
    val_frac: float = 0.0,
    device: str | torch.device | None = None,
) -> AggressorTrainResult:
    """Train an HSiKAN-based aggressor classifier on a signed graph.

    Parameters
    ----------
    g
        SignedGraph with ``n_nodes`` birds and signed interaction edges.
    y_aggressor
        Per-bird binary aggressor label (1 = aggressor, 0 = neutral),
        shape ``(n_nodes,)``.
    hidden
        SignedKAN hidden dimension.
    n_epochs
        Full-batch training iterations.
    arities
        Cycle arities to enumerate (default k=3 + k=4).
    val_frac
        If > 0, hold out this fraction of birds for validation; AUC
        is computed on the held-out set.  Default 0 means train and
        evaluate on all birds (the synthetic case where all labels
        are known).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available()
                                else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    # Cycle enumeration.
    triad_v_np, triad_sigma_np = _all_cycles(g, arities=arities,
                                               seed=seed)
    if triad_v_np is None:
        raise RuntimeError(
            "No cycles found in the signed graph; aggressor classifier "
            "needs k>=3 cycles.  Try lowering peck_radius / "
            "raising proximity_radius so more pairs interact, or "
            "running on a larger flock / longer recording."
        )
    triad_v = torch.from_numpy(triad_v_np).to(device)
    triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
    M_vt = build_vertex_triad_incidence(
        triad_v_np, g.n_nodes, device, mode="mean")

    # Train / val split.
    rng = np.random.default_rng(seed)
    n = g.n_nodes
    if val_frac > 0:
        idx = rng.permutation(n)
        n_val = int(round(n * val_frac))
        val_mask = np.zeros(n, dtype=bool)
        val_mask[idx[:n_val]] = True
        train_mask = ~val_mask
    else:
        train_mask = np.ones(n, dtype=bool)
        val_mask = train_mask
    y_t = torch.from_numpy(y_aggressor.astype(np.float32)).to(device)
    tm = torch.from_numpy(train_mask).to(device)
    vm = torch.from_numpy(val_mask).to(device)

    # Class imbalance: aggressors are typically a minority (15% in
    # synthetic).  Use BCE pos_weight = #neg / #pos.
    n_pos = int(y_aggressor[train_mask].sum())
    n_neg = int(train_mask.sum() - n_pos)
    pos_w = torch.tensor(
        float(max(n_neg, 1)) / float(max(n_pos, 1)), device=device,
    )

    model = HSiKANAggressorClassifier(
        n_nodes=g.n_nodes, hidden_dim=hidden,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                            weight_decay=1e-5)

    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        logits = model(triad_v, triad_sigma, M_vt)
        loss = F.binary_cross_entropy_with_logits(
            logits[tm], y_t[tm], pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        logits = model(triad_v, triad_sigma, M_vt)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Score on val slice (or all if val_frac=0).
    y_eval = y_aggressor[val_mask]
    p_eval = probs[val_mask]
    if len(np.unique(y_eval)) > 1:
        auc = float(roc_auc_score(y_eval, p_eval))
    else:
        auc = float("nan")
    f1m = float(f1_score(y_eval, (p_eval > 0.5).astype(int),
                           average="macro", zero_division=0))

    return AggressorTrainResult(
        auc=auc, f1m=f1m, probs=probs.astype(np.float32),
        train_time_s=train_time, n_params=model.num_parameters(),
    )


# ─── CLI smoke test ──────────────────────────────────────────────────


def main():
    import argparse
    from .simulator import ChickenFlockSim, simulate_flock
    from .interactions import (Trajectories,
                                trajectories_to_signed_graph,
                                InteractionEvent)

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-birds", type=int, default=20)
    ap.add_argument("--n-frames", type=int, default=300)
    ap.add_argument("--n-aggressors", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=120)
    ap.add_argument("--use-detector", action="store_true",
                    help="Use the kinematic peck-detector instead of "
                         "ground-truth simulator events.")
    args = ap.parse_args()

    print(f"Synthetic chicken-aggression sweep: {len(args.seeds)} seeds, "
          f"{args.n_birds} birds, {args.n_frames} frames, "
          f"{args.n_aggressors} aggressors\n")
    aucs = []; f1s = []
    for seed in args.seeds:
        cfg = ChickenFlockSim(n_birds=args.n_birds,
                                n_frames=args.n_frames,
                                n_aggressors=args.n_aggressors,
                                seed=seed)
        sim = simulate_flock(cfg)
        traj = Trajectories.from_simulator(sim)

        if args.use_detector:
            g, info = trajectories_to_signed_graph(traj)
        else:
            peck_evs = [
                InteractionEvent(e.frame, e.src, e.dst, e.type)
                for e in sim.events if e.type == "peck"
            ]
            prox_evs = [
                InteractionEvent(e.frame, e.src, e.dst, e.type)
                for e in sim.events if e.type == "proximity"
            ]
            g, info = trajectories_to_signed_graph(
                traj, peck_events=peck_evs, proximity_events=prox_evs)
        try:
            res = train_aggressor_classifier(
                g, sim.aggressor.astype(np.int64),
                hidden=args.hidden, n_epochs=args.n_epochs,
                seed=seed,
            )
            print(f"  seed={seed}: AUC={res.auc:.4f} F1m={res.f1m:.4f} "
                  f"({res.train_time_s:.1f}s, {res.n_params} params, "
                  f"graph: {g.stats()['n_edges']} edges, "
                  f"{info['n_peck_events']} pecks)")
            print(f"    aggressor probs (top-5): "
                  f"{sorted(enumerate(res.probs), key=lambda x: -x[1])[:5]}")
            print(f"    ground truth: "
                  f"{np.where(sim.aggressor)[0].tolist()}")
            aucs.append(res.auc); f1s.append(res.f1m)
        except RuntimeError as e:
            print(f"  seed={seed}: FAILED - {e}")
    if aucs:
        import statistics
        m_auc = statistics.mean(aucs)
        s_auc = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        m_f1  = statistics.mean(f1s)
        s_f1  = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        print(f"\nMean over {len(aucs)} seeds: "
              f"AUC = {m_auc:.4f} ± {s_auc:.4f}, "
              f"F1m = {m_f1:.4f} ± {s_f1:.4f}")


if __name__ == "__main__":
    main()
