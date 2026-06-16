"""HSiKAN with the inductive Cayley-rotor embedding, STRICT (train-only) triads.

The flagship signed-cycle KAN currently starts from a transductive
``nn.Embedding(n_nodes, d)`` table — the leakage channel the audit exposes. This
driver replaces it with the inductive rotor via the existing ``initial_h_v`` hook
on ``MultiLayerSignedKAN.encode_triads`` (no model surgery), making HSiKAN
inductive and leakage-free. Plan: docs/plans/2026-06-17-hsikan-rotor-leakage-free/.

LEAKAGE GATE (the load-bearing part): triads AND structural features are built from
TRAIN edges only; ``--shuffle-train-signs`` permutes the train signs before both,
and test AUROC must then fall to chance (~0.5). No headline until it does.

    python -m signedkan_wip.experiments.runs.run_hsikan_rotor \
        --dataset bitcoin_otc --embed rotor --seed 0 [--shuffle-train-signs]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from signedkan_wip.src.baselines.cayley_rotor_baseline import (
    STRUCT_DIM,
    _structural_features,
)
from signedkan_wip.src.core.geometric_triad_attention import (
    GeometricTriadAttentionPool,
    build_vertex_triad_pairs,
)
from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.core.signedkan import (
    MultiLayerSignedKAN,
    MultiLayerSignedKANConfig,
    build_vertex_triad_incidence,
)
from signedkan_wip.src.core.train import build_edge_to_triads
from signedkan_wip.src.datasets import SignedGraph, load, split
from signedkan_wip.src.embeddings.cayley_rotor import CayleyRotorEmbedding

SHUFFLE_SEED_OFFSET = 100003  # same as run_baseline_audit, for comparability


def _edge_incidence(edges, e2t, n_triads, device):
    """Sparse (E, n_triads) mean-pool incidence: M[e,t] = 1/|triads(e)|."""
    rows, cols, vals = [], [], []
    for ei, e in enumerate(edges):
        ids = e2t.get((min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))), [])
        if not ids:
            continue
        w = 1.0 / len(ids)
        for t in ids:
            rows.append(ei)
            cols.append(t)
            vals.append(w)
    if not rows:
        idx = torch.zeros((2, 0), dtype=torch.long)
        return torch.sparse_coo_tensor(idx, torch.zeros(0), (len(edges), n_triads)).to(device)
    idx = torch.tensor([rows, cols], dtype=torch.long)
    v = torch.tensor(vals, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, v, (len(edges), n_triads)).coalesce().to(device)


class RotorInjector(nn.Module):
    """structural features → Cayley-rotor → proj to ``hidden`` = ``initial_h_v``."""

    def __init__(self, hidden: int):
        super().__init__()
        self.emb = CayleyRotorEmbedding(n_blocks=max(1, hidden // 3),
                                        in_features=STRUCT_DIM)
        self.proj = nn.Linear(self.emb.embedding_dim, hidden)

    def forward(self, struct: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb(struct))


def _pair(e):
    return (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))


def _drop_train_pairs(edges, signs, tr_pairs):
    """Drop rows whose undirected (u,v) is also a train edge (true held-out set).

    Preconditions: ``len(edges) == len(signs)``; ``tr_pairs`` holds ``_pair``
    tuples. Postcondition: returned arrays are a row-aligned subset.
    """
    if len(edges) == 0:
        return edges, signs
    keep = np.array([_pair(e) not in tr_pairs for e in edges])
    return edges[keep], signs[keep]


def _pos_weight(s_tr, device):
    """Class-balancing ``pos_weight`` for ``binary_cross_entropy_with_logits``:
    ``n_neg / n_pos`` over the (post-shuffle) train signs. On the ~90%-positive
    Bitcoin fixtures this is ~0.11, lifting the rare-negative ranking signal.

    Preconditions: ``s_tr`` contains at least one positive and one negative sign;
    if not, returns ``None`` (degenerate — plain BCE).
    """
    n_pos = int((s_tr == 1).sum())
    n_neg = int(len(s_tr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    return torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation knobs for the link-prediction head.

    Defaults reproduce the 2026-06-17 verify recipe exactly (``lr=1e-3``,
    ``weight_decay=1e-5``, no grad clip, plain BCE, fixed 120 epochs, no early
    stop) so the prior ``hsikan_rotor_verify`` numbers stay reproducible. The
    non-default levers (``weight_decay=1e-4``, ``grad_clip=1.0``,
    ``class_weight``, ``early_stop``) are the ones the archived
    ``HSIKAN_GAP_CLOSING_PLAN`` measured as real AUROC wins on the
    HighwaySignedKAN family; this carries them to the leakage-free rotor driver.

    Invariant: ``n_epochs > 0`` and ``eval_every >= 1`` (asserted in ``_optimise``).
    """

    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 0.0          # 0 disables clipping
    class_weight: bool = False      # pos_weight = n_neg/n_pos in BCE
    early_stop: bool = False        # keep the best-val-AUROC state
    patience: int = 5               # early-stop patience, in eval steps
    eval_every: int = 5             # epochs between val evaluations
    n_epochs: int = 120


def _optimise(opt, train_loss, val_auroc, modules, cfg: TrainConfig) -> float:
    """Train ``modules`` for ``cfg.n_epochs``; optionally restore the best-val state.

    ``train_loss()`` runs one forward and returns the (scalar) loss tensor;
    ``val_auroc()`` returns the held-out val AUROC (eval mode handled by the
    closure). When ``cfg.early_stop`` the params are left at the epoch with the
    highest val AUROC (patience-bounded); otherwise at the final epoch.

    # Preconditions
    ``cfg.n_epochs > 0``; ``cfg.eval_every >= 1``; ``opt`` optimises exactly the
    parameters of ``modules``.
    # Postconditions
    Returns the selected (best, or last-evaluated) val AUROC, or ``nan`` if early
    stopping was off. ``modules`` are left in eval mode.
    """
    assert cfg.n_epochs > 0, "n_epochs must be positive"
    assert cfg.eval_every >= 1, "eval_every must be >= 1"
    params = [p for grp in opt.param_groups for p in grp["params"]]
    best_auc, best_state, bad = -math.inf, None, 0
    for epoch in range(cfg.n_epochs):
        for m in modules:
            m.train()
        loss = train_loss()
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if cfg.early_stop and (epoch + 1) % cfg.eval_every == 0:
            auc = val_auroc()
            if not math.isnan(auc) and auc > best_auc:
                best_auc, bad = auc, 0
                best_state = [copy.deepcopy(m.state_dict()) for m in modules]
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        for m, st in zip(modules, best_state):
            m.load_state_dict(st)
    for m in modules:
        m.eval()
    return best_auc if best_state is not None else float("nan")


def run(dataset: str, seed: int, embed: str, shuffle: bool,
        hidden: int = 32, head: str = "incidence", dedup: bool = False,
        geom_channel: str = "both", geom_temperature: float = 1.0,
        geom_residual: bool = True,
        cfg: TrainConfig = TrainConfig()) -> dict:
    """``head``: 'incidence' (edge-triad pool — can't reach held-out edges) or
    'bilinear' (node-embedding endpoint head, n_layers=2 so the M_vt pool flows
    cycle info into h_v — reaches held-out edges). ``dedup``: drop test edges
    whose (u,v) also appears in train (the true held-out set; fixes the
    split-by-index overlap leak)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load(dataset)
    tr, va, te = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr], g.signs[tr].copy()
    e_va, s_va = g.edges[va], g.signs[va]
    e_te, s_te = g.edges[te], g.signs[te]
    if shuffle:
        perm = np.random.default_rng(seed + SHUFFLE_SEED_OFFSET).permutation(len(s_tr))
        s_tr = s_tr[perm]
    if dedup:
        tr_pairs = {_pair(e) for e in e_tr}
        e_va, s_va = _drop_train_pairs(e_va, s_va, tr_pairs)
        e_te, s_te = _drop_train_pairs(e_te, s_te, tr_pairs)

    # STRICT: triads + features from TRAIN edges (post-shuffle) only.
    g_tr = SignedGraph(edges=e_tr, signs=s_tr, n_nodes=g.n_nodes)
    triads = construct(g_tr)
    triad_v = torch.tensor([t.v for t in triads], dtype=torch.long, device=device)
    triad_sigma = torch.tensor([t.sigma for t in triads], dtype=torch.long, device=device)
    n_tri = len(triads)

    # Endpoint heads ('bilinear', 'geom_attn') predict from node-endpoint
    # embeddings (reach held-out edges); 'incidence' pools triads into edges.
    use_endpoint = head in {"bilinear", "geom_attn"}
    geom = head == "geom_attn"
    n_layers = 2 if use_endpoint else 1
    model_cfg = MultiLayerSignedKANConfig(n_nodes=g.n_nodes, n_layers=n_layers,
                                          hidden_dim=hidden, grid=5, k=3,
                                          use_bilinear=use_endpoint)
    model = MultiLayerSignedKAN(model_cfg).to(device)
    geom_pool = None
    if use_endpoint:
        # The dead-start gamma (1e-3) is for joint training with a spline loss;
        # here the bilinear head is the only objective, so activate it.
        model.bilinear.gamma.data.fill_(1.0)
        M_vt = build_vertex_triad_incidence(triad_v.cpu().numpy(), g.n_nodes, device)
        e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
        e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
        e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
        M_va = M_te = None
        if geom:
            inc_vertex, inc_triad = build_vertex_triad_pairs(triad_v)
            geom_pool = GeometricTriadAttentionPool(
                d_node=hidden, d_triad=hidden, hidden=hidden, d_out=hidden,
                channel=geom_channel, temperature=geom_temperature).to(device)
    else:
        e2t = build_edge_to_triads(triads)
        M_tr = _edge_incidence(e_tr, e2t, n_tri, device)
        M_va = _edge_incidence(e_va, e2t, n_tri, device)
        M_te = _edge_incidence(e_te, e2t, n_tri, device)
        e_va_t = e_te_t = None

    inj = RotorInjector(hidden).to(device) if embed == "rotor" else None
    struct_t = (torch.from_numpy(_structural_features(e_tr, s_tr, g.n_nodes)).to(device)
                if embed == "rotor" else None)
    params = (list(model.parameters()) + (list(inj.parameters()) if inj else [])
              + (list(geom_pool.parameters()) if geom_pool is not None else []))
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    tgt = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    pos_weight = _pos_weight(s_tr, device) if cfg.class_weight else None

    def logits_for(edges_t, M):
        ihv = inj(struct_t) if inj is not None else None
        if use_endpoint:
            temb, h_v = model.encode_triads(triad_v, triad_sigma, M_vt,
                                            return_h_v=True, initial_h_v=ihv)
            if geom:
                pooled = geom_pool(h_v, temb, inc_vertex, inc_triad)
                # Residual (default): attention refines the rotor node embedding
                # instead of erasing it. 'replace' is kept for ablation.
                node = h_v + pooled if geom_residual else pooled
            else:
                node = h_v
            return model.bilinear(node[edges_t[:, 0]], node[edges_t[:, 1]])
        temb = model.encode_triads(triad_v, triad_sigma, None, initial_h_v=ihv)
        return model.classifier(torch.sparse.mm(M, temb)).squeeze(-1)

    modules = ([model] + ([inj] if inj is not None else [])
               + ([geom_pool] if geom_pool is not None else []))

    def train_loss():
        logits = logits_for(e_tr_t, None) if use_endpoint else logits_for(None, M_tr)
        return F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight)

    def val_auroc():
        for m in modules:
            m.eval()
        with torch.no_grad():
            vl = logits_for(e_va_t, None) if use_endpoint else logits_for(None, M_va)
            p = torch.sigmoid(vl).cpu().numpy()
        yv = (s_va == 1).astype(int)
        if len(np.unique(yv)) < 2:
            return float("nan")
        return float(roc_auc_score(yv, p))

    val_sel = _optimise(opt, train_loss, val_auroc, modules, cfg)

    with torch.no_grad():
        te_logits = logits_for(e_te_t, None) if use_endpoint else logits_for(None, M_te)
        probs = torch.sigmoid(te_logits).cpu().numpy()
    y = (s_te == 1).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    return dict(dataset=dataset, seed=seed, embed=embed, head=head, dedup=dedup,
                shuffle=shuffle, test_auroc=round(float(auc), 4),
                val_auroc=(round(val_sel, 4) if not math.isnan(val_sel) else None),
                n_test=int(len(s_te)),
                n_params=int(sum(p.numel() for p in params)), n_triads=n_tri,
                lr=cfg.lr, weight_decay=cfg.weight_decay, grad_clip=cfg.grad_clip,
                class_weight=cfg.class_weight, early_stop=cfg.early_stop,
                n_epochs=cfg.n_epochs,
                geom_channel=(geom_channel if geom else None),
                geom_residual=(geom_residual if geom else None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--embed", choices=["table", "rotor"], default="rotor")
    ap.add_argument("--head", choices=["incidence", "bilinear", "geom_attn"],
                    default="incidence",
                    help="bilinear = node-endpoint head; geom_attn = geometric "
                         "(quaternion+Clifford) triad-attention pool + bilinear")
    ap.add_argument("--geom-channel", choices=["both", "quaternion", "clifford"],
                    default="both", help="geom_attn score channel(s)")
    ap.add_argument("--geom-temperature", type=float, default=1.0)
    ap.add_argument("--geom-replace", action="store_true",
                    help="geom_attn: replace h_v with the pool (default: residual)")
    ap.add_argument("--dedup", action="store_true",
                    help="drop test edges whose (u,v) is also a train edge "
                         "(the true held-out set)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="max grad norm for clipping; 0 disables")
    ap.add_argument("--class-weight", action="store_true",
                    help="class-balanced BCE (pos_weight = n_neg/n_pos)")
    ap.add_argument("--early-stop", action="store_true",
                    help="restore the best val-AUROC state")
    ap.add_argument("--shuffle-train-signs", action="store_true")
    args = ap.parse_args()
    cfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay,
                      grad_clip=args.grad_clip, class_weight=args.class_weight,
                      early_stop=args.early_stop, n_epochs=args.n_epochs)
    r = run(args.dataset, args.seed, args.embed, args.shuffle_train_signs,
            head=args.head, dedup=args.dedup,
            geom_channel=args.geom_channel, geom_temperature=args.geom_temperature,
            geom_residual=not args.geom_replace,
            cfg=cfg)
    print(f"  HSiKAN[{r['embed']:5s}/{r['head']:9s} dedup={int(r['dedup'])}] "
          f"{r['dataset']:<14} seed={r['seed']} shuffle={r['shuffle']}  "
          f"AUROC={r['test_auroc']:.4f}  params={r['n_params']:,}  "
          f"n_test={r['n_test']:,}")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
