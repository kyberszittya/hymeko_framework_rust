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
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from signedkan_wip.src.baselines.cayley_rotor_baseline import (
    STRUCT_DIM,
    build_node_features,
    feature_dim,
)
from signedkan_wip.src.core.geometric_triad_attention import (
    GeometricTriadAttentionPool,
    build_vertex_triad_pairs,
)
from signedkan_wip.src.core.bilinear_head import ComplexDiagonalHead
from signedkan_wip.src.core.hyperedges import construct
from signedkan_wip.src.core.n_tuples import construct_k_arrays
from signedkan_wip.src.core.rotor_relative_head import RotorGeodesicHead, RotorRelativeHead
from signedkan_wip.src.core.signedkan import (
    MultiLayerSignedKAN,
    MultiLayerSignedKANConfig,
    build_vertex_triad_incidence,
)
from signedkan_wip.src.core.train import build_edge_to_triads
from signedkan_wip.src.datasets import (
    SignedGraph,
    drop_train_pairs,
    load,
    split,
    undirected_pair,
)
from signedkan_wip.src.embeddings.cayley_rotor import CayleyRotorEmbedding
from signedkan_wip.src.embeddings.signed_rotor_propagation import SignedRotorPropagation

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
    """structural features → Cayley-rotor → proj to ``hidden`` = ``initial_h_v``.

    ``n_refs`` is forwarded to the rotor embedding: ``n_refs >= 2`` makes the full
    rotation observable from the flattened ``e = R v0`` (route B), so the existing
    bilinear head can project all of it instead of the lossy 1-ref shadow."""

    def __init__(self, hidden: int, n_refs: int = 1, prop_rounds: int = 0,
                 prop_self_weight: float = 1.0, prop_learnable: bool = False,
                 struct_dim: int = STRUCT_DIM,
                 edges: torch.Tensor | None = None,
                 signs: torch.Tensor | None = None):
        super().__init__()
        self.emb = CayleyRotorEmbedding(n_blocks=max(1, hidden // 3),
                                        in_features=struct_dim, n_refs=n_refs)
        self.proj = nn.Linear(self.emb.embedding_dim, hidden)
        self.prop_rounds = int(prop_rounds)
        if self.prop_rounds > 0:
            if edges is None or signs is None:
                raise ValueError("prop_rounds>0 requires train edges + signs")
            # Learnable mode replaces the fixed sw with a per-block retention,
            # initialised at prop_self_weight (parity) — see SignedRotorPropagation.
            self.prop = SignedRotorPropagation(
                rounds=self.prop_rounds, self_weight=prop_self_weight,
                learnable=prop_learnable, n_blocks=self.emb.n_blocks)
            self.register_buffer("edges", edges)
            self.register_buffer("signs", signs)

    def propagated_rotors(self, struct: torch.Tensor) -> torch.Tensor:
        """The per-node rotors after optional signed propagation, ``(N, blk, 4)``
        — the rotors the model actually uses. Rotor-relative / geodesic heads
        must read *these* (not the un-propagated own-feature rotors)."""
        q = self.emb.rotors(struct)                  # (N, n_blocks, 4) own-feature rotors
        if self.prop_rounds > 0:
            # Manifold-aware signed propagation: each node absorbs its neighbours'
            # rotors on S³ before embedding (fixes the degree-only input).
            q = self.prop(q, self.edges, self.signs)
        return q

    def forward(self, struct: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb.embed_rotors(self.propagated_rotors(struct)))


# True-held-out dedup lives in the datasets layer (single source of truth, shared
# with run_baseline_audit + run_rotor_head_ablation). Kept under the historical
# private names so existing importers (tests, ablation) need no change.
_pair = undirected_pair
_drop_train_pairs = drop_train_pairs


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


@dataclass(frozen=True)
class GeomTrainedState:
    """Trained ``geom_attn`` handles handed to a ``run`` ``on_trained`` callback.

    A read-only snapshot for post-training diagnostics (e.g.
    ``geometric_triad_attention.summarise_gate``) — the pool plus the trained node
    and triad embeddings and the incidence index pairs, all in eval mode under
    ``no_grad``. Only populated for ``head == 'geom_attn'``.
    """

    pool: GeometricTriadAttentionPool
    h_v: torch.Tensor
    temb: torch.Tensor
    inc_vertex: torch.Tensor
    inc_triad: torch.Tensor
    inc_balance: torch.Tensor | None = None


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
        geom_residual: bool = True, geom_init_scale: float = 0.1,
        geom_learn_scale: bool = False, geom_sign_aware: bool = False,
        n_refs: int = 1, cycle_k: int | None = None,
        max_cycles: int | None = None, rotor_prop_rounds: int = 0,
        rotor_prop_self_weight: float = 1.0, rotor_prop_learnable: bool = False,
        feature_spec: tuple[str, ...] = ("degree",),
        cfg: TrainConfig = TrainConfig(),
        on_trained: Callable[[GeomTrainedState], None] | None = None) -> dict:
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

    # STRICT: hyperedges + features from TRAIN edges (post-shuffle) only.
    g_tr = SignedGraph(edges=e_tr, signs=s_tr, n_nodes=g.n_nodes)
    if cycle_k is not None:
        # Richer signed neighbourhoods: closed k-cycles via the existing Rust-
        # backed enumerator (no rebuild). edge_signs.prod gives the balance sign.
        v_arr, sigma_arr, es_arr = construct_k_arrays(
            g_tr, cycle_k, max_cycles=max_cycles, seed=seed)
        if v_arr.shape[0] == 0:
            raise ValueError(f"no {cycle_k}-cycles found for {dataset} seed={seed} "
                             f"(graph too sparse or cap too small)")
        triads = None
        triad_v = torch.from_numpy(v_arr.astype(np.int64)).to(device)
        triad_sigma = torch.from_numpy(sigma_arr.astype(np.int64)).to(device)
        balance = np.where(es_arr.astype(np.int64).prod(axis=1) == 1, 1.0, -1.0)
        triad_balance = torch.from_numpy(balance.astype(np.float32)).to(device)
        n_tri = int(v_arr.shape[0])
    else:
        triads = construct(g_tr)
        triad_v = torch.tensor([t.v for t in triads], dtype=torch.long, device=device)
        triad_sigma = torch.tensor([t.sigma for t in triads], dtype=torch.long, device=device)
        # Per-triad balance sign (+1 balanced / -1 unbalanced) for the sign-aware
        # readout — built from TRAIN triads only, leakage-safe like the triads.
        triad_balance = torch.tensor([1.0 if t.balanced else -1.0 for t in triads],
                                     dtype=torch.float32, device=device)
        n_tri = len(triads)

    # Endpoint heads predict from node-endpoint embeddings (reach held-out edges);
    # 'incidence' pools triads. Head ablation (real→complex→quaternion→rotation):
    #   bilinear (real), complex (ComplEx), geom_attn / rotor_rel (quaternion),
    #   rotate (RotatE/geodesic). complex REPLACES the bilinear head; rotor_rel /
    #   rotate ADD a rotor term to bilinear; geom_attn pools then bilinear.
    use_endpoint = head in {"bilinear", "geom_attn", "rotor_rel", "complex", "rotate"}
    geom = head == "geom_attn"
    rotrel = head == "rotor_rel"
    cplx = head == "complex"
    rotate = head == "rotate"
    n_layers = 2 if use_endpoint else 1
    model_cfg = MultiLayerSignedKANConfig(n_nodes=g.n_nodes, n_layers=n_layers,
                                          hidden_dim=hidden, grid=5, k=3,
                                          use_bilinear=use_endpoint and not cplx)
    model = MultiLayerSignedKAN(model_cfg).to(device)
    geom_pool = complex_head = rotate_head = None
    if use_endpoint:
        # The dead-start gamma (1e-3) is for joint training with a spline loss;
        # here the bilinear head is the only objective, so activate it.
        if model.bilinear is not None:
            model.bilinear.gamma.data.fill_(1.0)
        if cplx:
            complex_head = ComplexDiagonalHead(hidden, gamma_init=1.0).to(device)
        if rotate:
            rotate_head = RotorGeodesicHead(max(1, hidden // 3)).to(device)
        M_vt = build_vertex_triad_incidence(triad_v.cpu().numpy(), g.n_nodes, device)
        e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
        e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
        e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
        M_va = M_te = None
        if geom:
            inc_vertex, inc_triad = build_vertex_triad_pairs(triad_v)
            inc_balance = triad_balance[inc_triad]     # (P,) per-incidence b_t
            geom_pool = GeometricTriadAttentionPool(
                d_node=hidden, d_triad=hidden, hidden=hidden, d_out=hidden,
                channel=geom_channel, temperature=geom_temperature,
                score_init_scale=geom_init_scale, learn_scale=geom_learn_scale,
                sign_aware=geom_sign_aware).to(device)
    else:
        if triads is None:
            raise ValueError("--cycle-k requires an endpoint head "
                             "(bilinear/geom_attn/rotor_rel); the incidence head "
                             "needs the SignedTriad list")
        e2t = build_edge_to_triads(triads)
        M_tr = _edge_incidence(e_tr, e2t, n_tri, device)
        M_va = _edge_incidence(e_va, e2t, n_tri, device)
        M_te = _edge_incidence(e_te, e2t, n_tri, device)
        e_va_t = e_te_t = None

    _prop = embed == "rotor" and rotor_prop_rounds > 0
    prop_edges = torch.from_numpy(e_tr.astype(np.int64)).to(device) if _prop else None
    prop_signs = torch.from_numpy(np.where(s_tr > 0, 1.0, -1.0).astype(np.float32)).to(device) if _prop else None
    inj = (RotorInjector(hidden, n_refs=n_refs, prop_rounds=rotor_prop_rounds,
                         prop_self_weight=rotor_prop_self_weight,
                         prop_learnable=rotor_prop_learnable,
                         struct_dim=feature_dim(feature_spec),
                         edges=prop_edges, signs=prop_signs).to(device)
           if embed == "rotor" else None)
    # Leakage-free node input (degree by default; enrich via the feature registry).
    struct_t = (torch.from_numpy(
                    build_node_features(feature_spec, e_tr, s_tr, g.n_nodes)).to(device)
                if embed == "rotor" else None)
    rotor_rel_head = None
    if rotrel:
        if inj is None:
            raise ValueError("head='rotor_rel' requires embed='rotor' (needs the rotor q)")
        rotor_rel_head = RotorRelativeHead(inj.emb.n_blocks).to(device)
    if rotate and inj is None:
        raise ValueError("head='rotate' requires embed='rotor' (needs the rotor q)")
    extra_heads = [h for h in (geom_pool, rotor_rel_head, complex_head, rotate_head)
                   if h is not None]
    params = (list(model.parameters()) + (list(inj.parameters()) if inj else [])
              + [p for h in extra_heads for p in h.parameters()])
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    tgt = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    pos_weight = _pos_weight(s_tr, device) if cfg.class_weight else None

    def logits_for(edges_t, M):
        ihv = inj(struct_t) if inj is not None else None
        if use_endpoint:
            temb, h_v = model.encode_triads(triad_v, triad_sigma, M_vt,
                                            return_h_v=True, initial_h_v=ihv)
            if geom:
                pooled = geom_pool(h_v, temb, inc_vertex, inc_triad, inc_balance)
                # Residual (default): attention refines the rotor node embedding
                # instead of erasing it. 'replace' is kept for ablation.
                node = h_v + pooled if geom_residual else pooled
            else:
                node = h_v
            u, v = node[edges_t[:, 0]], node[edges_t[:, 1]]
            # complex REPLACES bilinear (real→complex readout on the same h_v);
            # rotor_rel / rotate ADD a rotor term on the PROPAGATED endpoint rotors.
            base = complex_head(u, v) if cplx else model.bilinear(u, v)
            if rotrel or rotate:
                q_all = inj.propagated_rotors(struct_t)
                q_u, q_v = q_all[edges_t[:, 0]], q_all[edges_t[:, 1]]
                term = rotor_rel_head if rotrel else rotate_head
                base = base + term(q_u, q_v)
            return base
        temb = model.encode_triads(triad_v, triad_sigma, None, initial_h_v=ihv)
        return model.classifier(torch.sparse.mm(M, temb)).squeeze(-1)

    modules = [model] + ([inj] if inj is not None else []) + extra_heads

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

    # Observer hook: hand the trained geom pool + embeddings to a diagnostic
    # (e.g. summarise_gate). Default None -> zero behaviour change for production.
    if on_trained is not None and geom:
        with torch.no_grad():
            ihv = inj(struct_t) if inj is not None else None
            temb_t, h_v_t = model.encode_triads(triad_v, triad_sigma, M_vt,
                                                return_h_v=True, initial_h_v=ihv)
        on_trained(GeomTrainedState(geom_pool, h_v_t, temb_t, inc_vertex,
                                    inc_triad, inc_balance))

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
                n_refs=(n_refs if embed == "rotor" else None),
                rotor_prop_rounds=(rotor_prop_rounds if embed == "rotor" else None),
                rotor_prop_self_weight=(rotor_prop_self_weight
                                        if (embed == "rotor" and rotor_prop_rounds > 0)
                                        else None),
                rotor_prop_learnable=(rotor_prop_learnable
                                      if (embed == "rotor" and rotor_prop_rounds > 0)
                                      else None),
                feature_spec=(list(feature_spec) if embed == "rotor" else None),
                cycle_k=cycle_k, max_cycles=max_cycles,
                geom_channel=(geom_channel if geom else None),
                geom_residual=(geom_residual if geom else None),
                geom_init_scale=(geom_init_scale if geom else None),
                geom_learn_scale=(geom_learn_scale if geom else None),
                geom_sign_aware=(geom_sign_aware if geom else None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--embed", choices=["table", "rotor"], default="rotor")
    ap.add_argument("--head",
                    choices=["incidence", "bilinear", "geom_attn", "rotor_rel",
                             "complex", "rotate"],
                    default="incidence",
                    help="endpoint head ablation: bilinear (real); complex "
                         "(ComplEx, replaces bilinear); geom_attn (geometric pool "
                         "+ bilinear); rotor_rel (bilinear + projected relative "
                         "rotor, QuatE); rotate (bilinear + geodesic angle, RotatE)")
    ap.add_argument("--n-refs", type=int, default=1,
                    help="rotor embedding reference vectors per block "
                         "(>=2 makes the full rotation observable from e=R·v0)")
    ap.add_argument("--rotor-prop-rounds", type=int, default=0,
                    help="rounds of signed slerp/nlerp rotor propagation over the "
                         "train graph (0=off; gives nodes a neighbourhood-aware rotor)")
    ap.add_argument("--rotor-prop-self-weight", type=float, default=1.0,
                    help="self-retention weight in rotor propagation "
                         "(higher = keep more of the node's own rotor; curbs over-smoothing)")
    ap.add_argument("--rotor-prop-learnable", action="store_true",
                    help="learn a per-block self-retention (init at --rotor-prop-self-weight; "
                         "removes the dataset-specific sw tuning)")
    ap.add_argument("--feature-spec", default="degree",
                    help="comma-separated leakage-free node features for the rotor input "
                         "(e.g. degree,walk_k3,cycle_k3; see structural_features registry)")
    ap.add_argument("--cycle-k", type=int, default=None,
                    help="pool closed signed k-cycles (k>=3) instead of triads "
                         "(reuses construct_k_arrays; endpoint head only)")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="cap on enumerated cycles before classification "
                         "(mandatory for k>3 on large graphs)")
    ap.add_argument("--geom-channel", choices=["both", "quaternion", "clifford"],
                    default="both", help="geom_attn score channel(s)")
    ap.add_argument("--geom-temperature", type=float, default=1.0)
    ap.add_argument("--geom-replace", action="store_true",
                    help="geom_attn: replace h_v with the pool (default: residual)")
    ap.add_argument("--geom-init-scale", type=float, default=0.1,
                    help="geom_attn: W_q/W_k init shrink (0.1=legacy, 1.0=woken)")
    ap.add_argument("--geom-learn-scale", action="store_true",
                    help="geom_attn: learnable log-scale on the score")
    ap.add_argument("--geom-sign-aware", action="store_true",
                    help="geom_attn: w = balance_sign * sigmoid(score) "
                         "(balance-signed relevance readout)")
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
            geom_residual=not args.geom_replace, geom_init_scale=args.geom_init_scale,
            geom_learn_scale=args.geom_learn_scale,
            geom_sign_aware=args.geom_sign_aware, n_refs=args.n_refs,
            cycle_k=args.cycle_k, max_cycles=args.max_cycles,
            rotor_prop_rounds=args.rotor_prop_rounds,
            rotor_prop_self_weight=args.rotor_prop_self_weight,
            rotor_prop_learnable=args.rotor_prop_learnable,
            feature_spec=tuple(s.strip() for s in args.feature_spec.split(",") if s.strip()),
            cfg=cfg)
    print(f"  HSiKAN[{r['embed']:5s}/{r['head']:9s} dedup={int(r['dedup'])}] "
          f"{r['dataset']:<14} seed={r['seed']} shuffle={r['shuffle']}  "
          f"AUROC={r['test_auroc']:.4f}  params={r['n_params']:,}  "
          f"n_test={r['n_test']:,}")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
