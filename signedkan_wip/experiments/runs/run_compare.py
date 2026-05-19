"""Phase 3.5 — comparison run: SignedKAN vs sign-blind VanillaKAN
on the same Bitcoin-Alpha and Bitcoin-OTC fixtures, same hidden
size, same seeds. Emits a single JSON file with paired results.

The hypothesis tested:
  - SignedKAN wins on macro-F1 (the class-balanced metric where
    sign asymmetry matters).
  - VanillaKAN matches or exceeds SignedKAN on AUC (where the
    dominant positive class makes ranking easy).

Run:
    python -m signedkan_wip.experiments.runs.run_compare \\
        --datasets bitcoin_alpha bitcoin_otc --hidden 16 \\
        --seeds 0 1 2 --n-epochs 100
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
from signedkan_wip.src.hyperedges import construct
from signedkan_wip.src.signedkan import (SignedKAN, SignedKANConfig,
                         MultiLayerSignedKAN, MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from signedkan_wip.src.baselines.vanilla_kan import VanillaKAN
from signedkan_wip.src.entropy_reg import (EntropyRegulariser, EntropyRegConfig,
                            CoefEntropyRegulariser,
                            SplineSmoothRegulariser)
from signedkan_wip.src.train import build_edge_to_triads
from signedkan_wip.src.triad_loss import TriadLoss, TriadLossConfig, build_triad_pairs
from signedkan_wip.src.n_tuple_loss import (NTupleBalanceLoss, NTupleBalanceLossConfig,
                            build_ntuple_balance_tensors)
from signedkan_wip.src.signed_laplacian import make_spectral_init
from signedkan_wip.src.cross_branch_reg import CrossBranchRegulariser, CrossBranchRegConfig
from signedkan_wip.src.attention import (SignedTriadAttention, attention_entropy_loss,
                          build_attention_pairs)
from signedkan_wip.src.participation_reg import (ParticipationRegulariser, triad_degree,
                                  HyperedgeDensityRegulariser, triad_density)


def build_edge_incidence(edges_array: np.ndarray, edge_to_triads: dict,
                          n_triads: int, device: torch.device) -> torch.Tensor:
    rows, cols, vals = [], [], []
    for ei, e in enumerate(edges_array):
        key = (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
        tri_ids = edge_to_triads.get(key, [])
        if not tri_ids:
            continue
        w = 1.0 / float(len(tri_ids))
        for t in tri_ids:
            rows.append(ei); cols.append(int(t)); vals.append(w)
    if not rows:
        return torch.zeros((edges_array.shape[0], n_triads), device=device)
    idx = torch.tensor([rows, cols], dtype=torch.long, device=device)
    v = torch.tensor(vals, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        idx, v, (edges_array.shape[0], n_triads)
    ).coalesce()


def evaluate(model, triad_v, triad_sigma, edges, signs, M, device,
             M_vt=None, edge_idx_t=None,
             attn_module=None, attn_pairs=None):
    model.eval()
    need_h_v = (getattr(model, "bilinear", None) is not None
                or attn_module is not None)
    with torch.no_grad():
        if isinstance(model, MultiLayerSignedKAN):
            out = model.encode_triads(triad_v.to(device),
                                       triad_sigma.to(device),
                                       M_vt, return_h_v=need_h_v)
        else:
            out = model.encode_triads(triad_v.to(device),
                                       triad_sigma.to(device),
                                       return_h_v=need_h_v)
        triad_emb, h_v_final = out if need_h_v else (out, None)
        if attn_module is not None and attn_pairs is not None:
            if h_v_final is None:
                h_v_final = model.node_embed.weight
            edge_emb, _ = attn_module(
                h_v_final, triad_emb, edge_idx_t,
                attn_pairs[0], attn_pairs[1], edge_idx_t.shape[0],
            )
        else:
            edge_emb = torch.sparse.mm(M, triad_emb)
        edge_logits = model.classifier(edge_emb).squeeze(-1)
        if getattr(model, "bilinear", None) is not None and edge_idx_t is not None:
            edge_logits = edge_logits + model.bilinear(
                h_v_final[edge_idx_t[:, 0]], h_v_final[edge_idx_t[:, 1]],
            )
        logits = edge_logits.cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds_pos = (probs > 0.5).astype(int)
    y = (signs == 1).astype(int)
    auc = (roc_auc_score(y, probs)
           if len(np.unique(y)) > 1 else float("nan"))
    f1_bin = f1_score(y, preds_pos, average="binary", zero_division=0)
    f1_mac = f1_score(y, preds_pos, average="macro",  zero_division=0)
    return dict(auc=auc, f1_binary=f1_bin, f1_macro=f1_mac)


def run_one(model_name: str, dataset: str, hidden: int, seed: int,
            n_epochs: int, lr: float = 5e-2,
            entropy_lam0: float = 0.0,
            entropy_eta:  float = 5.0,
            entropy_target: float = 0.5,
            entropy_lam_a: float = 1.0,
            entropy_lam_b: float = 1.0,
            entropy_kl_normalized: bool = False,
            entropy_momentum: float = 0.0,
            entropy_stride: int = 1,
            entropy_lam_kl: float = 0.0,
            coef_entropy_lam: float = 0.0,
            coef_entropy_target: float = 0.5,
            coef_smooth_lam: float = 0.0,
            early_stopping: bool = False,
            val_every: int = 5,
            grid: int = 5,
            class_weighted: bool = False,
            spline_kind: str = "bspline",
            n_layers: int = 1,
            spline_kinds: list[str] | None = None,
            pool_mode: str = "mean",
            jk_mode: str = "last",
            minibatch: bool = False,
            batch_size: int = 256,
            steps_per_epoch: int = 20,
            triad_loss_alpha: float = 0.0,
            triad_loss_margin: float = 0.5,
            ntuple_balance_alpha: float = 0.0,
            ntuple_balance_margin: float = 0.5,
            bce_weight: float = 1.0,
            use_bilinear: bool = False,
            bilinear_rank: int = 0,
            spectral_init: bool = False,
            spectral_k: int = 16,
            cross_branch_lam: float = 0.0,
            use_attention: bool = False,
            attention_entropy_lam: float = 0.0,
            l1_lam: float = 0.0,
            spline_residual: bool = False,
            spline_highway: bool = False,
            inner_skip: str = "auto",
            outer_skip: str = "auto",
            participation_lam: float = 0.0,
            participation_deg_mode: str = "sq_max",
            density_lam: float = 0.0,
            layer_norm_between: bool = False,
            share_weights: bool = False,
            use_minus_branch: bool = True,
            optimizer_kind: str = "adam",
            grad_clip: float = 0.0,
            weight_decay: float = 1e-5,
            init_scale: float = 0.1):
    """Train one model. If entropy_lam0 > 0 the spectral-entropy
    Lyapunov-safe regulariser is added to the loss; otherwise plain
    cross-entropy.

    If early_stopping=True, evaluate on the validation split every
    val_every epochs, keep the best-val-AUC checkpoint, and restore it
    before final test evaluation. Reports best_epoch and best_val_auc
    in the result dict.

    grid: Cox-de Boor spline grid size; default 5 matches Liu et al.
    KAN reference. Step-3 pruning trial sets grid=3.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    g = load(dataset)
    triads = construct(g)
    triad_v = torch.tensor([t.v for t in triads], dtype=torch.long)
    triad_sigma = torch.tensor([t.sigma for t in triads], dtype=torch.long)
    edge_to_triads = build_edge_to_triads(triads)

    tr_idx, va_idx, te_idx = split(g, seed=seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_va, s_va = g.edges[va_idx], g.signs[va_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    # Optional spectral initialisation: top-k signed-Laplacian eigenvectors
    # seeded into the first k columns of node_embed.weight.
    spectral_eigvec_t = None
    if spectral_init:
        spec_init_np = make_spectral_init(
            g.edges, g.signs.astype(np.float64), g.n_nodes,
            hidden_dim=hidden, k=min(spectral_k, hidden),
            noise_scale=0.1,
        )
        spectral_eigvec_t = torch.from_numpy(spec_init_np).to(device)

    if model_name in ("signedkan", "signedkan_entropy"):
        if n_layers > 1:
            kinds = (spline_kinds
                     if spline_kinds is not None
                     else [spline_kind] * n_layers)
            mcfg = MultiLayerSignedKANConfig(
                n_nodes=g.n_nodes, n_layers=n_layers,
                hidden_dim=hidden, grid=grid, k=3,
                use_minus_branch=use_minus_branch,
                init_scale=init_scale,
                spline_kinds=kinds,
                pool_mode=pool_mode,
                jk_mode=jk_mode,
                use_bilinear=use_bilinear,
                bilinear_rank=bilinear_rank,
                spectral_init_eigvec=spectral_eigvec_t,
                spline_residual=spline_residual,
                spline_highway=spline_highway,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                layer_norm_between=layer_norm_between,
                share_weights=share_weights,
            )
            model = MultiLayerSignedKAN(mcfg).to(device)
        else:
            cfg = SignedKANConfig(n_nodes=g.n_nodes, hidden_dim=hidden,
                                  grid=grid, k=3, spline_kind=spline_kind,
                                  use_minus_branch=use_minus_branch,
                                  init_scale=init_scale,
                                  use_bilinear=use_bilinear,
                                  bilinear_rank=bilinear_rank,
                                  spectral_init_eigvec=spectral_eigvec_t,
                                  spline_residual=spline_residual,
                                  spline_highway=spline_highway,
                                  inner_skip=inner_skip,
                                  outer_skip=outer_skip)
            model = SignedKAN(cfg).to(device)
    elif model_name == "vanillakan":
        model = VanillaKAN(n_nodes=g.n_nodes, hidden_dim=hidden,
                           grid=grid, k=3).to(device)
    else:
        raise ValueError(f"unknown model: {model_name}")

    use_entropy = (model_name == "signedkan_entropy" or entropy_lam0 > 0.0)
    if use_entropy:
        ereg = EntropyRegulariser(EntropyRegConfig(
            lam_0=max(entropy_lam0, 0.01),
            lam_a=entropy_lam_a, lam_b=entropy_lam_b,
            eta=entropy_eta, target_entropy=entropy_target,
            kl_normalized=entropy_kl_normalized,
            momentum=entropy_momentum,
            stride=entropy_stride,
            lam_KL=entropy_lam_kl,
        ))
    else:
        ereg = None

    # Tier 3 / A: spectral entropy on spline coefficients (S·C, G).
    # Independent state from the embedding-side ereg so the schedules
    # do not cross-contaminate.
    if coef_entropy_lam > 0.0:
        coef_ereg = CoefEntropyRegulariser(EntropyRegConfig(
            lam_0=coef_entropy_lam,
            lam_a=entropy_lam_a, lam_b=entropy_lam_b,
            eta=entropy_eta, target_entropy=coef_entropy_target,
            kl_normalized=entropy_kl_normalized,
            momentum=entropy_momentum,
            stride=entropy_stride,
        ))
    else:
        coef_ereg = None

    # Tier 6 / E: second-difference smoothness on spline coef tensors.
    coef_smooth_reg = (SplineSmoothRegulariser(coef_smooth_lam)
                        if coef_smooth_lam > 0.0 else None)

    # Build attention module first (its existence affects param count
    # and the optimiser setup below).
    if use_attention:
        # JK-concat at L>1 makes triad_emb dim = L*hidden, not hidden.
        d_t = (hidden * n_layers
               if (n_layers > 1 and jk_mode == "concat")
               else hidden)
        attn_module = SignedTriadAttention(hidden, d_t=d_t).to(device)
    else:
        attn_module = None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if attn_module is not None:
        n_params += sum(p.numel() for p in attn_module.parameters()
                         if p.requires_grad)
    all_params = list(model.parameters())
    if attn_module is not None:
        all_params += list(attn_module.parameters())
    if optimizer_kind == "adamw":
        opt = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(all_params, lr=lr, weight_decay=weight_decay)

    triad_v_dev = triad_v.to(device)
    triad_sigma_dev = triad_sigma.to(device)
    n_triads = triad_v.shape[0]
    M_train = build_edge_incidence(e_tr, edge_to_triads, n_triads, device)
    M_val   = build_edge_incidence(e_va, edge_to_triads, n_triads, device)
    M_test  = build_edge_incidence(e_te, edge_to_triads, n_triads, device)
    M_vt = (build_vertex_triad_incidence(triad_v.numpy(), g.n_nodes, device,
                                          mode=pool_mode)
            if isinstance(model, MultiLayerSignedKAN) else None)
    # Edge endpoints as tensors, for the bilinear head when enabled.
    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
    e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
    # Per-split attention pair tensors (edge_idx, triad_idx) when
    # signed-triad attention replaces the mean-pool aggregation.
    if use_attention:
        ap_tr = build_attention_pairs(e_tr, edge_to_triads)
        ap_va = build_attention_pairs(e_va, edge_to_triads)
        ap_te = build_attention_pairs(e_te, edge_to_triads)
        ap_tr = (ap_tr[0].to(device), ap_tr[1].to(device))
        ap_va = (ap_va[0].to(device), ap_va[1].to(device))
        ap_te = (ap_te[0].to(device), ap_te[1].to(device))
    else:
        ap_tr = ap_va = ap_te = None
    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    # Step 2a: class-weighted BCE. Up-weights the minority (-1) class by
    # the ratio of positive-to-negative training edges, making the
    # gradient see each class roughly equally.
    if class_weighted:
        n_pos = int((s_tr ==  1).sum())
        n_neg = int((s_tr == -1).sum())
        pos_weight_val = float(max(n_neg, 1)) / float(max(n_pos, 1))
        pos_weight_t = torch.tensor(pos_weight_val, device=device)
    else:
        pos_weight_t = None
    # Step 2b: balanced-minibatch sampler. Each step samples
    # batch_size/2 positives and batch_size/2 negatives from the
    # training edges with replacement; loss is computed only on the
    # sampled subset. The forward pass over triads stays full —
    # only the per-edge loss is subsampled.
    if minibatch:
        pos_idx_t = torch.from_numpy(
            (s_tr ==  1).nonzero()[0].astype(np.int64)).to(device)
        neg_idx_t = torch.from_numpy(
            (s_tr == -1).nonzero()[0].astype(np.int64)).to(device)
    else:
        pos_idx_t = neg_idx_t = None
    # Hypergraph tuple (triad) loss: pre-compute per-triad pair indices
    # and balance indicator. Only built when triad_loss_alpha > 0.
    cb_reg = (CrossBranchRegulariser(CrossBranchRegConfig(lam=cross_branch_lam))
              if cross_branch_lam > 0.0 else None)
    if cb_reg is not None:
        cb_reg = cb_reg.to(device)

    # Participation (R2) regulariser: penalises high-triad-degree
    # vertices' embedding magnitudes.
    if participation_lam > 0.0:
        part_reg = ParticipationRegulariser(
            lam=participation_lam,
            deg_mode=participation_deg_mode,
        ).to(device)
        deg_np = triad_degree(triads, g.n_nodes)
        part_reg.set_degrees(deg_np)
    else:
        part_reg = None

    # Hyperedge-density regulariser: penalises triads in dense
    # neighbourhoods (high vertex-overlap with other triads).
    if density_lam > 0.0:
        dens_reg = HyperedgeDensityRegulariser(lam=density_lam).to(device)
        dens_np = triad_density(triads, g.n_nodes)
        dens_reg.set_density(dens_np)
    else:
        dens_reg = None

    if triad_loss_alpha > 0.0:
        pair_idx, pair_sign, beta = build_triad_pairs(triads)
        pair_idx  = pair_idx.to(device)
        pair_sign = pair_sign.to(device)
        beta      = beta.to(device)
        triad_loss_fn = TriadLoss(TriadLossConfig(
            margin=triad_loss_margin, alpha=triad_loss_alpha,
        )).to(device)
    else:
        pair_idx = pair_sign = beta = triad_loss_fn = None

    # Arity-agnostic n-tuple balance loss (HSiKAN-side analog of SGCN's
    # extended structural balance loss). Uses Davis weak-balance β over
    # cycle edges; reduces to TriadLoss-equivalent at k=3.
    if ntuple_balance_alpha > 0.0:
        nb_pair_idx, nb_pair_sign, nb_tuple_id, nb_beta, nb_arity = (
            build_ntuple_balance_tensors(triads)
        )
        nb_pair_idx  = nb_pair_idx.to(device)
        nb_pair_sign = nb_pair_sign.to(device)
        nb_tuple_id  = nb_tuple_id.to(device)
        nb_beta      = nb_beta.to(device)
        nb_arity     = nb_arity.to(device)
        ntuple_balance_fn = NTupleBalanceLoss(NTupleBalanceLossConfig(
            margin=ntuple_balance_margin, alpha=ntuple_balance_alpha,
        )).to(device)
    else:
        ntuple_balance_fn = None
        nb_pair_idx = nb_pair_sign = nb_tuple_id = nb_beta = nb_arity = None

    # Best-checkpoint state for early stopping. Stored as a copy of
    # state_dict on CPU to keep GPU memory unrestricted; restored at
    # the end before final test evaluation.
    best_val_auc = -1.0
    best_state   = None
    best_epoch   = -1

    t0 = time.time()
    last_lam_eff = float("nan"); last_h_norm = float("nan")
    inner_steps = steps_per_epoch if minibatch else 1
    need_h_v = (use_bilinear and model.bilinear is not None)
    for epoch in range(n_epochs):
        model.train()
        for _ in range(inner_steps):
            if isinstance(model, MultiLayerSignedKAN):
                out = model.encode_triads(triad_v_dev, triad_sigma_dev,
                                           M_vt, return_h_v=need_h_v)
            else:
                out = model.encode_triads(triad_v_dev, triad_sigma_dev,
                                           return_h_v=need_h_v)
            triad_emb, h_v_final = out if need_h_v else (out, None)
            if attn_module is not None:
                # Need h_v_final for endpoint queries; compute on demand.
                if h_v_final is None:
                    h_v_final = model.node_embed.weight
                edge_emb, attn_scores = attn_module(
                    h_v_final, triad_emb, e_tr_t,
                    ap_tr[0], ap_tr[1], e_tr_t.shape[0],
                )
            else:
                edge_emb = torch.sparse.mm(M_train, triad_emb)
                attn_scores = None
            logits = model.classifier(edge_emb).squeeze(-1)
            if need_h_v:
                # Add bilinear endpoint score on the training edges.
                h_u_e = h_v_final[e_tr_t[:, 0]]
                h_v_e = h_v_final[e_tr_t[:, 1]]
                logits = logits + model.bilinear(h_u_e, h_v_e)
            if minibatch:
                # Balanced minibatch: half positives, half negatives,
                # sampled with replacement.
                n_half = batch_size // 2
                pi = pos_idx_t[torch.randint(0, len(pos_idx_t),
                                              (n_half,), device=device)]
                ni = neg_idx_t[torch.randint(0, len(neg_idx_t),
                                              (n_half,), device=device)]
                bi = torch.cat([pi, ni])
                bl = logits[bi]
                bt = target_tr[bi]
                # No pos_weight here: the batch is already balanced.
                bce = F.binary_cross_entropy_with_logits(bl, bt)
            elif pos_weight_t is not None:
                bce = F.binary_cross_entropy_with_logits(
                    logits, target_tr, pos_weight=pos_weight_t,
                )
            else:
                bce = F.binary_cross_entropy_with_logits(logits, target_tr)
            loss = bce_weight * bce
            if triad_loss_fn is not None:
                # Hypergraph tuple loss: pull balanced triads together,
                # push unbalanced apart, in node-embedding space.
                ltri = triad_loss_fn(model.node_embed.weight,
                                     pair_idx, pair_sign, beta)
                loss = loss + triad_loss_alpha * ltri
            if ntuple_balance_fn is not None:
                lnb = ntuple_balance_fn(model.node_embed.weight,
                                         nb_pair_idx, nb_pair_sign,
                                         nb_tuple_id, nb_beta, nb_arity)
                loss = loss + ntuple_balance_alpha * lnb
            if ereg is not None:
                reg = ereg(model.node_embed.weight)
                loss = loss + reg
                last_lam_eff = ereg.last_lam_eff
                last_h_norm  = ereg.last_h_norm
            if coef_ereg is not None:
                loss = loss + coef_ereg(model)
            if coef_smooth_reg is not None:
                loss = loss + coef_smooth_reg(model)
            if cb_reg is not None:
                loss = loss + cb_reg(model)
            if part_reg is not None:
                loss = loss + part_reg(model.node_embed.weight)
            if dens_reg is not None:
                # h_t was computed above (single-layer triad_emb;
                # multi-layer final-layer triad_emb).
                loss = loss + dens_reg(triad_emb)
            if l1_lam > 0.0:
                # Coefficient-level L1 to encourage sparsity during
                # training (self-pruning).
                l1 = sum(m.coef.abs().sum() for m in model.modules()
                          if hasattr(m, "coef")
                          and isinstance(m.coef, nn.Parameter)
                          and m.coef.dim() == 3)
                loss = loss + l1_lam * l1
            if attn_module is not None and attention_entropy_lam > 0.0 \
                    and attn_scores is not None:
                # Maximise per-edge attention entropy → MINIMISE
                # loss term  -lam_ae * H(alpha).
                H_attn = attention_entropy_loss(
                    attn_scores, ap_tr[0], e_tr_t.shape[0],
                )
                loss = loss - attention_entropy_lam * H_attn
            opt.zero_grad(); loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
            opt.step()

        if early_stopping and ((epoch + 1) % val_every == 0
                                or epoch == n_epochs - 1):
            v = evaluate(model, triad_v, triad_sigma,
                          e_va, s_va, M_val, device, M_vt=M_vt,
                          edge_idx_t=e_va_t,
                          attn_module=attn_module, attn_pairs=ap_va)
            v_auc = v["auc"]
            if v_auc > best_val_auc:
                best_val_auc = float(v_auc)
                best_epoch   = epoch + 1
                best_state   = {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items()}
    elapsed = time.time() - t0

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, triad_v, triad_sigma, e_te, s_te, M_test, device,
                    M_vt=M_vt, edge_idx_t=e_te_t,
                    attn_module=attn_module, attn_pairs=ap_te)
    return dict(model=model_name, dataset=dataset, hidden=hidden, seed=seed,
                lr=lr, n_epochs=n_epochs, grid=grid,
                early_stopping=early_stopping,
                class_weighted=class_weighted,
                spline_kind=spline_kind,
                n_layers=n_layers,
                spline_kinds=spline_kinds if spline_kinds else
                              ([spline_kind] * n_layers if n_layers > 1
                               else [spline_kind]),
                minibatch=minibatch,
                batch_size=batch_size if minibatch else None,
                steps_per_epoch=steps_per_epoch if minibatch else None,
                triad_loss_alpha=triad_loss_alpha,
                triad_loss_margin=triad_loss_margin,
                bce_weight=bce_weight,
                best_epoch=best_epoch if early_stopping else n_epochs,
                best_val_auc=best_val_auc if early_stopping else float("nan"),
                n_params=n_params, elapsed_s=elapsed,
                entropy_lam0=entropy_lam0 if use_entropy else 0.0,
                last_h_norm=last_h_norm, last_lam_eff=last_lam_eff,
                **{f"test_{k}": v for k, v in test.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bitcoin_alpha", "bitcoin_otc"])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--out",
                    default="signedkan_wip/experiments/results/compare.json")
    args = ap.parse_args()

    results = []
    for dataset in args.datasets:
        for model_name in ("signedkan", "vanillakan"):
            for seed in args.seeds:
                r = run_one(model_name, dataset, args.hidden, seed,
                             args.n_epochs, args.lr)
                print(f"  {model_name:12s} {dataset:14s} h={args.hidden} "
                      f"seed={seed}  AUC={r['test_auc']:.4f}  "
                      f"F1_mac={r['test_f1_macro']:.4f}  "
                      f"F1_bin={r['test_f1_binary']:.4f}  "
                      f"params={r['n_params']:,}  {r['elapsed_s']:.1f}s")
                results.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\\nwrote {out}")


if __name__ == "__main__":
    main()
