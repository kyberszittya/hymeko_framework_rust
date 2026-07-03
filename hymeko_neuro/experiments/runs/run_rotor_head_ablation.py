"""Fair head-algebra ablation: compare link-prediction heads on the SAME
quaternion-native representation (the propagated rotors q), standalone.

Why a separate harness: in run_hsikan_rotor the heads were confounded — ComplEx
read a *real* encoder output (no complex structure to exploit) while the rotor
heads bypassed the triad encoder and fought a shared gradient. To isolate the one
variable that matters (the head's *algebra*), this strips everything else: every
head scores the identical propagated rotors q_u, q_v (quaternion-native +
neighbourhood-aware), with no triad encoder and no additive bilinear. Absolute
AUROC drops (no cycle structure) but the *relative* head comparison is finally
honest: does staying in complex/quaternion algebra beat a plain real readout of
the same q?

    python -m hymeko_neuro.experiments.runs.run_rotor_head_ablation \
        --dataset bitcoin_otc --head quat --seed 0 [--shuffle-train-signs]

Plan: docs/plans/2026-06-17-link-head-ablation/ (the corrected, fair variant).
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from hymeko_neuro.baselines.cayley_rotor_baseline import _structural_features
from hymeko_neuro.hyperedge.bilinear_head import BilinearHead, ComplexDiagonalHead
from hymeko_neuro.hyperedge.rotor_relative_head import (
    RotorGeodesicHead,
    RotorRelativeHead,
)
from hymeko_neuro.data.datasets import load, split

from .run_hsikan_rotor import (
    SHUFFLE_SEED_OFFSET,
    RotorInjector,
    TrainConfig,
    _drop_train_pairs,
    _optimise,
    _pair,
    _pos_weight,
)

HEADS = ("real", "complex", "quat", "geodesic")


class RotorHeadModel(nn.Module):
    """Propagated rotors → one head → edge logit. The head is the only variable.

    ``real``/``complex`` flatten q to ``(E, 4·n_blocks)`` (real bilinear / ComplEx
    over the same numbers); ``quat``/``geodesic`` use the structured ``(E, n_blocks,
    4)`` rotors (QuatE relative-rotor projection / RotatE geodesic alignment). All
    heads activate ``gamma`` (they are the sole objective, not an additive term).
    """

    def __init__(self, hidden: int, head_kind: str, prop_rounds: int,
                 prop_self_weight: float, edges: torch.Tensor, signs: torch.Tensor):
        super().__init__()
        if head_kind not in HEADS:
            raise ValueError(f"head must be one of {HEADS}; got {head_kind!r}")
        self.kind = head_kind
        self.inj = RotorInjector(hidden, prop_rounds=prop_rounds,
                                 prop_self_weight=prop_self_weight,
                                 edges=edges, signs=signs)
        nb = self.inj.emb.n_blocks
        self.head = {
            "real": lambda: BilinearHead(4 * nb, gamma_init=1.0),
            "complex": lambda: ComplexDiagonalHead(4 * nb, gamma_init=1.0),
            "quat": lambda: RotorRelativeHead(nb, gamma_init=1.0),
            "geodesic": lambda: RotorGeodesicHead(nb, gamma_init=1.0),
        }[head_kind]()

    def forward(self, struct: torch.Tensor, edges_t: torch.Tensor) -> torch.Tensor:
        q = self.inj.propagated_rotors(struct)            # (N, n_blocks, 4)
        qu, qv = q[edges_t[:, 0]], q[edges_t[:, 1]]
        if self.kind in ("real", "complex"):
            return self.head(qu.reshape(qu.shape[0], -1), qv.reshape(qv.shape[0], -1))
        return self.head(qu, qv)


def run_head(dataset: str, seed: int, head_kind: str, prop_rounds: int = 2,
             prop_self_weight: float = 4.0, shuffle: bool = False,
             hidden: int = 32, cfg: TrainConfig = TrainConfig()) -> dict:
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
    tr_pairs = {_pair(e) for e in e_tr}                    # true held-out (dedup)
    e_va, s_va = _drop_train_pairs(e_va, s_va, tr_pairs)
    e_te, s_te = _drop_train_pairs(e_te, s_te, tr_pairs)

    struct_t = torch.from_numpy(_structural_features(e_tr, s_tr, g.n_nodes)).to(device)
    edges = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    signs = torch.from_numpy(np.where(s_tr > 0, 1.0, -1.0).astype(np.float32)).to(device)
    model = RotorHeadModel(hidden, head_kind, prop_rounds, prop_self_weight,
                           edges, signs).to(device)
    e_tr_t = edges
    e_va_t = torch.from_numpy(e_va.astype(np.int64)).to(device)
    e_te_t = torch.from_numpy(e_te.astype(np.int64)).to(device)
    tgt = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    pos_weight = _pos_weight(s_tr, device) if cfg.class_weight else None
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train_loss():
        return F.binary_cross_entropy_with_logits(model(struct_t, e_tr_t), tgt,
                                                  pos_weight=pos_weight)

    def val_auroc():
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(struct_t, e_va_t)).cpu().numpy()
        yv = (s_va == 1).astype(int)
        return float(roc_auc_score(yv, p)) if len(np.unique(yv)) > 1 else float("nan")

    val_sel = _optimise(opt, train_loss, val_auroc, [model], cfg)
    with torch.no_grad():
        probs = torch.sigmoid(model(struct_t, e_te_t)).cpu().numpy()
    y = (s_te == 1).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    return dict(dataset=dataset, seed=seed, head=head_kind, shuffle=shuffle,
                prop_rounds=prop_rounds, prop_self_weight=prop_self_weight,
                test_auroc=round(float(auc), 4),
                val_auroc=(round(val_sel, 4) if not math.isnan(val_sel) else None),
                n_test=int(len(s_te)),
                n_params=int(sum(p.numel() for p in model.parameters())))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="bitcoin_otc")
    ap.add_argument("--head", choices=HEADS, default="quat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotor-prop-rounds", type=int, default=2)
    ap.add_argument("--rotor-prop-self-weight", type=float, default=4.0)
    ap.add_argument("--shuffle-train-signs", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-epochs", type=int, default=120)
    args = ap.parse_args()
    cfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, grad_clip=1.0,
                      class_weight=True, early_stop=True, n_epochs=args.n_epochs)
    r = run_head(args.dataset, args.seed, args.head,
                 prop_rounds=args.rotor_prop_rounds,
                 prop_self_weight=args.rotor_prop_self_weight,
                 shuffle=args.shuffle_train_signs, cfg=cfg)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
