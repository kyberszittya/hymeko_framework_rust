"""Unified label-shuffle audit runner for every signed-link baseline.

One CLI (§6.5 #13), one strict train/eval loop, ``--model`` selects the strategy
from :mod:`signedkan_wip.src.baselines.registry`. This replaces the would-be
``run_<method>.py`` proliferation (§6.5 #1/#3): the loop here is the *only* place
training, early-stopping, and AUROC eval live for the seven baselines.

Protocol (strict + audit, matches ``run_gomb_smoke``):

* 80/10/10 edge split (``datasets.split``), train-only message passing — the
  strategy's ``build_context`` is handed ``e_tr, s_tr`` *exclusively*, so no
  test-edge sign reaches the encoder's adjacency.
* ``--shuffle-train-signs`` permutes the training signs with ``rng(seed+100003)``
  before context construction *and* BCE targets consume the shuffled view; test
  signs are untouched. A clean baseline's shuffled AUROC falls to chance.

Contract preserved verbatim: the final stdout line is a single JSON object with
``test_auroc`` and ``n_params``, so ``run_no_leak_benchmark._parse_result`` needs
no change.

CLI
---
    python -m signedkan_wip.experiments.runs.run_baseline_audit \
        --model sgcn --dataset bitcoin_alpha --seed 0 --n-epochs 120 \
        [--shuffle-train-signs] [--hidden H --n-layers L --n-heads Hd --lr LR]
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from signedkan_wip.src.baselines.reachability import ReachabilityRule, reachable_edges
from signedkan_wip.src.baselines.registry import (
    GraphMeta,
    HParams,
    get_baseline,
    list_baselines,
)
from signedkan_wip.src.datasets import load, split

SHUFFLE_SEED_OFFSET = 100003  # identical to run_gomb_smoke, for comparability


def _evaluate(
    model: torch.nn.Module,
    ctx: tuple,
    edges: np.ndarray,
    signs: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    """Held-out (AUROC, macro-F1). AUROC is ``nan`` if the slice is one-class."""
    model.eval()
    with torch.no_grad():
        z = model.encode_nodes(*ctx)
        edges_t = torch.from_numpy(edges.astype(np.int64)).to(device)
        logits = model.edge_logits(z, edges_t).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y = (signs == 1).astype(int)
    auc = (roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m)


def run_audit(
    model_name: str,
    dataset: str,
    seed: int,
    n_epochs: int | None = None,
    *,
    shuffle_train_signs: bool = False,
    reachability: ReachabilityRule = ReachabilityRule.STRICT,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one (model, dataset, seed) cell under a reachability rule.

    Preconditions: ``model_name`` is a registered baseline; ``dataset`` loads via
    ``datasets.load``; ``seed >= 0``.
    Postconditions: dict carries ``test_auroc`` (held-out, may be ``nan`` on a
    one-class slice) and ``n_params``. The encoder's message-passing context is
    seeded by ``reachability``: STRICT (default) = training edges only, so the
    result is bit-identical to the prior strict-only behaviour; TRANSDUCTIVE_*
    additionally admit test-edge topology / signs (the leak experiment).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strat = get_baseline(model_name)
    hp: HParams = strat.default_hparams().merged(
        n_epochs=n_epochs, **(overrides or {})
    )

    g = load(dataset)
    tr_idx, va_idx, te_idx = split(g, seed=seed)
    # Strict invariant: train/test edge index sets are disjoint (split contract);
    # the context is built from train edges only, so no test-edge sign leaks.
    assert not (set(tr_idx.tolist()) & set(te_idx.tolist())), \
        "train/test edge overlap — strict protocol violated"

    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx].copy()
    e_va, s_va = g.edges[va_idx], g.signs[va_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    if shuffle_train_signs:
        perm = np.random.default_rng(seed + SHUFFLE_SEED_OFFSET).permutation(len(s_tr))
        s_tr = s_tr[perm]

    # Reachability rule seeds the message-passing closure. STRICT → (e_tr, s_tr)
    # exactly (reduction); TRANSDUCTIVE_* admit test topology/signs. Built from
    # the post-shuffle train signs so the audit semantics compose.
    adj_edges, adj_signs = reachable_edges(reachability, e_tr, s_tr, e_te, s_te)
    meta = GraphMeta(n_nodes=g.n_nodes)
    ctx = strat.build_context(adj_edges, adj_signs, g.n_nodes, device)
    model = strat.build_model(meta, hp).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=hp.lr,
                           weight_decay=hp.weight_decay)

    e_tr_t = torch.from_numpy(e_tr.astype(np.int64)).to(device)
    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    s_tr_t = torch.from_numpy(s_tr.astype(np.float32)).to(device)
    if hp.class_weighted:
        n_pos = int((s_tr == 1).sum())
        n_neg = int((s_tr == -1).sum())
        pos_weight = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                                  device=device)
    else:
        pos_weight = None

    best_val_auc, best_state, best_epoch = -1.0, None, -1
    stale = 0  # consecutive val checks without improvement (patience early-exit)
    t0 = time.time()
    for epoch in range(hp.n_epochs):
        model.train()
        z = model.encode_nodes(*ctx)
        logits = model.edge_logits(z, e_tr_t)
        loss = F.binary_cross_entropy_with_logits(
            logits, target_tr, pos_weight=pos_weight)
        if hp.aux_weight > 0.0:
            loss = loss + hp.aux_weight * model.aux_loss(z, e_tr_t, s_tr_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if hp.early_stopping and ((epoch + 1) % hp.val_every == 0
                                  or epoch == hp.n_epochs - 1):
            v_auc, _ = _evaluate(model, ctx, e_va, s_va, device)
            if v_auc > best_val_auc:
                best_val_auc, best_epoch, stale = v_auc, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                stale += 1
                # Opt-in speed/accuracy tradeoff: exiting after ``patience`` stale
                # checks skips plateau epochs but CAN miss a late improvement
                # (measured: sgcl best_epoch 185/200), so the Table-1 grid leaves
                # it off and trains full n_epochs. Use only for exploratory runs.
                if hp.patience is not None and stale >= hp.patience:
                    break
    elapsed = time.time() - t0

    if hp.early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    test_auc, test_f1m = _evaluate(model, ctx, e_te, s_te, device)
    return dict(
        model=model_name, dataset=dataset, seed=seed,
        shuffle=shuffle_train_signs, reachability=reachability.value,
        n_epochs=hp.n_epochs,
        hidden=hp.hidden, n_layers=hp.n_layers, n_heads=hp.n_heads, lr=hp.lr,
        n_params=int(n_params), elapsed_s=round(elapsed, 2),
        best_epoch=best_epoch, best_val_auroc=round(best_val_auc, 4),
        edges_per_s=round(len(e_te) / max(elapsed, 1e-9), 1),
        test_auroc=round(test_auc, 4), test_f1_macro=round(test_f1m, 4),
        device=str(device),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help=f"one of {list_baselines()}")
    ap.add_argument("--dataset", default="bitcoin_alpha")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-epochs", type=int, default=120)
    ap.add_argument("--shuffle-train-signs", action="store_true",
                    help="permute train signs (audit control); test intact")
    ap.add_argument("--reachability", default="strict",
                    choices=[r.value for r in ReachabilityRule],
                    help="message-passing closure: strict|topo|full (default strict)")
    # Optional recipe overrides (the driver pins width/depth per method).
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="val checks w/o improvement before early exit (grid use)")
    args = ap.parse_args()

    overrides = {k: v for k, v in (
        ("hidden", args.hidden), ("n_layers", args.n_layers),
        ("n_heads", args.n_heads), ("lr", args.lr),
        ("patience", args.patience),
    ) if v is not None}

    r = run_audit(args.model, args.dataset, args.seed,
                  n_epochs=args.n_epochs,
                  shuffle_train_signs=args.shuffle_train_signs,
                  reachability=ReachabilityRule.from_str(args.reachability),
                  overrides=overrides)
    print(f"  {args.model:<10} {args.dataset:<14} seed={args.seed} "
          f"shuffle={args.shuffle_train_signs}  "
          f"AUROC={r['test_auroc']:.4f}  F1m={r['test_f1_macro']:.4f}  "
          f"params={r['n_params']:,}  {r['elapsed_s']:.1f}s")
    print(json.dumps(r))  # last line: parsed by run_no_leak_benchmark


if __name__ == "__main__":
    main()
