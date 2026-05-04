"""Cycle-pruning study for HSiKAN-mixed on Slashdot.

Question: how many of the trained model's cycles are *load-bearing*?

Procedure
---------
1. Train HSiKAN-mixed once at a known-good config (h=16, max_k4=200k,
   L=2, seed=0). 60 epochs, no early stopping, same recipe as the
   sweep so the result is comparable.
2. Score every cycle by its expected contribution to validation
   predictions:

       score[a, t] = α[a] · ‖M_e_val[:, t]‖₁ · ‖h_t_final[t]‖₂

   This is the L2 norm of cycle ``t``'s aggregate contribution to the
   validation edge embeddings, weighted by the learned arity weight.
3. For ``prune_frac ∈ {1.0, 0.5, 0.25, 0.10, 0.05, 0.01}`` (per arity)
   keep the top-K cycles by score, rebuild the per-arity sparse
   incidences on the subset, and re-evaluate test AUC + inference
   wall-clock. **No retraining.**
4. Emit a JSON with the AUC vs cycle-count curve.

Caveats
-------
- The score is a single forward pass on val; it doesn't account for
  cycle-cycle interactions through the layered vertex update. A more
  faithful score would be leave-one-out, but that's O(T) more
  forwards and we don't need that fidelity here.
- We retain the same prune fraction PER ARITY. αₖ already
  encodes "more cycles in arity k=4"; pruning within an arity asks
  whether the cycles selected for k=4 are equally important.
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
from .n_tuples import construct_k
from .mixed_arity_signedkan import (MixedAritySignedKAN,
                                      MixedAritySignedKANConfig,
                                      subsample_tuples,
                                      build_edge_to_tuples)
from .signedkan import (MultiLayerSignedKANConfig,
                         build_vertex_triad_incidence)
from .run_phase2_mixed_arity import _build_edge_incidence


def _build_per_arity_inputs(g, edges_array, arities, max_per, device, seed):
    per_arity_tuples = []
    for k in arities:
        cap = max_per.get(k)
        if k == 3:
            t_k = construct(g)
        else:
            t_k = construct_k(g, k=k, max_cycles=cap, seed=seed)
        if cap and len(t_k) > cap:
            t_k = subsample_tuples(t_k, cap, seed=seed)
        per_arity_tuples.append(t_k)

    per_arity_inputs = []
    for ai, k in enumerate(arities):
        tuples = per_arity_tuples[ai]
        triad_v_np = np.array([t.v for t in tuples], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in tuples], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        edge_to_tuples = build_edge_to_tuples(tuples)
        M_vt = build_vertex_triad_incidence(
            triad_v_np, g.n_nodes, device, mode="sum",
        )
        M_e = _build_edge_incidence(edges_array, edge_to_tuples,
                                      len(tuples), device)
        per_arity_inputs.append((triad_v, triad_sigma, M_vt, M_e))
    return per_arity_inputs, per_arity_tuples


def _pruned_inputs(per_arity_tuples, keep_idx_per_arity,
                    g, edges_array, device):
    """Build per_arity_inputs over a subset of cycles per arity."""
    per_arity_inputs = []
    sub_tuples_per_arity = []
    for ai, (tuples, keep_idx) in enumerate(zip(per_arity_tuples,
                                                  keep_idx_per_arity)):
        sub = [tuples[int(i)] for i in keep_idx]
        sub_tuples_per_arity.append(sub)
        triad_v_np = np.array([t.v for t in sub], dtype=np.int64)
        triad_sigma_np = np.array([t.sigma for t in sub], dtype=np.int64)
        triad_v = torch.from_numpy(triad_v_np).to(device)
        triad_sigma = torch.from_numpy(triad_sigma_np).to(device)
        edge_to_tuples = build_edge_to_tuples(sub)
        M_vt = build_vertex_triad_incidence(
            triad_v_np, g.n_nodes, device, mode="sum",
        )
        M_e = _build_edge_incidence(edges_array, edge_to_tuples,
                                      len(sub), device)
        per_arity_inputs.append((triad_v, triad_sigma, M_vt, M_e))
    return per_arity_inputs


def _score_cycles_per_arity(model, per_arity_inputs_val, alpha):
    """Per-cycle val-set contribution score, one tensor per arity."""
    cfg = model.cfg.base
    n_layers = cfg.n_layers
    layer = model.base.shared_layer
    h_v = model.node_embed.weight
    n_arities = len(model.cfg.arities)
    jk = cfg.jk_mode

    # Reproduce _encode_edges_full's intermediates per arity.
    per_arity_per_layer_t = [[] for _ in range(n_arities)]
    h_v_curr = h_v
    for li in range(n_layers):
        arity_h_t = []
        for triad_v, triad_sigma, _M_vt, _M_e in per_arity_inputs_val:
            with torch.no_grad():
                h_t = layer(h_v_curr, triad_v, triad_sigma)
            arity_h_t.append(h_t)
        for ai, h_t in enumerate(arity_h_t):
            per_arity_per_layer_t[ai].append(h_t)
        if li < n_layers - 1:
            h_v_step = torch.zeros_like(h_v_curr)
            for ai, h_t in enumerate(arity_h_t):
                M_vt = per_arity_inputs_val[ai][2]
                with torch.no_grad():
                    h_v_step = h_v_step + alpha[ai] * torch.sparse.mm(M_vt, h_t)
            h_v_curr = ((h_v_curr + h_v_step) if cfg.use_residual
                         else h_v_step)
            if model.base.layer_norms is not None:
                with torch.no_grad():
                    h_v_curr = model.base.layer_norms[li](h_v_curr)

    scores_per_arity = []
    for ai in range(n_arities):
        stack = per_arity_per_layer_t[ai]
        if jk == "last":
            h_final = stack[-1]
        elif jk == "sum":
            h_final = torch.stack(stack, dim=0).sum(dim=0)
        elif jk == "concat":
            h_final = torch.cat(stack, dim=-1)
        else:
            raise ValueError(jk)
        # ‖h_t_final[t]‖₂  per cycle.
        h_norm = h_final.norm(dim=-1)                       # (T,)
        # ‖M_e_val[:, t]‖₁  per cycle.
        M_e = per_arity_inputs_val[ai][3]
        idx = M_e._indices()
        val = M_e._values()
        cols = idx[1]
        T_a = M_e.shape[1]
        col_l1 = torch.zeros(T_a, device=h_v.device, dtype=val.dtype)
        col_l1.index_add_(0, cols, val.abs())
        score = float(alpha[ai].item()) * col_l1 * h_norm
        scores_per_arity.append(score)
    return scores_per_arity


def _evaluate(model, per_arity_inputs, edges, signs, device):
    model.eval()
    with torch.no_grad():
        edge_emb = model.encode_edges(per_arity_inputs)
        logits = model.classifier(edge_emb).squeeze(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    y = (signs == 1).astype(int)
    auc = (roc_auc_score(y, probs)
            if len(np.unique(y)) > 1 else float("nan"))
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    return float(auc), float(f1m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/phase7_slashdot_pruning.json")
    ap.add_argument("--n_epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--max_k3", type=int, default=30_000)
    ap.add_argument("--max_k4", type=int, default=200_000)
    ap.add_argument("--cycle_batch_size", type=int, default=10_000)
    ap.add_argument("--prune_fracs", nargs="+", type=float,
                    default=[1.0, 0.5, 0.25, 0.10, 0.05, 0.02, 0.01])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  hidden={args.hidden}  L={args.n_layers}  "
          f"max_k4={args.max_k4:,}  seed={args.seed}")

    g = load("slashdot")
    tr_idx, va_idx, te_idx = split(g, seed=args.seed)
    e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
    e_va, s_va = g.edges[va_idx], g.signs[va_idx]
    e_te, s_te = g.edges[te_idx], g.signs[te_idx]

    arities = (3, 4)
    max_per = {3: args.max_k3, 4: args.max_k4}

    print("Building per-arity inputs (train) ...")
    t0 = time.time()
    per_arity_train, per_arity_tuples = _build_per_arity_inputs(
        g, e_tr, arities, max_per, device, args.seed,
    )
    print(f"  train inputs: {time.time()-t0:.1f}s  "
          f"sizes={[len(t) for t in per_arity_tuples]}")
    # val/test inputs are built lazily AFTER training releases its
    # autograd state, to keep peak memory low on Slashdot k=4.

    # Fresh model — same recipe as the sweep cells.
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = MixedAritySignedKANConfig(
        base=MultiLayerSignedKANConfig(
            n_nodes=g.n_nodes, n_layers=args.n_layers,
            hidden_dim=args.hidden, grid=3, k=3,
            spline_kinds=["catmull_rom"] * args.n_layers,
            init_scale=0.05,
            pool_mode="sum",
            jk_mode="concat",
            layer_norm_between=True,
            share_weights=True,
            inner_skip="highway",
            outer_skip="none",
            use_residual=True,
        ),
        arities=arities,
        init_arity_logits=tuple([0.0] * len(arities)),
        cycle_batch_size=args.cycle_batch_size,
    )
    model = MixedAritySignedKAN(cfg).to(device)

    # Train.
    target_tr = torch.from_numpy((s_tr == 1).astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=0.0)
    print(f"\nTraining {args.n_epochs} epochs ...")
    t0 = time.time()
    for ep in range(args.n_epochs):
        model.train()
        edge_emb = model.encode_edges(per_arity_train)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, target_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:>3d}  train_loss={loss.item():.4f}")
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s")

    # Free training-time autograd state before building val/test.
    opt.zero_grad(set_to_none=True)
    del edge_emb, logits, loss, target_tr
    import gc; gc.collect()

    print("Building val/test inputs ...")
    t0 = time.time()
    per_arity_val, _ = _build_per_arity_inputs(
        g, e_va, arities, max_per, device, args.seed,
    )
    per_arity_test, _ = _build_per_arity_inputs(
        g, e_te, arities, max_per, device, args.seed,
    )
    print(f"  val+test inputs: {time.time()-t0:.1f}s")

    full_test_auc, full_test_f1 = _evaluate(
        model, per_arity_test, e_te, s_te, device,
    )
    alpha = model.alpha().detach().cpu()
    print(f"\nFull-cycle test AUC = {full_test_auc:.4f}  "
          f"F1m = {full_test_f1:.4f}  alpha = {alpha.tolist()}")

    # Score cycles on val.
    print("\nScoring cycles on validation set ...")
    t0 = time.time()
    scores = _score_cycles_per_arity(model, per_arity_val,
                                       model.alpha().detach())
    print(f"  scoring: {time.time()-t0:.1f}s  "
          f"scores per arity: {[s.shape[0] for s in scores]}")

    # Sort each arity's cycles by descending score.
    sort_idx_per_arity = [
        torch.argsort(s, descending=True) for s in scores
    ]

    # Switch model to non-batched encode_edges for honest inference timing.
    model.cfg.cycle_batch_size = None

    # Sweep prune levels.
    rows = []
    for frac in args.prune_fracs:
        keep_per_arity = [
            int(max(1, round(frac * len(t)))) for t in per_arity_tuples
        ]
        keep_idx_per_arity = [
            sort_idx_per_arity[ai][:k].cpu().numpy()
            for ai, k in enumerate(keep_per_arity)
        ]
        per_arity_test_pruned = _pruned_inputs(
            per_arity_tuples, keep_idx_per_arity,
            g, e_te, device,
        )
        # Inference timing: 5 warm-up forwards + 10 timed forwards.
        for _ in range(5):
            with torch.no_grad():
                _ = model.encode_edges(per_arity_test_pruned)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        N_TIMED = 10
        for _ in range(N_TIMED):
            with torch.no_grad():
                _ = model.encode_edges(per_arity_test_pruned)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.time() - t0) / N_TIMED * 1000.0

        auc, f1m = _evaluate(model, per_arity_test_pruned,
                              e_te, s_te, device)
        row = {
            "frac": frac,
            "keep_per_arity": keep_per_arity,
            "test_auc": auc,
            "test_f1_macro": f1m,
            "infer_ms": infer_ms,
        }
        rows.append(row)
        print(f"  frac={frac:.2f}  keep={keep_per_arity}  "
              f"AUC={auc:.4f}  F1m={f1m:.4f}  infer={infer_ms:.1f}ms")

    out = {
        "config": dict(
            seed=args.seed, hidden=args.hidden, n_layers=args.n_layers,
            max_k3=args.max_k3, max_k4=args.max_k4,
            n_epochs=args.n_epochs,
            cycle_batch_size=args.cycle_batch_size,
        ),
        "trained_alpha": alpha.tolist(),
        "trained_test_auc": full_test_auc,
        "trained_test_f1_macro": full_test_f1,
        "train_time_s": train_time,
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
