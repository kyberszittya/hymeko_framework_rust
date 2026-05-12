"""Sinusoid-control baselines for the §III.G "~91% sinusoidal" claim.

A reviewer can object: "any smooth function on $[-1, 1]$ sampled on a
small grid can be misclassified as sinusoidal by a 4-parameter sine
fit." This script answers that objection with three null-baselines:

  1. trained HSiKAN          — the published claim
  2. untrained HSiKAN        — random init, no gradient steps
  3. random spline coefs     — fresh ~N(0, init_scale) coef tensors,
                                same parametric family, no model
  4. Gaussian-process draws  — smooth functions of comparable bandwidth
                                that don't go through the spline basis
                                at all

The output is a 4-row table of sinusoidal-fraction (and full symbolic
histogram) per baseline, plus residual-MSE statistics. If trained
HSiKAN's sinusoidal-fraction is materially higher than all three nulls,
the Fourier-style decomposition claim defends itself.

Output: signedkan_wip/experiments/results/sinusoid_controls.json
        + a tiny markdown summary printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from .datasets import load, split
from .hyperedges import construct
from .signedkan import SignedKAN, SignedKANConfig
from .train import build_edge_to_triads
from .run_compare import build_edge_incidence
from .prune_distill import (
    distill_activation,
    fit_summary,
    fit_symbolic,
    sample_spline_activation,
)


# ─── helpers ─────────────────────────────────────────────────────────


def _evaluate(model, triad_v, triad_sigma, signs, M, device):
    model.eval()
    with torch.no_grad():
        triad_emb = model.encode_triads(triad_v.to(device),
                                          triad_sigma.to(device))
        edge_emb = torch.sparse.mm(M, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1).cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (signs == 1).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    f1m = f1_score(y, (probs > 0.5).astype(int), average="macro",
                   zero_division=0)
    return float(auc), float(f1m)


def _build_model(g, hidden, spline_kind, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cfg = SignedKANConfig(n_nodes=g.n_nodes, hidden_dim=hidden,
                            grid=5, k=3, spline_kind=spline_kind)
    return SignedKAN(cfg).to(device)


def _train(model, triad_v_dev, triad_sigma_dev, M_train, target_tr,
           pos_w, n_epochs, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(n_epochs):
        model.train()
        triad_emb = model.encode_triads(triad_v_dev, triad_sigma_dev)
        edge_emb = torch.sparse.mm(M_train, triad_emb)
        logits = model.classifier(edge_emb).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            logits, target_tr, pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()


def _resample_coefs_(activation, init_scale=0.1, seed=0):
    """Fresh ~N(0, init_scale) coefs in place. Same shape as the
    activation's existing tensor."""
    g = torch.Generator(device=activation.coef.device)
    g.manual_seed(seed)
    new = torch.randn(activation.coef.shape,
                       device=activation.coef.device,
                       generator=g) * init_scale
    activation.coef.data.copy_(new)


def _gp_draws(n_curves: int, n_samples: int = 200,
              length_scale: float = 0.3, sigma: float = 1.0,
              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw smooth functions on $[-1, 1]$ from a centred Gaussian
    process with a squared-exponential kernel. Returns ``(x, ys)``
    with ``ys`` shape ``(n_curves, n_samples)``.

    length_scale 0.3 + n_samples 200 over $[-1, 1]$ gives curves with
    1-3 zero-crossings on the interval — comparable wiggliness to
    grid-5 splines. We add a tiny diagonal jitter so the Cholesky
    decomposition stays stable."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_samples)
    diff = x[:, None] - x[None, :]
    K = sigma ** 2 * np.exp(-0.5 * (diff / length_scale) ** 2)
    K += 1e-6 * np.eye(n_samples)
    L = np.linalg.cholesky(K)
    z = rng.standard_normal((n_samples, n_curves))
    ys = (L @ z).T
    return x, ys


def _fit_curves(x: np.ndarray, ys: np.ndarray) -> Counter:
    """Bulk-fit a stack of $y$ samples to the symbolic library, return
    a histogram of best forms."""
    h: Counter = Counter()
    for i in range(ys.shape[0]):
        fit = fit_symbolic(x, ys[i])
        h[fit.form] += 1
    return h


def _hist_from_activation(activation) -> Counter:
    """Distill all (branch, channel) of an activation, return histogram."""
    fits = distill_activation(activation)
    return Counter(fit_summary(fits))


def _normalise(hist: Counter) -> dict:
    n = sum(hist.values())
    return {k: round(v / n, 4) for k, v in hist.items()} if n else {}


# ─── main ────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["bitcoin_alpha"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--spline-kind", default="catmull_rom",
                    choices=["bspline", "catmull_rom"])
    ap.add_argument("--n-gp-draws", type=int, default=400,
                    help="number of Gaussian-process baseline curves")
    ap.add_argument("--out", default=
                    "signedkan_wip/experiments/results/sinusoid_controls.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    for dataset in args.datasets:
        for seed in args.seeds:
            print(f"\n=== {dataset}  seed={seed} ===")

            # Bench plumbing — once per (dataset, seed).
            torch.manual_seed(seed); np.random.seed(seed)
            g = load(dataset)
            triads = construct(g)
            triad_v = torch.tensor([t.v for t in triads], dtype=torch.long)
            triad_sigma = torch.tensor([t.sigma for t in triads],
                                        dtype=torch.long)
            edge_to_triads = build_edge_to_triads(triads)
            tr_idx, va_idx, te_idx = split(g, seed=seed)
            e_tr, s_tr = g.edges[tr_idx], g.signs[tr_idx]
            e_te, s_te = g.edges[te_idx], g.signs[te_idx]
            n_triads = triad_v.shape[0]
            M_train = build_edge_incidence(e_tr, edge_to_triads, n_triads,
                                            device)
            M_test  = build_edge_incidence(e_te, edge_to_triads, n_triads,
                                            device)
            triad_v_dev = triad_v.to(device)
            triad_sigma_dev = triad_sigma.to(device)
            target_tr = torch.from_numpy(
                (s_tr == 1).astype(np.float32)).to(device)
            n_pos = int((s_tr ==  1).sum())
            n_neg = int((s_tr == -1).sum())
            pos_w = torch.tensor(float(max(n_neg, 1)) / float(max(n_pos, 1)),
                                  device=device)

            # ── Baseline 1 — trained HSiKAN ─────────────────────────
            print("  [trained]")
            model_t = _build_model(g, args.hidden, args.spline_kind,
                                    device, seed)
            t0 = time.time()
            _train(model_t, triad_v_dev, triad_sigma_dev, M_train,
                   target_tr, pos_w, args.n_epochs, args.lr, device)
            tr_time = time.time() - t0
            auc_t, f1_t = _evaluate(model_t, triad_v, triad_sigma,
                                     s_te, M_test, device)
            inner_t = _hist_from_activation(model_t.layer.inner)
            outer_t = _hist_from_activation(model_t.layer.outer)
            combined_t = inner_t + outer_t
            print(f"    trained ({tr_time:.1f}s)  AUC={auc_t:.4f}  "
                  f"F1m={f1_t:.4f}")
            print(f"    inner: {dict(inner_t)}")
            print(f"    outer: {dict(outer_t)}")

            # ── Baseline 2 — untrained HSiKAN ───────────────────────
            print("  [untrained]")
            model_u = _build_model(g, args.hidden, args.spline_kind,
                                    device, seed + 10000)
            inner_u = _hist_from_activation(model_u.layer.inner)
            outer_u = _hist_from_activation(model_u.layer.outer)
            combined_u = inner_u + outer_u
            print(f"    inner: {dict(inner_u)}")
            print(f"    outer: {dict(outer_u)}")

            # ── Baseline 3 — random spline coefs ────────────────────
            print("  [random_coefs]")
            model_r = _build_model(g, args.hidden, args.spline_kind,
                                    device, seed + 20000)
            _resample_coefs_(model_r.layer.inner, init_scale=0.1,
                              seed=seed + 30000)
            _resample_coefs_(model_r.layer.outer, init_scale=0.1,
                              seed=seed + 31000)
            inner_r = _hist_from_activation(model_r.layer.inner)
            outer_r = _hist_from_activation(model_r.layer.outer)
            combined_r = inner_r + outer_r
            print(f"    inner: {dict(inner_r)}")
            print(f"    outer: {dict(outer_r)}")

            # ── Baseline 4 — Gaussian-process smooth-function draws ─
            print(f"  [gp_draws  n={args.n_gp_draws}]")
            x_gp, ys_gp = _gp_draws(args.n_gp_draws,
                                      seed=seed + 40000)
            gp_hist = _fit_curves(x_gp, ys_gp)
            print(f"    gp:   {dict(gp_hist)}")

            rows.append(dict(
                dataset=dataset, seed=seed,
                hidden=args.hidden, spline_kind=args.spline_kind,
                trained=dict(
                    auc=auc_t, f1m=f1_t,
                    inner_hist=dict(inner_t),
                    outer_hist=dict(outer_t),
                    combined_hist=dict(combined_t),
                    combined_norm=_normalise(combined_t),
                ),
                untrained=dict(
                    inner_hist=dict(inner_u),
                    outer_hist=dict(outer_u),
                    combined_hist=dict(combined_u),
                    combined_norm=_normalise(combined_u),
                ),
                random_coefs=dict(
                    inner_hist=dict(inner_r),
                    outer_hist=dict(outer_r),
                    combined_hist=dict(combined_r),
                    combined_norm=_normalise(combined_r),
                ),
                gp_draws=dict(
                    n=args.n_gp_draws,
                    hist=dict(gp_hist),
                    norm=_normalise(gp_hist),
                ),
            ))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}  ({len(rows)} rows)")

    # Pretty cross-baseline summary, averaged over seeds.
    def _avg_sin(rows_subset, key):
        fracs = []
        for r in rows_subset:
            cn = r[key].get("combined_norm") if key != "gp_draws" \
                  else r[key].get("norm")
            if cn:
                fracs.append(cn.get("sine", 0.0))
        return sum(fracs) / len(fracs) if fracs else 0.0

    print("\n── Sinusoidal-fraction summary (mean over seeds) ──")
    for ds in args.datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        print(f"  {ds:>16s}  trained={_avg_sin(ds_rows, 'trained'):.3f}  "
              f"untrained={_avg_sin(ds_rows, 'untrained'):.3f}  "
              f"random_coefs={_avg_sin(ds_rows, 'random_coefs'):.3f}  "
              f"gp_draws={_avg_sin(ds_rows, 'gp_draws'):.3f}")


if __name__ == "__main__":
    main()
